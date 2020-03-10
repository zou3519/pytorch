#include <ATen/Batching.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <ATen/WrapDimUtils.h>

#include <torch/jit.h>
#include <torch/script.h>
#include <torch/types.h>

namespace at {

int64_t BatchTensorImpl::batch_size() const {
  return rep_.sizes()[batch_dim_];
}


bool isBatchTensor(const Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(BatchTensorKey);
}

int64_t maxLevel(const std::vector<Tensor>& maybeBatchTensors) {
  int64_t max = -1;
  auto it = maybeBatchTensors.begin();
  auto end_it = maybeBatchTensors.end();
  while (it != end_it) {
    it = std::find_if(it, end_it, isBatchTensor);
    if (it != end_it) {
      const auto* batchTensor = static_cast<const BatchTensorImpl*>(it->unsafeGetTensorImpl());
      if (batchTensor->level_ > max) {
        max = batchTensor->level_;
      }
      it++;
    }
  }
  return max;
}

std::pair<Tensor,optional<int64_t>> unwrapAtLevel(const Tensor& tensor, int64_t level) {
  if (!isBatchTensor(tensor)) {
    return { tensor, nullopt };
  }
  auto* batch_tensor = getBatched(tensor);
  if (batch_tensor->level_ != level) {
    TORCH_INTERNAL_ASSERT(batch_tensor->level_ < level);
    return { tensor, nullopt };
  }
  return { batch_tensor->rep_, batch_tensor->batch_dim_ };
}

Tensor broadcastTo(const Tensor& tensor, int64_t ndim) {
  auto old_sizes = tensor.sizes();
  if (old_sizes.size() == ndim) {
    return tensor;
  }
  TORCH_INTERNAL_ASSERT(old_sizes.size() <= ndim);
  // TODO: This is really slow, we should probably write a new operator for
  // this. Note that we can't call view because it is not "semantic" enough.
  // It might be possible to just call reshape here.
  int64_t diff = ndim - old_sizes.size();
  Tensor result = tensor;
  for (int64_t i = 0; i < diff; ++i) {
    result = result.unsqueeze(0);
  }
  return result;
}

Tensor moveBatchDimToFront(
    const Tensor& tensor,
    optional<int64_t> batch_dim,
    int64_t result_dim) {
  if (!batch_dim) {
    return broadcastTo(tensor, result_dim);
  }
  auto bdim = *batch_dim;
  auto extra_dims = result_dim - tensor.dim();
  auto result = broadcastTo(tensor, result_dim);
  auto actual_bdim = bdim + extra_dims;
  if (actual_bdim == 0) {
    return result;
  }
  // should be an op...
  std::vector<int64_t> permutation(result_dim);
  permutation[0] = actual_bdim;
  for (int64_t i = 1; i < result_dim; i++) {
    if (i <= actual_bdim) {
      permutation[i] = i - 1;
    } else {
      permutation[i] = i;
    }
  }
  result = result.permute(permutation);
  return result;
}

int64_t actualDim(int64_t dim, optional<int64_t> maybe_batch_dim) {
  if (maybe_batch_dim && dim >= *maybe_batch_dim) {
    return dim + 1;
  }
  return dim;
}

std::tuple<int64_t,int64_t>
discoverBatchSizeAndLevel(torch::jit::Stack* stack) {
  int64_t max_level = -1;
  int64_t batch_size = -1;
  for (auto& ivalue : *stack) {
    if (!ivalue.isTensor()) continue;
    auto tensor = ivalue.toTensor();
    if (!isBatched(tensor)) continue;
    auto* batched = getBatched(tensor);
    if (batched->level_ > max_level) {
      max_level = batched->level_;
      // TODO: should probably validate somewhere that the batch sizes are the same
      batch_size = batched->batch_size();
    }
  }
  TORCH_INTERNAL_ASSERT(batch_size != -1);
  TORCH_INTERNAL_ASSERT(max_level != -1);
  return { batch_size, max_level };
}

int64_t minDim(const Tensor& tensor, optional<int64_t> batch_dim) {
  auto result = tensor.dim(); 
  if (!batch_dim) {
    result += 1;
  }
  return result;
}

// Copy pasta'ed from backed_fallback_test.cpp
void callBoxedWorkaround(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // This should just be op.callBoxed(stack), but that doesn't work for all ops yet.
  // Note: If op.callBoxed(stack) works for you, then that is preferrable because
  // it's much faster and doesn't come with a dependency on JIT code.
  // Instead, we take a path through the JIT operator registry, which has a boxed
  // calling mechanism that works for all ops from native_functions.yaml.

  auto s = Symbol::fromQualString(op.schema().name());
  auto operators = torch::jit::getAllOperatorsFor(s);
  // Find the exact match
  std::shared_ptr<torch::jit::Operator> jit_op;
  for (const auto& candidate_op : operators) {
    auto candidate_schema = candidate_op->schema();
    // NB: this is a VERY slow equality test
    if (candidate_schema == op.schema()) {
      jit_op = candidate_op;
      break;
    }
  }
  TORCH_INTERNAL_ASSERT(jit_op);

  auto offset = jit_op->getOperation()(*stack);
  TORCH_INTERNAL_ASSERT(offset == 0);
}

void batchTensorFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // const auto& schema = op.schema();
  // TORCH_CHECK(
  //     !schema.is_mutable() && !schema.hasAnyAliasInfo(),
  //     "Batching rule not implemented for ", schema(), "; ",
  //     "the fallback path doesn't work on in-place or view ops.");
  //
  TORCH_WARN("Batching rule not implemented for ", op.schema(), " falling back "
             "to slow (for-loop) implementation");
  TORCH_CHECK(std::all_of(op.schema().returns().begin(),
                          op.schema().returns().end(),
                          [] (const Argument& arg) { return arg.type() == TensorType::get(); }),
              "Batching rule not implemented for ", op.schema(), ". ",
              "We could not generate a fallback.");


  auto num_arguments = op.schema().arguments().size();
  auto num_returns = op.schema().returns().size();

  // Unwrap all arguments
  auto args = torch::jit::pop(*stack, num_arguments);

  int64_t batch_size;
  int64_t level;
  std::tie(batch_size, level) = discoverBatchSizeAndLevel(&args);

  // Set up batch_size stacks, one for each unbatched computation.
  std::vector<torch::jit::Stack> unbatched_stacks(batch_size);
  auto pushToEachStack = [&](const auto& ivalue) {
    for (auto& stack : unbatched_stacks) {
      torch::jit::push(stack, ivalue);
    }
  };
  for (const auto& ivalue : args) {
    if (!ivalue.isTensor()) {
      pushToEachStack(ivalue);
      continue;
    }
    const auto& tensor = ivalue.toTensor();
    if (!isBatched(tensor)) {
      pushToEachStack(tensor);
      continue;
    }
    const auto* batched = getBatched(tensor);
    if (batched->level_ != level) {
      pushToEachStack(ivalue);
      continue;
    }
    const auto tensors = batched->rep_.unbind(batched->batch_dim_);
    for (auto stack_idx = 0; stack_idx < batch_size; stack_idx++) {
      torch::jit::push(unbatched_stacks[stack_idx], tensors[stack_idx]);
    }
  }

  // Call the op on each stack.
  for (auto& stack : unbatched_stacks) {
    callBoxedWorkaround(op, &stack);
  }

  // Only support num_returns == 1 for now. Also assume Tensor returns
  TORCH_INTERNAL_ASSERT(num_returns == 1);
  std::vector<Tensor> tensors;
  for (const auto& stack : unbatched_stacks) {
    tensors.push_back(stack[0].toTensor());
  }
  auto stacked = at::stack(tensors);
  auto batched = makeBatched(stacked, 0, level);
  torch::jit::push(*stack, batched);
}

typedef std::pair<Tensor,optional<int64_t>> TensorAndBdim;

std::pair<TensorAndBdim,int64_t> unwrap(const Tensor& tensor) {
  auto* batch_tensor = getBatched(tensor);
  return { { batch_tensor->rep_, batch_tensor->batch_dim_ }, batch_tensor->level_ };
}

TensorAndBdim unsqueeze_batching_rule(const TensorAndBdim& self, int64_t dim) {
  const auto& tensor = self.first;
  const auto& maybe_batch_dim = self.second;

  dim = maybe_wrap_dim(dim, tensor.dim());
  auto actual_dim = actualDim(dim, maybe_batch_dim);
  optional<int64_t> new_batch_dim = nullopt;
  if (maybe_batch_dim) {
    auto bdim = *maybe_batch_dim;
    new_batch_dim = bdim < actual_dim ? bdim : bdim + 1;
  }
  return { tensor.unsqueeze(actual_dim), new_batch_dim };
}

std::shared_ptr<torch::jit::script::CompilationUnit> module;

void init_module() {
  module = torch::jit::compile(R"JIT(
def broadcast_to(tensor: Tensor, ndim: int):
    old_sizes = tensor.size()
    if len(old_sizes) == ndim:
        return tensor
    assert len(old_sizes) <= ndim
    diff = ndim - len(old_sizes)
    for i in range(diff):
        tensor = tensor.unsqueeze(0)
    return tensor

def move_batch_dim_to_front(tensor: Tensor,
                            batch_dim: Optional[int],
                            result_dim: int):
    if batch_dim is None:
        return broadcast_to(tensor, result_dim)
    extra_dims = result_dim - tensor.dim()
    result = broadcast_to(tensor, result_dim)
    transpose_dim = batch_dim + extra_dims
    if transpose_dim is None:
        return result
    return result.transpose(0, batch_dim + extra_dims)

def min_result_dim(tensor: Tensor, batch_dim: Optional[int]) -> int:
    result = tensor.dim()
    if batch_dim is None:
        result += 1
    return result

def mul_batching_rule(self: Tensor, self_bdim: Optional[int],
                      other: Tensor, other_bdim: Optional[int]):
    self_dim = min_result_dim(self, self_bdim)
    other_dim = min_result_dim(other, other_bdim)
    result_dim = max(self_dim, other_dim)

    self = move_batch_dim_to_front(self, self_bdim, result_dim)
    other = move_batch_dim_to_front(other, other_bdim, result_dim)
    return self * other, 0
)JIT");
}

std::shared_ptr<torch::jit::script::CompilationUnit> get_module() {
  if (!module) {
    init_module();
  }
  return module;
}

std::pair<Tensor,optional<int64_t>> conv2d_batching_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& weight, optional<int64_t> weight_bdim,
    const Tensor& bias, optional<int64_t> bias_bdim,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  if (weight_bdim) {
    TORCH_CHECK(false, "NYI: conv2d_batching_rule for batched weight");
  }
  if (bias_bdim) {
    TORCH_CHECK(false, "NYI: conv2d_batching_rule for batched bias");
  }
  auto result_dim = minDim(self, self_bdim);
  auto self_ = moveBatchDimToFront(self, self_bdim, result_dim);
  auto self_sizes = self_.sizes(); 

  auto self_4d = self_.flatten(0, 1);
  auto result_4d = at::conv2d(
      self_4d, weight, bias, stride, padding, dilation);
  return {
    result_4d.unflatten(0, {self_sizes.begin(), self_sizes.begin() + 2}),
    /*result_bdim=*/0
  };
}

// #define USE_TORCHSCRIPT_BATCHING_RULE

#ifndef USE_TORCHSCRIPT_BATCHING_RULE
std::pair<Tensor,optional<int64_t>> mul_batching_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& other, optional<int64_t> other_bdim) {
  auto self_dim = minDim(self, self_bdim);
  auto other_dim = minDim(other, other_bdim);
  auto result_dim = std::max({self_dim, other_dim});

  auto self_value = moveBatchDimToFront(self, self_bdim, result_dim);
  auto other_value = moveBatchDimToFront(other, other_bdim, result_dim);
  return { at::mul(self_value, other_value), 0 };
}
#else
std::pair<Tensor,optional<int64_t>> mul_batching_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& other, optional<int64_t> other_bdim) {
  auto result = get_module()->run_method(
      "mul_batching_rule",
      self, self_bdim, other, other_bdim).toTuple();
  auto res = result->elements()[0].toTensor();
  auto bdim = result->elements()[1].toOptional<int64_t>();
  return { res, bdim };
}
#endif

// TODO: it's not fine that we moved the batch dim,
// but that should be easy to fix.
std::pair<Tensor&,optional<int64_t>> mul__batching_rule(
    Tensor& self, optional<int64_t> self_bdim,
    const Tensor& other, optional<int64_t> other_bdim) {
  auto self_dim = minDim(self, self_bdim);
  auto other_dim = minDim(other, other_bdim);
  auto result_dim = std::max({self_dim, other_dim});

  // NB: Produces view
  auto self_value = moveBatchDimToFront(self, self_bdim, result_dim);
  auto other_value = moveBatchDimToFront(other, other_bdim, result_dim);

  // Probably want a nice error message here.
  self_value.mul_(other_value);
  return { self, 0 } ;
}

Tensor& BatchedTensor_mul_(Tensor& self, const Tensor& other) {
  // The following lines need to happen in each kernel
  auto cur_level = maxLevel({self, other});
  auto self_and_bdim = unwrapAtLevel(self, cur_level);
  auto other_and_bdim = unwrapAtLevel(other, cur_level);

  mul__batching_rule(
      self_and_bdim.first, self_and_bdim.second,
      other_and_bdim.first, other_and_bdim.second);
  return self;
}

Tensor BatchedTensor_conv2d(const Tensor& input, const Tensor& weight,
                            const Tensor& bias, IntArrayRef stride,
                            IntArrayRef padding, IntArrayRef dilation,
                            int64_t groups) {
  // The following lines need to happen in each kernel
  auto cur_level = maxLevel({input, weight, bias});
  auto input_and_bdim = unwrapAtLevel(input, cur_level);
  auto weight_and_bdim = unwrapAtLevel(weight, cur_level);
  auto bias_and_bdim = unwrapAtLevel(bias, cur_level);

  auto result_and_bdim = conv2d_batching_rule(
      input_and_bdim.first,
      input_and_bdim.second,
      weight_and_bdim.first,
      weight_and_bdim.second,
      bias_and_bdim.first,
      bias_and_bdim.second,
      stride, padding, dilation, groups);
  return makeBatched(
      result_and_bdim.first,
      result_and_bdim.second,
      cur_level);
}

// TODO: the fallback runs the un-batched kernel in a for loop.
// However, in many cases, operators are composed of other operators.
// If those operators have batched versions, then we don't need to
// run our for-loop-fallback. There should probably be some way to specify that.
auto registry = c10::Dispatcher::singleton().registerBackendFallbackKernel(
    BatchTensorKey,
    KernelFunction::makeFromBoxedFunction<&batchTensorFallback>()
);

static auto registry2 = torch::RegisterOperators()
  // Some operations need to be transformed to their batched versions
  .op(torch::RegisterOperators::options()
      .schema("aten::_make_batched(Tensor self, int? batch_dim, int level) -> Tensor")
      .kernel(BatchTensorKey, &at::native::_make_batched))
  .op(torch::RegisterOperators::options()
      .schema("aten::_unwrap_batched(Tensor self, int level) -> Tensor")
      .kernel(BatchTensorKey, &at::native::_unwrap_batched))
  .op(torch::RegisterOperators::options()
      .schema("aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)")
      .kernel(BatchTensorKey, [] (const Tensor& self, int64_t dim) -> Tensor {
        TensorAndBdim unwrapped;
        int64_t cur_level;
        std::tie(unwrapped, cur_level) = unwrap(self);
        auto result_with_batch = unsqueeze_batching_rule(unwrapped, dim);
        return makeBatched(
            result_with_batch.first,
            result_with_batch.second,
            cur_level);
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)")
      .kernel(BatchTensorKey, [] (const Tensor& self, int64_t dim0, int64_t dim1) -> Tensor {
        // TODO: don't forget to wrap dim0 & dim1
        auto* self_batched = getBatched(self);
        auto batch_dim = self_batched->batch_dim_;
        return makeBatched(
          self_batched->rep_.transpose(
            actualDim(dim0, batch_dim),
            actualDim(dim1, batch_dim)),
          batch_dim,
          self_batched->level_);
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::mul.Tensor(Tensor self, Tensor other) -> Tensor")
      .kernel(BatchTensorKey, [] (const Tensor& self, const Tensor& other) -> Tensor {
        // The following lines need to happen in each kernel
        auto cur_level = maxLevel({self, other});
        auto self_and_bdim = unwrapAtLevel(self, cur_level);
        auto other_and_bdim = unwrapAtLevel(other, cur_level);

        auto result_with_batch = mul_batching_rule(
            self_and_bdim.first, self_and_bdim.second,
            other_and_bdim.first, other_and_bdim.second);
        return makeBatched(
            result_with_batch.first,
            result_with_batch.second,
            cur_level);
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")
      .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), &BatchedTensor_mul_>(BatchTensorKey))
  .op(torch::RegisterOperators::options()
      .schema("aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
      .kernel(BatchTensorKey, [] (const Tensor& self, const Tensor& other, Scalar alpha) -> Tensor {
        // The following lines need to happen in each kernel
        auto level = maxLevel({self, other});
        auto self_unbatched = unwrapAtLevel(self, level);
        auto other_unbatched = unwrapAtLevel(other, level);

        auto self_dim = minDim(self_unbatched.first, self_unbatched.second);
        auto other_dim = minDim(other_unbatched.first, other_unbatched.second);
        auto result_dim = std::max({self_dim, other_dim});

        auto self_value = moveBatchDimToFront(
            self_unbatched.first,
            self_unbatched.second,
            result_dim);
        auto other_value = moveBatchDimToFront(
            other_unbatched.first,
            other_unbatched.second,
            result_dim);
        return makeBatched(
            at::add(self_value, other_value),
            0,  // since we moved the batchdim to the front
            level);
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::detach(Tensor self) -> (Tensor)")
      .kernel(BatchTensorKey, [] (const Tensor& self) -> Tensor {
        auto* batched = getBatched(self);
        return makeBatched(
            batched->rep_.detach(),
            batched->batch_dim_,
            batched->level_);
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::_is_batched(Tensor self) -> bool")
      .kernel(BatchTensorKey, [] (const Tensor& self) -> bool {
        return true;
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::_batch_dim(Tensor self) -> int")
      .kernel(BatchTensorKey, [] (const Tensor& self) -> int64_t {
        return native::_batch_dim(self); // wut
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::size.int(Tensor self, int dim) -> int")
      .kernel(BatchTensorKey, [] (const Tensor& self, int64_t dim) -> int64_t {
        dim = maybe_wrap_dim(dim, self.dim());
        return self.sizes()[dim];
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor")
      .impl_unboxedOnlyKernel<Tensor (const Tensor&, const Tensor&, const Tensor&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), &BatchedTensor_conv2d>(BatchTensorKey))
  // .op(torch::RegisterOperators::options()
  //     .schema("aten::view(Tensor(a) self, int[] size) -> Tensor(a)")
  //     .kernel(BatchTensorKey, [] (const Tensor& self, IntArrayRef size) -> Tensor {
  //       SmallVector<int64_t,5> new_sizes;
  //       new_sizes.reserve(size.size());
  //       auto* batched = getBatched(self);
  //       return makeBatched(
  //           batched->rep_.detach(),
  //           batched->batch_dim_,
  //           batched->level_);
  //     }))
  // I don't know how to override the following
  // .op(torch::RegisterOperators::options()
  //     .schema("aten::to.device(Tensor self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> (Tensor)")
  //     .kernel(BatchTensorKey, [] (const Tensor& self, Device device, ScalarType dtype, bool non_blocking, bool copy, optional<MemoryFormat> memory_format) -> Tensor {
  //       auto* batched = getBatched(self);
  //       return makeBatched(
  //           batched->rep_.to(device, dtype, non_blocking, copy, memory_format),
  //           batched->batch_dim_,
  //           batched->level_);
  //     }))
  ;

}
