#include <ATen/Batching.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/BatchingRules.h>

#include <torch/jit.h>
#include <torch/script.h>
#include <torch/types.h>
#include <bitset>
#include <iostream>
#include <utility>

namespace at {

#if !defined(C10_MOBILE) || defined(FEATURE_TORCH_MOBILE)

thread_local VmapState vmap_state;

VmapLevel enterVmapLevel(int64_t batch_size) {
  auto result = vmap_state.addLevel(batch_size);
  if (result == 1) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::VmapMode, true);
    std::cout << "Enter VmapMode" << std::endl;
  }
  return result;
}

int64_t exitVmapLevel() {
  auto result = vmap_state.popLevel();
  if (result.first == 1) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::VmapMode, false);
    std::cout << "Exit VmapMode" << std::endl;
  }
  return result.second;
}

VmapState* getVmapState() {
  return &vmap_state;
}

#else

void enterVmapLevel(int64_t batch_size) {
  throw std::runtime_error("vmap is not supported on mobile");
}

int64_t exitVmapLevel() {
  throw std::runtime_error("vmap is not supported on mobile");
}

VmapState* getVmapState() {
  return nullptr;
}


#endif




/*
 * Operator Registrations for BatchedTensorKey.
 * Contains some glue to hook up the batching rules to BatchedTensorImpl.
 */

void batchTensorFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

Tensor BatchedTensor_mul(const Tensor& self, const Tensor& other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  Tensor self_, other_, result_;
  BatchDimsRef self_bdims, other_bdims;
  BatchDims result_bdims;
  std::tie(self_, self_bdims) = unpackBatched(self);
  std::tie(other_, other_bdims) = unpackBatched(other);
  std::tie(result_, result_bdims) = mul_batching_rule(self_, self_bdims, other_, other_bdims);
  return detail::make_tensor<BatchTensorImpl>(result_, result_bdims);
}

Tensor BatchedTensor_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  Tensor self_, other_, result_;
  BatchDimsRef self_bdims, other_bdims;
  BatchDims result_bdims;
  std::tie(self_, self_bdims) = unpackBatched(self);
  std::tie(other_, other_bdims) = unpackBatched(other);
  std::tie(result_, result_bdims) = add_batching_rule(self_, self_bdims, other_, other_bdims, alpha);
  return detail::make_tensor<BatchTensorImpl>(result_, result_bdims);
}

Tensor BatchedTensor_conv2d(const Tensor& input, const Tensor& weight,
                            const Tensor& bias, IntArrayRef stride,
                            IntArrayRef padding, IntArrayRef dilation,
                            int64_t groups) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  Tensor input_, weight_, bias_;
  BatchDimsRef input_bdims, weight_bdims, bias_bdims;
  IntArrayRef unflatten_sizes;

  if (isBatched(weight) || isBatched(bias)) {
    auto maybe_ophandle = Dispatcher::singleton().findSchema(OperatorName("aten::conv2d", ""));
    TORCH_INTERNAL_ASSERT(maybe_ophandle.has_value());
    torch::jit::Stack stack = { input, weight, bias, stride, padding, dilation, groups };
    batchTensorFallback(*maybe_ophandle, &stack);
    return stack.back().toTensor();
  }

  std::tie(input_, input_bdims) = unpackBatched(input);
  std::tie(weight_, weight_bdims) = unpackBatched(weight);
  std::tie(bias_, bias_bdims) = unpackBatched(bias);

  Tensor result_;
  BatchDims result_bdims;
  std::tie(result_, result_bdims) = conv2d_batching_rule(
      input_, input_bdims,
      weight_, weight_bdims,
      bias_, bias_bdims,
      stride, padding, dilation, groups);
  return detail::make_tensor<BatchTensorImpl>(result_, result_bdims);
}

Tensor BatchedTensor_dropout(const Tensor& input, double p, bool train) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
	Tensor input_, result_;
	BatchDimsRef input_bdims;
	BatchDims result_bdims;

	std::tie(input_, input_bdims) = unpackBatched(input);
  std::tie(result_, result_bdims) = dropout_batching_rule(input_, input_bdims, p, train);
  return detail::make_tensor<BatchTensorImpl>(result_, result_bdims);
}

template <Tensor (*Op)(const Tensor&)>
Tensor BatchedTensor_unary_pw_op(const Tensor& input) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
	Tensor input_, result_;
	BatchDimsRef input_bdims;
	BatchDims result_bdims;

	std::tie(input_, input_bdims) = unpackBatched(input);
  std::tie(result_, result_bdims) = unary_pw_batching_rule<Op>(input_, input_bdims);
  return detail::make_tensor<BatchTensorImpl>(result_, result_bdims);
}

Tensor BatchedTensor_relu(const Tensor& input) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  return BatchedTensor_unary_pw_op<at::relu>(input);
}

template <Tensor& (Tensor::*Op)() const>
Tensor& BatchedTensor_unary_pw_inplace_op(Tensor& input) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
	Tensor input_;
	BatchDimsRef input_bdims;

	std::tie(input_, input_bdims) = unpackBatched(input);
	(input_.*Op)();
  return input;
}

template <typename F, F Func, typename... Args>
Tensor& BatchedTensor_unary_pw_inplace_fn_varargs(Tensor& input, Args... args) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
	Tensor input_;
	BatchDimsRef input_bdims;

	std::tie(input_, input_bdims) = unpackBatched(input);
	Func(input_, args...);
  return input;
}

template <typename F, F Func, typename... Args>
Tensor& BatchedTensor_unary_pw_inplace_meth_varargs(Tensor& input, Args... args) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
	Tensor input_;
	BatchDimsRef input_bdims;

	std::tie(input_, input_bdims) = unpackBatched(input);
	(input_.*Func)(args...);
  return input;
}

template <Tensor& (*Op)(Tensor&)>
Tensor& BatchedTensor_unary_pw_inplace_fn(Tensor& input) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
	Tensor input_;
	BatchDimsRef input_bdims;

	std::tie(input_, input_bdims) = unpackBatched(input);
	Op(input_);
  return input;
}

Tensor BatchedTensor_sum(const Tensor& self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  Tensor self_, result_;
  BatchDimsRef self_bdims;
  BatchDims result_bdims;

  std::tie(self_, self_bdims) = unpackBatched(self);
  std::tie(result_, result_bdims) = sum_batching_rule(self_, self_bdims, dim, keepdim, dtype);
  return detail::make_tensor<BatchTensorImpl>(result_, result_bdims);
}

Tensor BatchedTensor_transpose(const Tensor & self, int64_t dim0, int64_t dim1) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  Tensor self_, result_;
  BatchDimsRef self_bdims;
  BatchDims result_bdims;

  std::tie(self_, self_bdims) = unpackBatched(self);
  std::tie(result_, result_bdims) = transpose_batching_rule(self_, self_bdims, dim0, dim1);
  return detail::make_tensor<BatchTensorImpl>(result_, result_bdims);
}

Tensor& BatchedTensor_transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  // NB: need some clever bookkeeping for htis
  TORCH_INTERNAL_ASSERT(false, "NYI");
}

Tensor BatchedTensor_squeeze(const Tensor& self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  // NB: need some clever bookkeeping (reuse inferSqueezeGeometry) for this
  TORCH_INTERNAL_ASSERT(false, "NYI");
}

Tensor BatchedTensor_squeeze_dim(const Tensor& self, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  Tensor self_, result_;
  BatchDimsRef self_bdims;
  BatchDims result_bdims;

  std::tie(self_, self_bdims) = unpackBatched(self);
  std::tie(result_, result_bdims) = squeeze_dim_batching_rule(self_, self_bdims, dim);
  return detail::make_tensor<BatchTensorImpl>(result_, result_bdims);
}

template<typename Func, typename... T>
Tensor BatchedTensor_wrapper(const Func& batching_rule, const Tensor& self, T&&... args) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  Tensor self_, result_;
  BatchDimsRef self_bdims;
  BatchDims result_bdims;

  std::tie(self_, self_bdims) = unpackBatched(self);
  std::tie(result_, result_bdims) = batching_rule(self_, self_bdims, std::forward<T>(args)...);
  return detail::make_tensor<BatchTensorImpl>(result_, result_bdims);
}

// NB: We can save 4 LOC if we figure out how to pass the batching rule as a template parameter...
Tensor BatchedTensor_unsqueeze(const Tensor& self, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  return BatchedTensor_wrapper(unsqueeze_batching_rule, self, dim);
}

Tensor BatchedTensor_permute(const Tensor& self, IntArrayRef dims) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  return BatchedTensor_wrapper(permute_batching_rule, self, dims);
}

Tensor BatchedTensor_view(const Tensor& self, IntArrayRef size) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  return BatchedTensor_wrapper(view_batching_rule, self, size);
}

Tensor BatchedTensor_reshape(const Tensor& self, IntArrayRef shape) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  return BatchedTensor_wrapper(reshape_batching_rule, self, shape);
}

Tensor BatchedTensor_slice(const Tensor& self, int64_t dim, int64_t start, int64_t end, int64_t step) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  return BatchedTensor_wrapper(slice_batching_rule, self, dim, start, end, step);
}

Tensor BatchedTensor_select(const Tensor& self, int64_t dim, int64_t index) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  return BatchedTensor_wrapper(select_batching_rule, self, dim, index);
}

Tensor BatchedTensor_index(const Tensor& self, TensorList indices) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  return BatchedTensor_wrapper(index_batching_rule, self, indices);
}

std::vector<Tensor> BatchedTensor_chunk(const Tensor& self, int64_t chunks, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  Tensor self_;
  BatchDimsRef self_bdims;
  std::vector<Tensor> result;
  BatchDims result_bdims;

  std::tie(self_, self_bdims) = unpackBatched(self);
  std::tie(result, result_bdims) = chunk_batching_rule(self_, self_bdims, chunks, dim);
  for (int64_t i = 0; i < result.size(); i++) {
    result[i] = detail::make_tensor<BatchTensorImpl>(result[i], result_bdims);
  }
  return result;
}

// Copy pasta'ed from backed_fallback_test.cpp
void callBoxedWorkaround(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
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

// Modifies `stack` in-place.
// Returns: levels_bitset, batch_sizes
std::pair<std::bitset<64>,std::vector<int64_t>>
broadcastBdimsAtFront(torch::jit::Stack& stack) {
  std::unordered_map<int64_t,int64_t> level_to_size;
  std::bitset<64> levels_bitset;
  int64_t max_level = -1;
  BatchDimsRef bdims;
  Tensor arg;

  // Compute levels_bitset, level_to_size mapping
  for (const auto& ivalue : stack) {
    if (!ivalue.isTensor()) continue;
    std::tie(arg, bdims) = unpackBatched(ivalue.toTensor());
    auto arg_levels = createLevelsBitset(bdims);
    levels_bitset |= arg_levels;
    if (!arg.defined()) continue;
    auto arg_sizes = arg.sizes();
    for (const auto& bdim : bdims) {
      if (level_to_size.count(bdim.level())) {
        TORCH_CHECK(level_to_size[bdim.level()] == arg_sizes[bdim.index()]);
      } else {
        level_to_size[bdim.level()] = arg_sizes[bdim.index()];
      }
    }
  }

  // Get max_level
  for (int64_t idx = 0; idx < 64; idx++) {
    if (levels_bitset[idx]) {
      max_level = idx; 
    }
  }

  // Get batch_sizes
  std::vector<int64_t> batch_sizes;
  for (int64_t level = 0; level <= max_level; level++) {
    if (!levels_bitset[level]) {
      continue;
    }
    batch_sizes.push_back(level_to_size[level]);
  }

  // Move all bdims to front, align them, and expand them.
  for (int64_t idx = 0; idx < stack.size(); idx++) {
    const auto& ivalue = stack[idx];
    if (!ivalue.isTensor() || !ivalue.toTensor().defined()) {
      continue;
    }
    std::tie(arg, bdims) = unpackBatched(ivalue.toTensor());
    arg = moveBdimsToFront(arg, bdims);
    arg = alignTensorTo(arg, bdims, levels_bitset, max_level, arg.dim() - bdims.size());
    std::vector<int64_t> expanded_sizes(batch_sizes);
    expanded_sizes.insert(
        expanded_sizes.end(),
        arg.sizes().begin() + batch_sizes.size(),
        arg.sizes().end());
    stack[idx] = arg.expand(expanded_sizes);
  }
  return { levels_bitset, batch_sizes };
}

void batchTensorFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  TORCH_CHECK(
      !schema.is_mutable() && !schema.hasAnyAliasInfo(),
      "Batching rule not implemented for ", schema, "; ",
      "the fallback path doesn't work on in-place or view ops.");
  
  TORCH_WARN("Batching rule not implemented for ", op.schema(), " falling back "
             "to slow (for-loop) implementation");
  TORCH_CHECK(std::all_of(op.schema().returns().begin(),
                          op.schema().returns().end(),
                          [] (const Argument& arg) { return arg.type() == TensorType::get(); }),
              "Batching rule not implemented for ", op.schema(), ". ",
              "We could not generate a fallback.");


  auto num_arguments = op.schema().arguments().size();
  auto num_returns = op.schema().returns().size();

  std::bitset<64> levels_bitset;
  std::vector<int64_t> batch_sizes;

  auto args = torch::jit::pop(*stack, num_arguments);
  std::tie(levels_bitset, batch_sizes) = broadcastBdimsAtFront(args);
  auto total_batches = std::accumulate(
      batch_sizes.begin(), batch_sizes.end(),
      1, std::multiplies<int64_t>());


  // Set up batch_size stacks, one for each unbatched computation.
  std::vector<torch::jit::Stack> unbatched_stacks(total_batches);
  auto pushToEachStack = [&](const auto& ivalue) {
    for (auto& stack : unbatched_stacks) {
      torch::jit::push(stack, ivalue);
    }
  };
  for (int64_t linear_idx = 0; linear_idx < total_batches; linear_idx++) {
    std::vector<int64_t> idx = computeIndex(linear_idx, batch_sizes);
    for (const auto& ivalue : args) {
      if (!ivalue.isTensor() || !ivalue.toTensor().defined()) {
        torch::jit::push(unbatched_stacks[linear_idx], ivalue);
        continue;
      }
      auto tensor = ivalue.toTensor();
      if (!tensor.defined()) {
        torch::jit::push(unbatched_stacks[linear_idx], ivalue);
        continue;
      }
      torch::jit::push(
          unbatched_stacks[linear_idx],
          torch::jit::IValue(selectAll(tensor, idx)));
    }
  }

  // Call the op on each stack.
  for (auto& stack : unbatched_stacks) {
    callBoxedWorkaround(op, &stack);
  }

  // Only support num_returns == 1 for now. Also assume Tensor returns
  TORCH_INTERNAL_ASSERT(num_returns == 1);
  for (int64_t return_idx = 0; return_idx < num_returns; return_idx++) {
    std::vector<Tensor> output_shards;
    for (const auto& stack : unbatched_stacks) {
      output_shards.push_back(stack[return_idx].toTensor());
    }
    auto flat_output = at::stack(output_shards);
    std::vector<int64_t> output_sizes(batch_sizes);
    output_sizes.insert(
        output_sizes.end(),
        flat_output.sizes().begin() + 1,
        flat_output.sizes().end());
    auto output = detail::make_tensor<BatchTensorImpl>(
        flat_output.view(output_sizes),
        computeFrontBatchDims(levels_bitset));
    torch::jit::push(*stack, std::move(output));
  }
}

int64_t BatchedTensor_size(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());
  return self.sizes()[dim];
}

template <typename F, F Method, typename... Args>
Tensor& inplaceMethodFallback1(Tensor& input, Args... args) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
	Tensor input_;
	BatchDimsRef input_bdims;

  std::tie(input_, input_bdims) = unpackBatched(input);
  input_ = moveBdimsToFront(input_, input_bdims);
  IntArrayRef batch_sizes = { input_.sizes().begin(), input_.sizes().begin() + input_bdims.size() };

  auto total_batches = std::accumulate(
      batch_sizes.begin(), batch_sizes.end(),
      1, std::multiplies<int64_t>());
  for (int64_t linear_idx = 0; linear_idx < total_batches; linear_idx++) {
    std::vector<int64_t> idx = computeIndex(linear_idx, batch_sizes);
    (selectAll(input_, idx).*Method)(args...);
  }
  return input;
}

template <typename F, F Method, typename... Args>
Tensor& inplaceMethodFallback2(Tensor& input, const Tensor& other, Args... args) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
	Tensor input_, other_;
  std::vector<int64_t> batch_sizes;

  // TODO: we don't really need torch::jit::Stack...
  torch::jit::Stack tensors = { input, other };
  std::tie(std::ignore, batch_sizes) = broadcastBdimsAtFront(tensors);
  std::tie(input_, std::ignore) = unpackBatched(tensors[0].toTensor());
  std::tie(other_, std::ignore) = unpackBatched(tensors[1].toTensor());

  auto total_batches = std::accumulate(
      batch_sizes.begin(), batch_sizes.end(),
      1, std::multiplies<int64_t>());
  for (int64_t linear_idx = 0; linear_idx < total_batches; linear_idx++) {
    std::vector<int64_t> idx = computeIndex(linear_idx, batch_sizes);
    (selectAll(input_, idx).*Method)(selectAll(other_, idx), args...);
  }
  return input;
}

template <typename F, F Method, typename... Args>
Tensor& inplaceMethodFallback3(Tensor& input, const Tensor& second, const Tensor& third, Args... args) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
	Tensor input_, second_, third_;
  std::vector<int64_t> batch_sizes;

  // TODO: we don't really need torch::jit::Stack...
  torch::jit::Stack tensors = { input, second, third };
  std::tie(std::ignore, batch_sizes) = broadcastBdimsAtFront(tensors);
  std::tie(input_, std::ignore) = unpackBatched(tensors[0].toTensor());
  std::tie(second_, std::ignore) = unpackBatched(tensors[1].toTensor());
  std::tie(third_, std::ignore) = unpackBatched(tensors[2].toTensor());

  auto total_batches = std::accumulate(
      batch_sizes.begin(), batch_sizes.end(),
      1, std::multiplies<int64_t>());
  for (int64_t linear_idx = 0; linear_idx < total_batches; linear_idx++) {
    std::vector<int64_t> idx = computeIndex(linear_idx, batch_sizes);
    (selectAll(input_, idx).*Method)(selectAll(second_, idx), selectAll(third_, idx), args...);
  }
  return input;
}


// TODO: the fallback runs the un-batched kernel in a for loop.
// However, in many cases, operators are composed of other operators.
// If those operators have batched versions, then we don't need to
// run our for-loop-fallback. There should probably be some way to specify that.
// TODO: add BatchedTensorId
TORCH_LIBRARY_IMPL(_, Vmap, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&batchTensorFallback>());
}

TORCH_LIBRARY_IMPL(aten, Vmap, m) {
  // vmap-specific things
  m.impl("_make_batched", at::native::_make_batched);
  m.impl("_unwrap_batched", native::_unwrap_batched);
  m.impl("_is_batched", native::_is_batched);
  m.impl("size.int", BatchedTensor_size);

  // operators
  m.impl("mul.Tensor", BatchedTensor_mul);
  m.impl("add.Tensor", BatchedTensor_add);
  m.impl("dropout", BatchedTensor_dropout);
  m.impl("relu", BatchedTensor_relu);
  m.impl_UNBOXED("sum.dim_IntList", BatchedTensor_sum);
  m.impl_UNBOXED("conv2d", BatchedTensor_conv2d);

  // views
  m.impl_UNBOXED("transpose.int", BatchedTensor_transpose);
  m.impl_UNBOXED("transpose_", BatchedTensor_transpose_);
  m.impl_UNBOXED("squeeze", BatchedTensor_squeeze);
  m.impl_UNBOXED("squeeze.dim", BatchedTensor_squeeze_dim);
  m.impl_UNBOXED("unsqueeze", BatchedTensor_unsqueeze);
  m.impl_UNBOXED("permute", BatchedTensor_permute);
  m.impl_UNBOXED("view", BatchedTensor_view);
  m.impl_UNBOXED("reshape", BatchedTensor_reshape);
  m.impl_UNBOXED("alias", BatchedTensor_unary_pw_op<at::alias>);
  m.impl_UNBOXED("select.int", BatchedTensor_select);
  m.impl_UNBOXED("slice.Tensor", BatchedTensor_slice);
  m.impl_UNBOXED("index.Tensor", BatchedTensor_index);
  m.impl("detach", BatchedTensor_unary_pw_op<at::detach>);
  m.impl_UNBOXED("chunk", BatchedTensor_chunk);

  // composite
  m.impl_UNBOXED("t", native::t);
  m.impl_UNBOXED("numpy_T", native::numpy_T);

  // in-place

  // pointwise unary inplace
  m.impl_UNBOXED("abs_", BatchedTensor_unary_pw_inplace_op<&Tensor::abs_>);
  m.impl_UNBOXED("acos_", BatchedTensor_unary_pw_inplace_op<&Tensor::acos_>);
  m.impl_UNBOXED("asin_", BatchedTensor_unary_pw_inplace_op<&Tensor::asin_>);
  m.impl_UNBOXED("atan_", BatchedTensor_unary_pw_inplace_op<&Tensor::atan_>);
  m.impl_UNBOXED("bitwise_not_", BatchedTensor_unary_pw_inplace_op<&Tensor::bitwise_not_>);
  m.impl_UNBOXED("logical_not_", BatchedTensor_unary_pw_inplace_op<&Tensor::logical_not_>);
  m.impl_UNBOXED("ceil_", BatchedTensor_unary_pw_inplace_op<&Tensor::ceil_>);
  m.impl_UNBOXED("cos_", BatchedTensor_unary_pw_inplace_op<&Tensor::cos_>);
  m.impl_UNBOXED("cosh_", BatchedTensor_unary_pw_inplace_op<&Tensor::cosh_>);
  m.impl_UNBOXED("erf_", BatchedTensor_unary_pw_inplace_op<&Tensor::erf_>);
  m.impl_UNBOXED("erfc_", BatchedTensor_unary_pw_inplace_op<&Tensor::erfc_>);
  m.impl_UNBOXED("exp_", BatchedTensor_unary_pw_inplace_op<&Tensor::exp_>);
  m.impl_UNBOXED("expm1_", BatchedTensor_unary_pw_inplace_op<&Tensor::expm1_>);
  m.impl_UNBOXED("floor_", BatchedTensor_unary_pw_inplace_op<&Tensor::floor_>);
  m.impl_UNBOXED("frac_", BatchedTensor_unary_pw_inplace_op<&Tensor::frac_>);
  m.impl_UNBOXED("log_", BatchedTensor_unary_pw_inplace_op<&Tensor::log_>);
  m.impl_UNBOXED("log10_", BatchedTensor_unary_pw_inplace_op<&Tensor::log10_>);
  m.impl_UNBOXED("log1p_", BatchedTensor_unary_pw_inplace_op<&Tensor::log1p_>);
  m.impl_UNBOXED("log2_", BatchedTensor_unary_pw_inplace_op<&Tensor::log2_>);
  m.impl_UNBOXED("reciprocal_", BatchedTensor_unary_pw_inplace_op<&Tensor::reciprocal_>);
  m.impl_UNBOXED("neg_", BatchedTensor_unary_pw_inplace_op<&Tensor::neg_>);
  m.impl_UNBOXED("round_", BatchedTensor_unary_pw_inplace_op<&Tensor::round_>);
  m.impl_UNBOXED("relu_", BatchedTensor_unary_pw_inplace_op<&Tensor::relu_>);
  m.impl_UNBOXED("rsqrt_", BatchedTensor_unary_pw_inplace_op<&Tensor::rsqrt_>);
  // NB: selu_ has no method variant
  m.impl_UNBOXED("selu_", BatchedTensor_unary_pw_inplace_fn<at::selu_>);
  m.impl_UNBOXED("sigmoid_", BatchedTensor_unary_pw_inplace_op<&Tensor::sigmoid_>);
  m.impl_UNBOXED("sin_", BatchedTensor_unary_pw_inplace_op<&Tensor::sin_>);
  m.impl_UNBOXED("sinh_", BatchedTensor_unary_pw_inplace_op<&Tensor::sinh_>);
  m.impl_UNBOXED("detach_", BatchedTensor_unary_pw_inplace_op<&Tensor::detach_>);
  m.impl_UNBOXED("sqrt_", BatchedTensor_unary_pw_inplace_op<&Tensor::sqrt_>);
  m.impl_UNBOXED("square_", BatchedTensor_unary_pw_inplace_op<&Tensor::square_>);
  m.impl_UNBOXED("tan_", BatchedTensor_unary_pw_inplace_op<&Tensor::tan_>);
  m.impl_UNBOXED("tanh_", BatchedTensor_unary_pw_inplace_op<&Tensor::tanh_>);
  m.impl_UNBOXED("trunc_", BatchedTensor_unary_pw_inplace_op<&Tensor::trunc_>);
  m.impl_UNBOXED("zero_", BatchedTensor_unary_pw_inplace_op<&Tensor::zero_>);
  m.impl_UNBOXED("lgamma_", BatchedTensor_unary_pw_inplace_op<&Tensor::lgamma_>);
  m.impl_UNBOXED("digamma_", BatchedTensor_unary_pw_inplace_op<&Tensor::digamma_>);
  m.impl_UNBOXED("erfinv_", BatchedTensor_unary_pw_inplace_op<&Tensor::erfinv_>);
  m.impl_UNBOXED("sign_", BatchedTensor_unary_pw_inplace_op<&Tensor::sign_>);

  // pointwise unary inplace, extra arguments
  m.impl_UNBOXED("clamp_", BatchedTensor_unary_pw_inplace_fn_varargs<
      decltype(&at::clamp_), at::clamp_, optional<Scalar>, optional<Scalar>>);
  m.impl_UNBOXED("clamp_max_", BatchedTensor_unary_pw_inplace_fn_varargs<
      decltype(&at::clamp_max_), at::clamp_max_, Scalar>);
  m.impl_UNBOXED("clamp_min_", BatchedTensor_unary_pw_inplace_fn_varargs<
      decltype(&at::clamp_min_), at::clamp_min_, Scalar>);
  m.impl_UNBOXED("hardtanh_", BatchedTensor_unary_pw_inplace_fn_varargs<
      decltype(&at::hardtanh_), at::hardtanh_, Scalar, Scalar>);
  m.impl_UNBOXED("requires_grad_", BatchedTensor_unary_pw_inplace_meth_varargs<
      decltype(&Tensor::requires_grad_), &Tensor::requires_grad_, bool>);
  m.impl_UNBOXED("dropout_", BatchedTensor_unary_pw_inplace_fn_varargs<
      decltype(&at::dropout_), at::dropout_, double, bool>);
  m.impl_UNBOXED("feature_dropout_", BatchedTensor_unary_pw_inplace_fn_varargs<
      decltype(&at::feature_dropout_), at::feature_dropout_, double, bool>);
  m.impl_UNBOXED("alpha_dropout_", BatchedTensor_unary_pw_inplace_fn_varargs<
      decltype(&at::alpha_dropout_), at::alpha_dropout_, double, bool>);
  m.impl_UNBOXED("feature_alpha_dropout_", BatchedTensor_unary_pw_inplace_fn_varargs<
      decltype(&at::feature_alpha_dropout_), at::feature_alpha_dropout_, double, bool>);
  m.impl_UNBOXED("mvlgamma_", BatchedTensor_unary_pw_inplace_meth_varargs<
      decltype(&Tensor::mvlgamma_), &Tensor::mvlgamma_, int64_t>);
  m.impl_UNBOXED("rrelu_", BatchedTensor_unary_pw_inplace_fn_varargs<
      decltype(&at::rrelu_), &at::rrelu_, Scalar, Scalar, bool, optional<Generator>>);
  m.impl_UNBOXED("threshold_", BatchedTensor_unary_pw_inplace_fn_varargs<
      decltype(&at::threshold_), &at::threshold_, Scalar, Scalar>);
  m.impl_UNBOXED("polygamma_", BatchedTensor_unary_pw_inplace_meth_varargs<
      decltype(&Tensor::polygamma_), &Tensor::polygamma_, int64_t>);
  // TODO: random_ has like 3 overloads
  // m.impl_UNBOXED("random_", BatchedTensor_unary_pw_inplace_meth_varargs<
  //     decltype(&Tensor::random_), &Tensor::random_, optional<Generator>>);
  m.impl_UNBOXED("uniform_", BatchedTensor_unary_pw_inplace_meth_varargs<
      decltype(&Tensor::uniform_), &Tensor::uniform_, double, double, optional<Generator>>);
  m.impl_UNBOXED("cauchy_", BatchedTensor_unary_pw_inplace_meth_varargs<
      decltype(&Tensor::cauchy_), &Tensor::cauchy_, double, double, optional<Generator>>);
  m.impl_UNBOXED("log_normal_", BatchedTensor_unary_pw_inplace_meth_varargs<
      decltype(&Tensor::log_normal_), &Tensor::log_normal_, double, double, optional<Generator>>);
  m.impl_UNBOXED("exponential_", BatchedTensor_unary_pw_inplace_meth_varargs<
      decltype(&Tensor::exponential_), &Tensor::exponential_, double, optional<Generator>>);
  m.impl_UNBOXED("geometric_", BatchedTensor_unary_pw_inplace_meth_varargs<
      decltype(&Tensor::geometric_), &Tensor::geometric_, double, optional<Generator>>);
  m.impl_UNBOXED("normal_", BatchedTensor_unary_pw_inplace_meth_varargs<
      decltype(&Tensor::normal_), &Tensor::normal_, double, double, optional<Generator>>);
  m.impl_UNBOXED("elu_", BatchedTensor_unary_pw_inplace_fn_varargs<
      decltype(&at::elu_), &at::elu_, Scalar, Scalar, Scalar>);
  m.impl_UNBOXED("leaky_relu_", BatchedTensor_unary_pw_inplace_fn_varargs<
      decltype(&at::leaky_relu_), &at::leaky_relu_, Scalar>);
  m.impl_UNBOXED("celu_", BatchedTensor_unary_pw_inplace_fn_varargs<
      decltype(&at::celu_), &at::celu_, Scalar>);

#define INPLACE_FALLBACK(NUM_TENSORS, NAME) \
  m.impl_UNBOXED(#NAME, inplaceMethodFallback##NUM_TENSORS<decltype(&Tensor::NAME), &Tensor::NAME>);
#define INPLACE_FALLBACKV(NUM_TENSORS, NAME, ...) \
  m.impl_UNBOXED(#NAME, inplaceMethodFallback##NUM_TENSORS<decltype(&Tensor::NAME), &Tensor::NAME, __VA_ARGS__>);

  INPLACE_FALLBACKV(1, tril_, int64_t);
  INPLACE_FALLBACKV(1, triu_, int64_t);
  INPLACE_FALLBACKV(1, fill_diagonal_, Scalar, int64_t);
  INPLACE_FALLBACKV(1, renorm_, Scalar, int64_t, Scalar);
  INPLACE_FALLBACKV(2, copy_, bool);
  INPLACE_FALLBACKV(3, addcdiv_, Scalar);
  INPLACE_FALLBACKV(3, addcmul_, Scalar);
  INPLACE_FALLBACKV(3, addmv_, Scalar, Scalar);
  INPLACE_FALLBACKV(3, addr_, Scalar, Scalar);
  INPLACE_FALLBACKV(3, baddbmm_, Scalar, Scalar);
  INPLACE_FALLBACKV(3, addmm_, Scalar, Scalar);
  INPLACE_FALLBACKV(3, put_, bool);

  INPLACE_FALLBACK(2, logical_xor_);
  INPLACE_FALLBACK(2, logical_and_);
  INPLACE_FALLBACK(2, logical_or_);
  INPLACE_FALLBACK(3, masked_scatter_);


#undef INPLACE_FALLBACK
#undef INPLACE_FALLBACKV

  m.impl_UNBOXED("add_.Tensor", inplaceMethodFallback2<
      Tensor& (Tensor::*)(const Tensor&, Scalar) const,
      static_cast<Tensor& (Tensor::*)(const Tensor&, Scalar) const>(&Tensor::add_),
      Scalar>);
  m.impl_UNBOXED("sub_.Tensor", inplaceMethodFallback2<
      Tensor& (Tensor::*)(const Tensor&, Scalar) const,
      static_cast<Tensor& (Tensor::*)(const Tensor&, Scalar) const>(&Tensor::sub_),
      Scalar>);

#define BINARY_INPLACE_FALLBACK(NAME) \
  m.impl_UNBOXED(#NAME".Tensor", inplaceMethodFallback2< \
      Tensor& (Tensor::*)(const Tensor&) const, \
      static_cast<Tensor& (Tensor::*)(const Tensor&) const>(&Tensor::NAME)>); \
  m.impl_UNBOXED(#NAME".Scalar", inplaceMethodFallback1< \
      Tensor& (Tensor::*)(Scalar) const, \
      static_cast<Tensor& (Tensor::*)(Scalar) const>(&Tensor::NAME), \
      Scalar>);


  BINARY_INPLACE_FALLBACK(mul_);
  BINARY_INPLACE_FALLBACK(div_);
  BINARY_INPLACE_FALLBACK(lt_);
  BINARY_INPLACE_FALLBACK(le_);
  BINARY_INPLACE_FALLBACK(gt_);
  BINARY_INPLACE_FALLBACK(ge_);
  BINARY_INPLACE_FALLBACK(eq_);
  BINARY_INPLACE_FALLBACK(ne_);
  BINARY_INPLACE_FALLBACK(bitwise_and_);
  BINARY_INPLACE_FALLBACK(bitwise_or_);
  BINARY_INPLACE_FALLBACK(bitwise_xor_);
  BINARY_INPLACE_FALLBACK(pow_);
  BINARY_INPLACE_FALLBACK(fmod_);
  BINARY_INPLACE_FALLBACK(remainder_);
#undef BINARY_INPLACE_FALLBACK
}

Tensor BatchedTensor_rand(IntArrayRef size, const TensorOptions& options) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::VmapMode);
  auto* vmap_state = getVmapState();
  auto& stack = vmap_state->stack();
  std::vector<int64_t> new_sizes;
  BatchDims new_bdims;
  for (auto idx = 0; idx < stack.size(); idx++) {
    auto& level_and_size = stack[idx];
    new_sizes.push_back(level_and_size.second);
    new_bdims.push_back({idx, level_and_size.first});
  }
  new_sizes.insert(
      new_sizes.end(),
      size.begin(),
      size.end());
  return detail::make_tensor<BatchTensorImpl>(at::rand(new_sizes, options), new_bdims);
}

TORCH_LIBRARY_IMPL(_, VmapMode, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, VmapMode, m) {
  m.impl_UNBOXED("rand", BatchedTensor_rand);
}


} // namespace at
