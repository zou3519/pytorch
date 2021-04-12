#include <ATen/TensorWrapper.h>
#include <torch/library.h>
#include <ATen/DynamicLayer.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/BatchedTensorImpl.h>

namespace at {

void dumpTensor(std::ostream& ss, const Tensor& tensor) {
  auto* wrapped = maybeGetTensorWrapper(tensor);
  if (!wrapped) {
    auto* batched = maybeGetBatchedImpl(tensor);
    if (batched) {
      ss << "Batched[" << batched->bdims() << ", ";
      dumpTensor(ss, batched->value());
      ss << "]";
      return;
    }
    ss << "Tensor" << tensor.sizes();
    return;
  }
  if (wrapped->is_alive()) {
    ss << "Wrapper[";
  } else {
    ss << "DeadWrapper[";
  }
  ss << wrapped->level() << ", ";
  dumpTensor(ss, wrapped->value());
  ss << "]";
}

void dumpTensorCout(const Tensor& tensor) {
  dumpTensor(std::cout, tensor);
  std::cout << std::endl;
}

c10::intrusive_ptr<TensorWrapper> makeTensorWrapperPtr(const Tensor& tensor, int64_t level, int64_t should_be_alive) {
  // TODO: denylist non-cuda/cpu backends to avoid funny business
  DispatchKeySet key_set;
  if (tensor.is_cuda()) {
    key_set = key_set.add(DispatchKey::CUDA);
    key_set = key_set.add(DispatchKey::AutogradCUDA);
  } else {
    key_set = key_set.add(DispatchKey::CPU);
    key_set = key_set.add(DispatchKey::AutogradCPU);
  }
  key_set = key_set.add(DispatchKey::TensorWrapper);
  if (should_be_alive) {
    auto& data = getGlobalDynmetaData();
    TORCH_INTERNAL_ASSERT(data.find(level) != data.end());
    return c10::make_intrusive<TensorWrapper>(key_set, tensor, level, data[level]);
  } else {
    return c10::make_intrusive<TensorWrapper>(key_set, tensor, level, std::make_shared<bool>(false));
  }
}

Tensor makeTensorWrapper(const Tensor& tensor, int64_t level) {
  auto& data = getGlobalDynmetaData();

  // TODO: denylist non-cuda/cpu backends to avoid funny business
  DispatchKeySet key_set;
  if (tensor.is_cuda()) {
    key_set = key_set.add(DispatchKey::CUDA);
    key_set = key_set.add(DispatchKey::AutogradCUDA);
  } else {
    key_set = key_set.add(DispatchKey::CPU);
    key_set = key_set.add(DispatchKey::AutogradCPU);
  }
  key_set = key_set.add(DispatchKey::TensorWrapper);
  TORCH_INTERNAL_ASSERT(data.find(level) != data.end());
  auto result = at::detail::make_tensor<TensorWrapper>(key_set, tensor, level, data[level]);
  TORCH_INTERNAL_ASSERT(result.key_set().has(DispatchKey::TensorWrapper));
  return result;
}

bool TensorWrapper::is_alive() const {
  return *is_alive_;
}

c10::intrusive_ptr<TensorImpl> TensorWrapper::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  auto dest_impl = makeTensorWrapperPtr(value(), level(), is_alive());
  dest_impl->set_version_counter(version_counter);

  // TODO: is this even right?
  dest_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
  return dest_impl;
}

c10::intrusive_ptr<TensorImpl> TensorWrapper::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  auto dest_impl = makeTensorWrapperPtr(value(), level(), is_alive());
  dest_impl->set_version_counter(version_counter);

  // TODO: is this even right?
  dest_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
  return dest_impl;
}

void TensorWrapper::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
  TORCH_INTERNAL_ASSERT(false, "NYI");
}

TensorWrapper::TensorWrapper(
    c10::DispatchKeySet key_set,
    Tensor value,
    int64_t level,
    std::shared_ptr<bool> is_alive,
    bool use_value_sizes_strides)
  : TensorImpl(key_set, value.dtype(), value.device())
  , value_(std::move(value))
  , level_(level)
  , is_alive_(std::move(is_alive))
{
  TORCH_INTERNAL_ASSERT(value_.defined());
  set_storage_access_should_throw();

  // TODO: need to reset sizes/strides on mutation
  if (use_value_sizes_strides) {
    auto dim = value_.dim();
    auto sizes = value_.sizes();
    auto strides = value_.strides();
    sizes_and_strides_.resize(value_.dim());
    for (int64_t i = 0; i < dim; i++) {
      sizes_and_strides_.size_at_unchecked(i) = sizes[i];
      sizes_and_strides_.stride_at_unchecked(i) = strides[i];
    }

    refresh_numel();
    refresh_contiguous();
  }
}

// The following are some internal inherited methods that we do not support.
// They should never get called.
void TensorWrapper::set_size(int64_t dim, int64_t new_size) {
  TORCH_INTERNAL_ASSERT(false, "Can't set_size for TensorWrapper");
}
void TensorWrapper::set_stride(int64_t dim, int64_t new_stride) {
  TORCH_INTERNAL_ASSERT(false, "Can't set_stride for TensorWrapper");
}
void TensorWrapper::set_storage_offset(int64_t storage_offset) {
  TORCH_INTERNAL_ASSERT(false, "Can't set_storage_offset for TensorWrapper");
}

const char* TensorWrapper::tensorimpl_type_name() const {
  return "TensorWrapper";
}


TensorWrapper* maybeGetTensorWrapper(const Tensor& tensor) {
  if (!tensor.key_set().has(DispatchKey::TensorWrapper)) {
    return nullptr;
  }
  return (TensorWrapper*)(tensor.unsafeGetTensorImpl());
}

static void foreachTensorInplace(std::vector<IValue>& args, int64_t begin, int64_t end,
    std::function<Tensor(const Tensor&)> func) {
  TORCH_INTERNAL_ASSERT(begin >= 0);
  TORCH_INTERNAL_ASSERT(end >= 0);
  TORCH_INTERNAL_ASSERT(begin <= end);
  for (int64_t idx = begin; idx < end; idx++) {
    auto ivalue = args[idx];
    if (ivalue.isTensorList()) {
      TORCH_INTERNAL_ASSERT(false, "NYI: TensorList");
    }
    if (!ivalue.isTensor()) {
      continue;
    }
    Tensor value = ivalue.toTensor();
    Tensor replacement = func(value);
    args[idx] = replacement; // TODO: std::move?
    if (ivalue.toTensor().defined()) {
      TORCH_INTERNAL_ASSERT(args[idx].toTensor().defined());
    }
  }
}

static Tensor unwrapIfDead(const Tensor& tensor) {
  auto* wrapped = maybeGetTensorWrapper(tensor);
  if (!wrapped) {
    return tensor;
  }
  if (wrapped->is_alive()) {
    return tensor;
  }
  return wrapped->value();
}

void dead_tensor_wrapper_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto args_size = op.schema().arguments().size();
  int64_t unwrapped_count = 0;
  auto unwrapIfDeadAndIncrement = [&](const Tensor& tensor) {
    auto* wrapped = maybeGetTensorWrapper(tensor);
    if (!wrapped) {
      return tensor;
    }
    if (wrapped->is_alive()) {
      return tensor;
    }
    unwrapped_count++;
    return wrapped->value();
  };

  foreachTensorInplace(*stack, stack->size() - args_size, stack->size(), unwrapIfDeadAndIncrement);
  TORCH_INTERNAL_ASSERT(unwrapped_count > 0, "Should have at least one dead wrapper");

  // re-dispatch
  op.callBoxed(stack);
}

// TensorWrapper backend fallback: Unwrap and fallthrough.

TORCH_LIBRARY_IMPL(_, TensorWrapper, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dead_tensor_wrapper_fallback>());
}

} // namespace at
