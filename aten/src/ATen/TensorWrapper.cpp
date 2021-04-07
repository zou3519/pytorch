#include <ATen/TensorWrapper.h>
#include <torch/library.h>

namespace at {

void dumpTensor(std::ostream& ss, const Tensor& tensor) {
  auto* wrapped = maybeGetTensorWrapper(tensor);
  if (!wrapped) {
    ss << "Tensor" << tensor.sizes();
    return;
  }
  ss << "Wrapper[" << wrapped->level() << ", ";
  dumpTensor(ss, wrapped->value());
  ss << "]";
}

void dumpTensorCout(const Tensor& tensor) {
  dumpTensor(std::cout, tensor);
  std::cout << std::endl;
}

c10::intrusive_ptr<TensorWrapper> makeTensorWrapperPtr(const Tensor& tensor, int64_t level) {
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
  return c10::make_intrusive<TensorWrapper>(key_set, tensor, level);
}

Tensor makeTensorWrapper(const Tensor& tensor, int64_t level) {
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
  auto result = at::detail::make_tensor<TensorWrapper>(key_set, tensor, level);
  TORCH_INTERNAL_ASSERT(result.key_set().has(DispatchKey::TensorWrapper));
  return result;
}

c10::intrusive_ptr<TensorImpl> TensorWrapper::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  auto dest_impl = makeTensorWrapperPtr(value(), level());
  dest_impl->set_version_counter(version_counter);

  // TODO: is this even right?
  dest_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
  return dest_impl;
}

c10::intrusive_ptr<TensorImpl> TensorWrapper::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  auto dest_impl = makeTensorWrapperPtr(value(), level());
  dest_impl->set_version_counter(version_counter);

  // TODO: is this even right?
  dest_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
  return dest_impl;
}

void TensorWrapper::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
  TORCH_INTERNAL_ASSERT(false, "NYI");
}

TensorWrapper::TensorWrapper(c10::DispatchKeySet key_set, Tensor value, int64_t level)
  : TensorImpl(key_set, value.dtype(), value.device())
  , value_(std::move(value))
  , level_(level)
{
  TORCH_INTERNAL_ASSERT(value_.defined());
  set_storage_access_should_throw();

  // TODO: need to reset sizes/strides on mutation
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

TORCH_LIBRARY_IMPL(_, TensorWrapper, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

} // namespace at
