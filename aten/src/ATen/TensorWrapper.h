#pragma once

#include <ATen/Tensor.h>

namespace at {

struct TORCH_API TensorWrapper : public c10::TensorImpl {
  explicit TensorWrapper(
      c10::DispatchKeySet key_set,
      Tensor value,
      int64_t level,
      std::shared_ptr<bool> is_alive,
      bool use_value_sizes_strides = true);

  // Override a bunch of methods inherited from TensorImpl to return error messages
  void set_size(int64_t dim, int64_t new_size) override;
  void set_stride(int64_t dim, int64_t new_stride) override;
  void set_storage_offset(int64_t storage_offset) override;

  const Tensor& value() const {
    return value_;
  }
  int64_t level() const {
    return level_;
  }
  bool is_alive() const;

  // Overrides necessary for autograd
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const override;
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

 private:
  const char* tensorimpl_type_name() const override;
  Tensor value_;
  int64_t level_;

  // When we exit the level, this wrapper may be marked as "not alive".
  // Wrappers that are not alive:
  // 1) May still have autograd metadata on them
  // 2) Forward dispatches to the underlying value()
  std::shared_ptr<bool> is_alive_;
};

TORCH_API Tensor makeTensorWrapper(const Tensor& tensor, int64_t level);
TORCH_API TensorWrapper* maybeGetTensorWrapper(const Tensor& tensor);
TORCH_API void dumpTensor(std::ostream & ss, const Tensor& tensor);
TORCH_API void dumpTensorCout(const Tensor& tensor);

} // namespace at
