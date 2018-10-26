#include "ATen/ATen.h"
#include "ATen/native/Resize.h"

namespace at { namespace native {

Tensor& resize_(Tensor& self, IntList size) {
  auto* self_ = self.unsafeGetTensorImpl();
  auto tid = self_->type_id();
  if (tid == CPUTensorId()) {
    resizeTensorImpl<Backend::CPU>(self_, size, /*strides=*/c10::nullopt);
  } else if (tid == CUDATensorId()) {
    resizeTensorImpl<Backend::CUDA>(self_, size, /*strides=*/c10::nullopt);
  } else {
    AT_ASSERT(false);
  }
  self_->maybe_zero_dim(size.size() == 0);
  return self;
}

}}
