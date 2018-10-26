#pragma once

#include "ATen/ATen.h"
#include "TH/THTensor.hpp"

#include "ATen/core/typeid.h"

#ifdef __CUDACC__
#include "ATen/cuda/ATenCUDAGeneral.h"
#include "THC/THCTensor.hpp"
#endif

namespace at { namespace native {

// ----------------- Some TH/THC agnostic Storage functions -------------------
template <Backend backend>
StorageImpl* newStorageImpl(const caffe2::TypeMeta& dtype) {
  AT_ASSERT(false);
}
template <>
inline StorageImpl* newStorageImpl<Backend::CPU>(const caffe2::TypeMeta& dtype) {
  return THStorage_new(dtype);
}
#ifdef __CUDACC__
template <>
inline StorageImpl* newStorageImpl<Backend::CUDA>(const caffe2::TypeMeta& dtype) {
  return THCStorage_new(globalContext().getTHCState(), dtype);
}
#endif

template <Backend backend>
void resizeStorageImpl(StorageImpl* storage, ptrdiff_t new_size) {
  AT_ASSERT(false);
}
template <>
inline void resizeStorageImpl<Backend::CPU>(StorageImpl* storage, ptrdiff_t new_size) {
  return THStorage_resize(storage, new_size);
}
#ifdef __CUDACC__
template <>
inline void resizeStorageImpl<Backend::CUDA>(StorageImpl* storage, ptrdiff_t new_size) {
  return THCStorage_resize(globalContext().getTHCState(), storage, new_size);
}
#endif

template <Backend backend>
void setNewStorageIfCPU(TensorImpl* self) {
  AT_ERROR("Tensor: invalid null storage");
}
template <>
inline void setNewStorageIfCPU<Backend::CPU>(TensorImpl* self) {
  THTensor_stealAndSetStoragePtr(self, newStorageImpl<Backend::CPU>(self->dtype()));
}

// These functions are called by native::resize_ as well as (legacy) TH resize.
// They are not in TH/THTensor.cpp because the at namespace is easier
// to benchmark than TH; I can't get gbenchmark to call fns from THTensor.cpp

template <Backend backend>
static inline void maybeResizeStorage(TensorImpl* self, int64_t new_size) {
  if (new_size + self->storage_offset() > 0) {
    if (!THTensor_getStoragePtr(self)) {
      setNewStorageIfCPU<backend>(self);
    }
    if (new_size + self->storage_offset() > self->storage().numel()) {
      resizeStorageImpl<backend>(
          THTensor_getStoragePtr(self), new_size + self->storage_offset());
    }
  }
}

template <Backend backend, bool device_guard=true>
inline TensorImpl* resizeTensorImpl(
    TensorImpl* self,
    IntList size,
    c10::optional<IntList> stride) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  c10::optional<DeviceGuard> maybe_guard;
  if (device_guard && backend == Backend::CUDA) {
    maybe_guard = DeviceGuard(self->storage().device().index());
  }

  size_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    // NB: storage size can be different from numel.
    for (size_t dim = 0; dim < size.size(); ++dim) {
      storage_size += (size[dim] - 1) * stride.value()[dim];
    }
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }

  maybeResizeStorage<backend>(self, storage_size);
  return self;
}

}}
