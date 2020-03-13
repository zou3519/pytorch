#pragma once

#include <ATen/ATen.h>

namespace at {

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(
      TensorList rep)
    : TensorImpl(
        c10::DispatchKeySet(DispatchKey::NestedTensorId),
        rep[0].dtype(),
        rep[0].device()
      )
    , rep_(rep.vec())
  {
    TORCH_INTERNAL_ASSERT(rep.size() > 0);
    int64_t dim = rep[0].dim();
    TORCH_INTERNAL_ASSERT(
        std::all_of(rep.begin(), rep.end(),
          [&](const Tensor& t) -> bool { return t.dim() == dim; }));

    // NB: A lot of internals work needs to be done to get this to work with None sizes.
    sizes_ = rep_[0].sizes();
    sizes_.insert(sizes_.begin(), rep.size());
  }

  // Either list or packed representation
  std::vector<Tensor> rep_;
};




}

