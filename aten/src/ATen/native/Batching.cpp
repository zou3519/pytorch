#include <ATen/BatchedTensorImpl.h>

namespace at { namespace native {

Tensor _make_batched(const Tensor& self, optional<int64_t> batch_dim, int64_t level) {
  if (!batch_dim.has_value()) {
    return self;
  }
  return addBatchDim(self, level, *batch_dim);
}

static Tensor unwrapBatched(Tensor tensor, int64_t ntimes=1) {
  for (auto time = 0; time < ntimes; time++) {
    TORCH_INTERNAL_ASSERT(isBatched(tensor));
    const auto* batched = getBatched(tensor);
    BatchDims new_bdims = { batched->bdims().begin(), batched->bdims().end() };
    new_bdims.pop_back();
    if (new_bdims.size() > 0) {
      return makeBatched(batched->value(), std::move(new_bdims));
    } else {
      return batched->value();
    }
  }
  return tensor;
}

Tensor _unwrap_batched(const Tensor& self, int64_t level, int64_t batch_size, int64_t out_dim) {
  return removeBatchDim(self, level, batch_size, out_dim);
}

bool _is_batched(const Tensor& self) {
  return isBatched(self);
}

} // namespace native
} // namespace at
