#include <ATen/BatchingRules.h>
#include <ATen/ATen.h>

namespace at {

std::pair<Tensor,BatchDims> sum_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    IntArrayRef dims, bool keepdim, c10::optional<ScalarType> dtype) {
  // NB: We don't really need to move the batch dims to the front.
  // One alternative way to do this is to keep them where they are and compute
  // the required `dims` to reduce over. However, assuming that the batch
  // dims are at front greatly simplifies the `dims` calculation.
  auto self_ = moveBatchDimsToFront(self, self_bdims);
  auto result_bdims = moveBatchDimsToFront(self_bdims);
  auto tensor_dims = self_.dim() - self_bdims.size();

  // Real dims to reduce over
  std::vector<int64_t> actual_dims;
  actual_dims.reserve(dims.size());
  for (int64_t dim : dims) {
    dim = maybe_wrap_dim(dim, tensor_dims);
    actual_dims.push_back(dim + self_bdims.size());
  }

  auto result = at::sum(self_, actual_dims, keepdim, dtype);
  return { result, result_bdims };
}

std::pair<Tensor,BatchDims>
mul_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    const Tensor& other, BatchDimsRef other_bdims) {
  Tensor self_, other_;
  BatchDims result_bdims;
  std::tie(self_, other_, result_bdims) = alignBatchDimsAtFront(self, self_bdims, other, other_bdims);
  return { at::mul(self_, other_), result_bdims };
}

}

