#include <ATen/Batching.h>
#include <ATen/BatchingUtils.h>
#include <ATen/WrapDimUtils.h>

namespace at {

/*
 * Batching rules for regular tensors.
 * To override any of these, register it as an operator.
 * NB: BatchDimsRef isn't supported for operators syntax. Might need to break it
 * into two vector<int64_t>.
 */ 


std::pair<Tensor,BatchDims>
add_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    const Tensor& other, BatchDimsRef other_bdims,
    Scalar alpha) {
  Tensor self_, other_;
  BatchDims result_bdims;
  std::tie(self_, other_, result_bdims) = alignBdimsAtFront(self, self_bdims, other, other_bdims);
  return { at::add(self_, other_, alpha), result_bdims };
}

std::pair<Tensor,BatchDims>
mul_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    const Tensor& other, BatchDimsRef other_bdims) {
  Tensor self_, other_;
  BatchDims result_bdims;
  std::tie(self_, other_, result_bdims) = alignBdimsAtFront(self, self_bdims, other, other_bdims);
  return { at::mul(self_, other_), result_bdims };
}


}
