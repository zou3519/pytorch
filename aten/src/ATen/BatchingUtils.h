#include <ATen/BatchedTensorImpl.h>

namespace at {

/*
 * Utility functions used to implement batching rules.
 *
 * NB: All of these do NOT accept Tensors backed by BatchedTensor, unless
 * otherwise specified. These APIs usually operate on "unpacked BatchedTensors",
 * i.e. a (value Tensor, BatchDims) pair. This is for performance reasons:
 * we do not want to always wrap and unwrap BatchedTensors; we try to
 * only unwrap once per input tensor per operator and wrap once per output
 * tensor per operator.
 */ 

// If the input is a Tensor backed with a BatchedTensorImpl, then
// this function returns the underlying Tensor and BatchDims.
// If the input is a Tensor backed with regular TensorImpl, then
// this function returns the tensor and empty BatchDims.
TORCH_API std::pair<Tensor, BatchDimsRef> unpackBatched(const Tensor& self);

// Creates a bitset describing which levels are present in `bdims`.
// For example:
//   createLevelsBitset([(lvl=1, dim=2), (lvl=3, dim=1)]) -> 1010000...
std::bitset<kVmapMaxTensorDims> createLevelsBitset(BatchDimsRef bdims);

// Moves the specified BatchDims to the front of `self`, ordered by their level.
// Returns a view of the original tensor.
//
// For example:
//   moveBatchDimsToFront(ones(2, 3, 5), [(lvl=1, dim=2), (lvl=2, dim=1)])
// would return a view of size [5, 3, 2].
TORCH_API Tensor moveBatchDimsToFront(const Tensor& self, BatchDimsRef bdims);

// Reindexes batch dims (out-of-place) assuming they appear at the front of
// a tensor.
// For example:
//   moveBatchDimsToFront([(lvl=1, dim=2), (lvl=3, dim=1)])
// returns:
//   [(lvl=1, dim=0), (lvl=3, dim=0)]
TORCH_API BatchDims moveBatchDimsToFront(BatchDimsRef bdims);

// If the input is a Tensor backed with a BatchedTensorImpl, then
// this function returns the underlying Tensor and BatchDims.
// If the input is a Tensor backed with regular TensorImpl, then
// this function returns the tensor and empty BatchDims.
//
// NB: needs TORCH_API for aten/src/ATen/test/vmap_test.cpp
TORCH_API std::pair<Tensor, BatchDimsRef> unpackBatched(const Tensor& self);
std::bitset<kVmapMaxTensorDims> createLevelsBitset(BatchDimsRef bdims);
BatchDims moveBatchDimsToFront(BatchDimsRef bdims);
Tensor moveBatchDimsToFront(const Tensor& self, BatchDimsRef bdims);
std::pair<Tensor, BatchDimsRef> unpackBatched(const Tensor& self);

// Moves all the batch dims to the front of both tensors, aligns those dims
// in order of level, and returns the aligned tensors as well as the new
// BatchDims spec for both tensors. The aligned tensors are views of the
// originals.
//
// For example:
//  alignBatchDimsAtFront(ones(2, 3), [(lvl=1, dim=0)],
//                        ones(3, 5), [(lvl=2, dim=1)])
// returns:
//   - ones(2, 1, 3) (view of the original)
//   - ones(1, 5, 3) (view of the original)
//   - [(lvl=1, dim=0), (lvl=2, dim=1)]
std::tuple<Tensor,Tensor,BatchDims> alignBatchDimsAtFront(
    const Tensor& self,
    BatchDimsRef self_bdims,
    const Tensor& other,
    BatchDimsRef other_bdims);

// TODO: write something here...
Tensor alignTensorTo(
    const Tensor& tensor,
    BatchDimsRef tensor_bdims,
    std::bitset<kVmapMaxTensorDims> result_levels,
    int64_t max_result_level,
    int64_t num_result_regular_dims);

BatchDims computeFrontBatchDims(std::bitset<kVmapMaxTensorDims> levels_bitset);

}
