#include <ATen/BatchingUtils.h>
#include <ATen/ATen.h>

namespace at {

static bool areBdimsAtFrontInOrder(BatchDimsRef bdims) {
  for (int64_t idx = 0; idx < bdims.size(); idx++) {
    if (bdims[idx].dim() != idx) {
      return false;
    }
  }
  return true;
}

BatchDims moveBatchDimsToFront(BatchDimsRef bdims) {
  BatchDims result;
  result.reserve(bdims.size());
  for (int64_t idx = 0; idx < bdims.size(); idx++) {
    result.push_back(BatchDim(bdims[idx].level(), idx));
  }
  return result;
}

Tensor moveBatchDimsToFront(const Tensor& self, BatchDimsRef bdims) {
  if (areBdimsAtFrontInOrder(bdims)) {
    return self;
  }
  const auto self_sizes = self.sizes();
  std::vector<int64_t> permutation(self_sizes.size(), 0);
  permutation.reserve(self_sizes.size());
  const auto is_bdim = createBatchDimBitset(bdims);
  int64_t idx = 0;
  for (const auto& bdim : bdims) {
    permutation[idx++] = bdim.dim();
  }
  for (int64_t ptr = 0; idx < self_sizes.size(); ptr++) {
    if (is_bdim[ptr]) {
      continue;
    }
    permutation[idx++] = ptr;
  }
  return self.permute(permutation);
}

std::pair<Tensor, BatchDimsRef> unpackBatched(const Tensor& self) {
  const auto* batched = maybeGetBatched(self);
  if (batched) {
    return { batched->value(), batched->bdims() };
  }
  return { self, {} };
}

std::bitset<kVmapMaxTensorDims> createLevelsBitset(BatchDimsRef bdims) {
  std::bitset<kVmapMaxTensorDims> result;
  for (const auto& bdim : bdims) {
    result.set(bdim.level());
  }
  return result;
}

Tensor alignTensorTo(
    const Tensor& tensor,
    BatchDimsRef tensor_bdims,
    std::bitset<kVmapMaxTensorDims> result_levels,
    int64_t max_result_level,
    int64_t num_result_regular_dims) {
  // NB: Two prerequisites:
  // 1. tensor_bdims are all at the front of tensor
  // 2. all of tensor_bdims are accounted for in result_levels
  auto tensor_sizes = tensor.sizes();
  int64_t num_result_bdims = result_levels.count();
  int64_t num_tensor_regular_dims = tensor_sizes.size() - tensor_bdims.size();
  if (num_tensor_regular_dims == num_result_regular_dims &&
      tensor_bdims.size() == num_result_bdims) {
    return tensor;
  }

  SmallVector<int64_t,8> aligned_sizes(num_result_bdims + num_result_regular_dims, 1);

  // align the regular (non-bdims) first
  std::copy(
      tensor_sizes.rbegin(),
      tensor_sizes.rbegin() + num_tensor_regular_dims,
      aligned_sizes.rbegin());

  // align the bdims
  int64_t level = 0;
  int64_t dim = 0;
  auto tensor_bdims_iter = tensor_bdims.begin();
  while (level <= max_result_level && tensor_bdims_iter != tensor_bdims.end()) {
    if (!result_levels[level]) {
      level++;
      continue;
    }
    if (tensor_bdims_iter->level() == level) {
      aligned_sizes[dim] = tensor_sizes[tensor_bdims_iter - tensor_bdims.begin()]; 
      level++;
      tensor_bdims_iter++;
    } else if (tensor_bdims_iter->level() < level) {
      tensor_bdims_iter++; 
    } else if (tensor_bdims_iter->level() > level) {
      level++;
    }
    dim++;
  }

  return tensor.view(aligned_sizes);
}

static BatchDims computeFrontBatchDims(std::bitset<kVmapMaxTensorDims> levels_bitset) {
  BatchDims bdims;
  int64_t dim = 0;
  for (int64_t level = 0; level < kVmapMaxTensorDims; level++) {
    if (!levels_bitset[level]) {
      continue;
    }
    bdims.push_back({level, dim++});
  }
  return bdims;
}

std::tuple<Tensor,Tensor,BatchDims> alignBatchDimsAtFront(
    const Tensor& self,
    BatchDimsRef self_bdims,
    const Tensor& other,
    BatchDimsRef other_bdims) {

  // Step 1: Permute the bdims to the front of the tensors
  auto self_ = moveBatchDimsToFront(self, self_bdims);
  auto other_ = moveBatchDimsToFront(other, other_bdims);

  auto self_sizes = self.sizes();
  auto other_sizes = other.sizes();

  // Step 2: Align the bdims
  auto self_levels = createLevelsBitset(self_bdims);
  auto other_levels = createLevelsBitset(other_bdims);
  auto result_levels = self_levels | other_levels;
  auto max_result_level = 0;
  if (self_bdims.size() == 0) {
    max_result_level = other_bdims.back().level();
  } else if (other_bdims.size() == 0) {
    max_result_level = self_bdims.back().level();
  } else {
    max_result_level = std::max(self_bdims.back().level(), other_bdims.back().level());
  }
  auto num_result_regular_dims = std::max(
      self_sizes.size() - self_bdims.size(),
      other_sizes.size() - other_bdims.size());
  self_ = alignTensorTo(self_, self_bdims, result_levels, max_result_level, num_result_regular_dims);
  other_ = alignTensorTo(other_, other_bdims, result_levels, max_result_level, num_result_regular_dims);

  // Step 3: construct the result bdims
  BatchDims result_bdims = computeFrontBatchDims(result_levels);

  return { self_, other_, result_bdims };
}

}
