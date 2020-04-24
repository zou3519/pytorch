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

template <Tensor (*Op)(const Tensor&)>
std::pair<Tensor,BatchDims> unary_pw_batching_rule(const Tensor& self, BatchDimsRef self_bdims) {
  return { Op(self), { self_bdims.begin(), self_bdims.end() } };
}

std::pair<Tensor,BatchDims>
dropout_batching_rule(const Tensor& self, BatchDimsRef self_bdims, double p, bool train) {
  return { at::dropout(self, p, train), { self_bdims.begin(), self_bdims.end() } };
}

std::pair<Tensor,BatchDims>
dropout__batching_rule(Tensor& self, BatchDimsRef self_bdims, double p, bool train) {
  return { at::dropout_(self, p, train), { self_bdims.begin(), self_bdims.end() } };
}

std::pair<Tensor,BatchDims> conv2d_batching_rule(
    const Tensor& input, BatchDimsRef input_bdims,
    const Tensor& weight, BatchDimsRef weight_bdims,
    const Tensor& bias, BatchDimsRef bias_bdims,
    IntArrayRef stride, IntArrayRef padding,
    IntArrayRef dilation, int64_t groups) {
  if (weight_bdims.size() > 0) {
    TORCH_INTERNAL_ASSERT(false);
  }
  if (bias_bdims.size() > 0) {
    TORCH_INTERNAL_ASSERT(false);
  }

  auto input_ = moveBdimsToFront(input, input_bdims);
  auto result_bdims = moveBdimsToFront(input_bdims);

  if (input_.dim() <= 4) {
    // No need to flatten
    auto result = at::conv2d(input_, weight, bias, stride, padding, dilation, groups);
    return { result, result_bdims };
  }

  auto num_dims_to_flatten = input_bdims.size() + 1;
  auto flat_input_ = input_.flatten(0, num_dims_to_flatten - 1);
  auto flat_result = at::conv2d(flat_input_, weight, bias, stride, padding, dilation, groups);
  auto result = flat_result.unflatten(
      0, IntArrayRef(input.sizes().begin(), input.sizes().begin() + num_dims_to_flatten));

  return { result, result_bdims };
}

std::pair<Tensor,BatchDims> sum_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    IntArrayRef dims, bool keepdim, c10::optional<ScalarType> dtype) {
  auto self_ = moveBdimsToFront(self, self_bdims);
  auto result_bdims = moveBdimsToFront(self_bdims);

  // Real dims to reduce over
  SmallVector<int64_t,8> actual_dims;
  for (int64_t dim : dims) {
    actual_dims.push_back(dim + self_bdims.size());
  }

  auto result = at::sum(self_, actual_dims, keepdim, dtype);
  return { result, result_bdims };
}

std::pair<Tensor,BatchDims> transpose_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    int64_t dim0, int64_t dim1) {
  auto self_ = moveBdimsToFront(self, self_bdims);
  auto result_bdims = moveBdimsToFront(self_bdims);

  auto ndims = self.dim() - self_bdims.size();
  auto nbdims = self_bdims.size();
  dim0 = maybe_wrap_dim(dim0, ndims);
  dim1 = maybe_wrap_dim(dim1, ndims);

  auto result = at::transpose(self_, nbdims + dim0, nbdims + dim1);
  return { result, result_bdims };
}

std::pair<Tensor,BatchDims> squeeze_dim_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    int64_t dim) {
  auto self_ = moveBdimsToFront(self, self_bdims);
  auto result_bdims = moveBdimsToFront(self_bdims);

  auto ndims = self.dim() - self_bdims.size();
  auto nbdims = self_bdims.size();
  dim = maybe_wrap_dim(dim, ndims);

  auto result = at::squeeze(self_, nbdims + dim);
  return { result, result_bdims };
}

std::pair<Tensor,BatchDims> unsqueeze_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    int64_t dim) {
  auto self_ = moveBdimsToFront(self, self_bdims);
  auto result_bdims = moveBdimsToFront(self_bdims);

  auto ndims = self.dim() - self_bdims.size();
  auto nbdims = self_bdims.size();
  dim = maybe_wrap_dim(dim, ndims);

  auto result = at::unsqueeze(self_, nbdims + dim);
  return { result, result_bdims };
}

std::pair<Tensor,BatchDims> permute_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    IntArrayRef dims) {
  auto self_ = moveBdimsToFront(self, self_bdims);
  auto result_bdims = moveBdimsToFront(self_bdims);

  auto ndims = self.dim() - self_bdims.size();
  auto nbdims = self_bdims.size();

  SmallVector<int64_t,8> actual_dims;
  for (auto i = 0; i < self_bdims.size(); i++) {
    actual_dims.push_back(i);
  }
  for (const auto& dim : dims) {
    actual_dims.push_back(maybe_wrap_dim(dim, ndims) + self_bdims.size());
  }

  auto result = self_.permute(actual_dims);
  return { result, result_bdims };
}

std::pair<Tensor,BatchDims> reshape_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    IntArrayRef shape) {
  auto self_ = moveBdimsToFront(self, self_bdims);
  auto result_bdims = moveBdimsToFront(self_bdims);

  auto ndims = self.dim() - self_bdims.size();
  auto nbdims = self_bdims.size();
  auto self_sizes = self.sizes();

  SmallVector<int64_t,8> actual_shape;
  actual_shape.insert(
      actual_shape.end(),
      self_sizes.begin(),
      self_sizes.begin() + self_bdims.size());
  actual_shape.insert(
      actual_shape.end(),
      shape.begin(),
      shape.end());

  auto result = self_.reshape(actual_shape);
  return { result, result_bdims };
}

std::pair<Tensor,BatchDims> view_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    IntArrayRef size) {
  auto self_ = moveBdimsToFront(self, self_bdims);
  auto result_bdims = moveBdimsToFront(self_bdims);

  auto ndims = self.dim() - self_bdims.size();
  auto nbdims = self_bdims.size();
  auto self_sizes = self.sizes();

  SmallVector<int64_t,8> actual_shape;
  actual_shape.insert(
      actual_shape.end(),
      self_sizes.begin(),
      self_sizes.begin() + self_bdims.size());
  actual_shape.insert(
      actual_shape.end(),
      size.begin(),
      size.end());

  auto result = self_.view(actual_shape);
  return { result, result_bdims };
}

std::pair<Tensor,BatchDims> slice_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    int64_t dim, int64_t start, int64_t end, int64_t step) {
  auto self_ = moveBdimsToFront(self, self_bdims);
  auto result_bdims = moveBdimsToFront(self_bdims);

  auto ndims = self.dim() - self_bdims.size();
  auto nbdims = self_bdims.size();
  dim = maybe_wrap_dim(dim, ndims);

  auto result = at::slice(self_, dim + nbdims, start, end, step);
  return { result, result_bdims };
}

std::pair<Tensor,BatchDims> select_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    int64_t dim, int64_t index) {
  auto self_ = moveBdimsToFront(self, self_bdims);
  auto result_bdims = moveBdimsToFront(self_bdims);

  auto ndims = self.dim() - self_bdims.size();
  auto nbdims = self_bdims.size();
  dim = maybe_wrap_dim(dim, ndims);

  auto result = at::select(self_, dim + nbdims, index);
  return { result, result_bdims };
}

std::pair<Tensor,BatchDims> index_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims, TensorList indices) {
  auto self_ = moveBdimsToFront(self, self_bdims);
  auto result_bdims = moveBdimsToFront(self_bdims);

  // It's sufficient to use ":" for each of the batch dims.
  // For example, if the user did tensor[[0, 0], [2, 2]], all
  // we have to do is index real_tensor[:, [0, 0], [2, 2]]
  std::vector<Tensor> actual_indices;
  for (const auto& bdim : result_bdims) {
    actual_indices.push_back({});
  }
  actual_indices.insert(
      actual_indices.end(),
      indices.begin(),
      indices.end());
  auto result = at::index(self_, actual_indices);
  return { result, result_bdims };
}

// NB: Smallvector<5> or something (<= 5 vmap dims)
std::vector<int64_t> computeIndex(int64_t linear_idx, IntArrayRef sizes) {
  std::vector<int64_t> result;
  result.reserve(sizes.size());
  for (auto it = sizes.rbegin(); it != sizes.rend(); it++) {
    auto remainder = linear_idx % *it;
    result.push_back(remainder);
    linear_idx -= remainder;
    linear_idx /= *it;
  }
  std::reverse(std::begin(result), std::end(result));
  return result;
}


inline Tensor selectAll(const Tensor& tensor, IntArrayRef indices) {
  auto tensor_ = tensor;
  // NB: there's probably a faster way of doing this.
  for (int64_t dim = 0; dim < indices.size(); dim++) {
    tensor_ = tensor_.select(0, indices[dim]);
  }
  return tensor_;
}

// // Only useful for fill_diagonal_, lol...
// template <typename F, F Func, typename Args...>
// void inplace_fallback_rule1(Tensor& self, BatchDimsRef self_bdims, Args... args) {
//   auto self_ = moveBdimsToFront(self, self_bdims);
//   auto result_bdims = moveBdimsToFront(self, self_bdims);
//   auto self_sizes = self_.sizes();
//   
//   auto batch_sizes = IntArrayRef(self_sizes.begin(), self_sizes.begin() + result_bdims.size());
//   auto total_batches = std::accumulate(
//       batch_sizes.begin(), batch_sizes.end(),
//       1, std::multiplies<int64_t>());
// 
//   for (int64_t linear_idx = 0; linear_idx < total_batches; linear_idx++) {
//     std::vector<int64_t> idx = computeIndex(linear_idx, batch_sizes);
//     auto tensor_slice = selectAll(self_, idx);
//     Func(tensor_slice, args...);
//   }
// }

} // namespace at
