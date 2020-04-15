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
  return { Op(self), self_bdims.vec() };
}

template <Tensor& (*Op)(Tensor&)>
void unary_pw_inplace_batching_rule(Tensor& self, BatchDimsRef self_bdims) {
  Op(self);
}

std::pair<Tensor,BatchDims>
dropout_batching_rule(const Tensor& self, BatchDimsRef self_bdims, double p, bool train) {
  return { at::dropout(self, p, train), self_bdims.vec() };
}

std::pair<Tensor,BatchDims>
dropout__batching_rule(Tensor& self, BatchDimsRef self_bdims, double p, bool train) {
  return { at::dropout_(self, p, train), self_bdims.vec() };
}

std::pair<Tensor,BatchDims> conv2d_batching_rule(
    const Tensor& input, BatchDimsRef input_bdims,
    const Tensor& weight, BatchDimsRef weight_bdims,
    const Tensor& bias, BatchDimsRef bias_bdims,
    IntArrayRef stride, IntArrayRef padding,
    IntArrayRef dilation, int64_t groups) {
  if (weight_bdims.size() > 0) {
    // TODO: call fallback
    TORCH_CHECK(false, "NYI: conv2d_batching_rule for batched weight");
  }
  if (bias_bdims.size() > 0) {
    // TODO: call fallback
    TORCH_CHECK(false, "NYI: conv2d_batching_rule for batched bias");
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
  std::vector<int64_t> actual_dims;
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

  std::vector<int64_t> actual_dims;
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

  std::vector<int64_t> actual_shape;
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

  std::vector<int64_t> actual_shape;
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

} // namespace at
