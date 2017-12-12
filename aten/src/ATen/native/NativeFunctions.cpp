#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/ExpandUtils.h"
#include <functional>
#include <numeric>

namespace at {
namespace native {

Tensor type_as(const Tensor& self, const Tensor& other) {
  return self.toType(other.type());
}

Tensor expand_as(const Tensor& self, const Tensor& other) {
  return self.expand(other.sizes());
}

std::vector<Tensor> split(const Tensor& self, int64_t split_size, int64_t dim) {
  int64_t dim_size = self.size(dim);
  int64_t num_splits = (dim_size + split_size - 1) / split_size;
  std::vector<Tensor> splits(num_splits);
  int64_t last_split_size = split_size - (split_size * num_splits - dim_size);

  for (int64_t i = 0; i < num_splits; ++i) {
    auto length = i < num_splits - 1 ? split_size : last_split_size;
    splits[i] = self.narrow(dim, i * split_size, length);
  }
  return splits;
}

Tensor slice(const Tensor& self, int64_t start, int64_t end, int64_t step, int64_t dim) {
  int64_t ndim = self.dim();
  AT_ASSERT(ndim > 0, "slice() cannot be applied to a 0-dim tensor.");
  dim = maybe_wrap_dim(dim, ndim);
  auto sizes = std::vector<int64_t>(self.sizes());
  auto strides = std::vector<int64_t>(self.strides());
  if (step <= 0) {
    // TODO: support negative strides
    throw std::runtime_error("slice step must be positive");
  }
  if (start < 0) {
    start += sizes[dim];
  }
  if (end < 0) {
    end += sizes[dim];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= sizes[dim]) {
    start = sizes[dim];
  }
  if (end < start) {
    end = start;
  } else if (end >= sizes[dim]) {
    end = sizes[dim];
  }
  auto storage_offset = self.storage_offset() + start * strides[dim];
  auto len = end - start;
  sizes[dim] = (len + step - 1) / step;  // round-up
  strides[dim] *= step;
  return self.as_strided(sizes, strides, storage_offset);
}

Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  AT_ASSERT(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  auto cur_size = self.size(dim);
  if (start < 0 || start >= cur_size) {
    runtime_error("start out of range");
  }
  if (length <= 0 || start > cur_size - length) {
    runtime_error("length out of range");
  }
  return at::native::slice(self, start, start + length, 1, dim);
}

Tensor select(const Tensor& self, int64_t dim, int64_t index) {
  int64_t ndim = self.dim();
  AT_ASSERT(ndim > 0, "select() cannot be applied to a 0-dim tensor.");
  dim = maybe_wrap_dim(dim, ndim);
  auto size = self.size(dim);
  if (index < -size || index >= size) {
    std::stringstream ss;
    ss << "select(): index " << index << " out of range for tensor of size ";
    ss << self.sizes() << " at dimension " << dim;
    throw std::runtime_error(ss.str());
  }
  if (index < 0) {
    index += size;
  }
  auto sizes = std::vector<int64_t>(self.sizes());
  auto strides = std::vector<int64_t>(self.strides());
  auto storage_offset = self.storage_offset() + index * strides[dim];
  sizes.erase(sizes.begin() + dim);
  strides.erase(strides.begin() + dim);
  return self.as_strided(sizes, strides, storage_offset);
}

std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim) {
  int64_t split_size = (self.size(dim) + chunks - 1) / chunks;
  // ensure this is dispatched through Tensor/Type, rather than the native function directly.
  return self.split(split_size, dim);
}

int64_t size(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());
  // wrap_dim guarantees bounds are correct.
  return self.sizes()[dim];
}

int64_t stride(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());
  // wrap_dim guarantees bounds are correct.
  return self.strides()[dim];
}

bool is_nonzero(const Tensor& self) {
  if (self.numel() != 1) {
    runtime_error("bool value of Tensor with more than one value is ambiguous");
  }
  Scalar localScalar = self.pImpl->localScalar();
  if (localScalar.isFloatingPoint()) {
    return localScalar.to<double>() != 0;
  } else if (localScalar.isIntegral()){
    return localScalar.to<int64_t>() != 0;
  }
  runtime_error("expected non-Tensor backed scalar");
}

bool is_same_size(const Tensor& self, const Tensor& other) {
  return self.sizes().equals(other.sizes());
}

bool is_cuda(const Tensor& self) {
  return self.type().is_cuda();
}

bool is_distributed(const Tensor& self) {
  return self.type().is_distributed();
}

bool is_sparse(const Tensor& self) {
  return self.type().is_sparse();
}

Tensor permute(const Tensor& self, IntList dims) {
  auto nDims = self.dim();
  if (dims.size() != (size_t)nDims) {
    runtime_error("number of dims don't match in permute");
  }
  auto oldSizes = self.sizes();
  auto oldStrides = self.strides();
  std::vector<int64_t> newSizes(nDims);
  std::vector<int64_t> newStrides(nDims);
  std::vector<bool> seen(nDims);
  for (int64_t i = 0; i < nDims; i++) {
    auto dim = maybe_wrap_dim(dims[i], nDims);
    if (seen[dim]) {
      runtime_error("repeated dim in permute");
    }
    seen[dim] = true;
    newSizes[i] = oldSizes[dim];
    newStrides[i] = oldStrides[dim];
  }
  return self.as_strided(newSizes, newStrides);
}

Tensor expand(const Tensor& self, IntList size) {
  if (size.size() < (size_t)self.dim()) {
    std::ostringstream ss;
    ss << "expand(" << self.type() << "{" << self.sizes() << "}, size=" << size
       << "): the number of sizes provided (" << size.size() << ") "
       << "must be greater or equal to the number of dimensions in the tensor ("
       << self.dim() << ")";
    throw std::runtime_error(ss.str());
  }

  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) = inferExpandGeometry(self, size);

  return self.as_strided(expandedSizes, expandedStrides);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferSqueezeGeometry(const Tensor &tensor) {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  for(int64_t d = 0; d < tensor.dim(); d++) {
    if(tensor.sizes()[d] != 1) {
      sizes.push_back(tensor.sizes()[d]);
      strides.push_back(tensor.strides()[d]);
    }
  }

  return std::make_tuple(sizes, strides);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferSqueezeGeometry(const Tensor& tensor, int64_t dim) {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  for(int64_t d = 0; d < tensor.dim(); d++) {
    if(d != dim || tensor.sizes()[dim] != 1) {
      sizes.push_back(tensor.sizes()[d]);
      strides.push_back(tensor.strides()[d]);
    }
  }
  return std::make_tuple(sizes, strides);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferUnsqueezeGeometry(const Tensor& tensor, int64_t dim) {
  if (tensor.numel() == 0) {
    throw std::runtime_error("cannot unsqueeze empty tensor");
  }

  std::vector<int64_t> sizes(tensor.sizes());
  std::vector<int64_t> strides(tensor.strides());
  int64_t new_stride = dim >= tensor.dim() - 1 ? 1 : sizes[dim] * strides[dim];
  sizes.insert(sizes.begin() + dim, 1);
  strides.insert(strides.begin() + dim, new_stride);

  return std::make_tuple(sizes, strides);
}

Tensor squeeze(const Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

Tensor squeeze(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());

  if (self.sizes()[dim] != 1) {
    return self.as_strided(self.sizes().vec(), self.strides().vec());
  }
  auto g = inferSqueezeGeometry(self, dim);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

Tensor & squeeze_(Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

Tensor & squeeze_(Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());

  if (self.sizes()[dim] != 1) {
    return self.as_strided_(self.sizes().vec(), self.strides().vec());
  }
  auto g = inferSqueezeGeometry(self, dim);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

Tensor sparseUnsqueeze(const Tensor& self, int64_t dim) {
  std::vector<int64_t> sizes(self.sizes());
  sizes.insert(sizes.begin() + dim, 1);

  if (dim > self.nDimensionI()) {
    auto new_values = self._values().unsqueeze_(dim - self.nDimensionI());
    return self.type().tensor(self._indices(), new_values, sizes);
  }

  auto new_indices = self._indices();
  auto zeros = new_indices.type().zeros({1, new_indices.size(1)});
  if (dim == 0) {
    new_indices = at::cat({zeros, new_indices}, 0);
  } else if (dim == self.nDimensionI()) {
    new_indices = at::cat({new_indices, zeros}, 0);
  } else {
    auto first_half = new_indices.narrow(0, 0, dim);
    auto second_half = new_indices.narrow(0, dim, self.nDimensionI() - dim);
    new_indices = at::cat({first_half, zeros, second_half}, 0);
  }
  return self.type().tensor(new_indices, self._values(), sizes);
}

Tensor sparseCat(TensorList tensors, int64_t dim) {
  if (tensors.size() == 0) {
    // Should return an empty sparse tensor but those don't really exist atm
    throw new std::runtime_error("NYI: cat on an empty list of sparse tensors");
  }
  std::vector<int64_t> sizes(tensors[0].sizes());
  std::vector<int64_t> cat_dim_sizes;
  std::vector<Tensor> indices_list;
  std::vector<Tensor> values_list;
  int64_t result_cat_dim_size = 0;
  int64_t nDimensionI = tensors[0].nDimensionI();

  for (auto& tensor : tensors) {
    indices_list.push_back(tensor._indices().clone());
    values_list.push_back(tensor._values().clone());
    cat_dim_sizes.push_back(tensor.size(dim));
    result_cat_dim_size += tensor.size(dim);
  }

  /*
   * Increment each indices tensor until the final index is correct,
   * then cat all of the indices together.
   */
  if (dim < nDimensionI) {
    int64_t current_size = 0;
    for (size_t i = 0; i < indices_list.size(); i++) {
      auto dim_indices = indices_list[i].select(0, dim);
      dim_indices += current_size;
      current_size += cat_dim_sizes[i];
    }

    sizes[dim] = current_size;
    auto new_indices = at::cat(indices_list, 1);
    auto new_values = at::cat(values_list, 0);
    return tensors[0].type().tensor(new_indices, new_values, sizes);
  }

  /*
   * Pad each values tensor with zero tensors and then cat all of
   * them together.
   */
  int64_t before_padding_size = 0;
  for (size_t i = 0; i < values_list.size(); i++) {
    auto after_padding_size =
        result_cat_dim_size - (before_padding_size + cat_dim_sizes[i]);
    auto& values = values_list[i];
    auto values_cat_dim = dim - nDimensionI + 1;

    std::vector<int64_t> padding_sizes(values.sizes());
    padding_sizes[values_cat_dim] = before_padding_size;
    auto before_zeros = values.type().zeros(padding_sizes);

    padding_sizes[values_cat_dim] = after_padding_size;
    auto after_zeros = values.type().zeros(padding_sizes);

    /*
     * Hack: we don't yet support tensors with 0 size in certain dims,
     * only dim-0 tensors. zeros({2, 0}) will give a tensor with size(2,).
     */
    std::vector<Tensor> to_cat;
    if (before_padding_size != 0) {
      to_cat.push_back(before_zeros);
    }
    to_cat.push_back(values);
    if (after_padding_size != 0) {
      to_cat.push_back(after_zeros);
    }

    values = at::cat(to_cat, values_cat_dim);
    before_padding_size += cat_dim_sizes[i];
  }
  sizes[dim] = result_cat_dim_size;
  auto new_indices = at::cat(indices_list, 1);
  auto new_values = at::cat(values_list, 0);
  return tensors[0].type().tensor(new_indices, new_values, sizes);
}

Tensor unsqueeze(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);

  if (self.is_sparse()) {
    return sparseUnsqueeze(self, dim);
  }

  auto g = inferUnsqueezeGeometry(self, dim);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

Tensor & unsqueeze_(Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);

  auto g = inferUnsqueezeGeometry(self, dim);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

// For backward, we save svd.
// http://www.ics.forth.gr/cvrl/publications/conferences/2000_eccv_SVD_jacobian.pdf
// But instead of gesvd SVD A = U(A) Sig(A) V(A)^T, which doesn't specify signs
// of determinants of U and V, we consider det(A) = \prod Sig_(A), where
//   1. A = U_(A) Sig_(A) V(A)^T
//   2. Sig_(A) and U_(A) can be different in signs in first row/col from
//      their counterparts so that U_(A) * V_(A) have +1 determinant
std::tuple<Tensor, Tensor, Tensor, Tensor> _det_with_svd(const Tensor& self) {
  if (!at::isFloatingType(self.type().scalarType()) ||
      self.dim() != 2 || self.size(0) != self.size(1)) {
    std::ostringstream ss;
    ss << "det(" << self.type() << "{" << self.sizes() << "}): expected a 2D"
       << "square tensor of floating types";
    throw std::runtime_error(ss.str());
  }
  // check symmetric
  bool symmetric = self.equal(self.transpose(0, 1));

  auto svd = self.svd(true);
  auto sigma = std::get<1>(svd);
  auto u = std::get<0>(svd);
  auto v = std::get<2>(svd);
  auto det = sigma.prod();
  if (!symmetric) {
    auto qr = self.geqrf();
    auto a = std::get<0>(qr);
    auto tau = std::get<1>(qr);
    // non-zero values in tau represent Householder reflectors, which has -1 det
    int64_t num_reflectors = tau.nonzero().size(0);
    auto qr_det = a.diag().prod();
    if (num_reflectors % 2 == 1) {
      qr_det = -qr_det;
    }
    det = qr_det;  // QR is more stable than svd, so use it anyways
    if ((qr_det < 0).any() ^ (det < 0).any()) {  // if different sign
      u.narrow(1, 0, 1).mul_(-1);
      sigma.narrow(0, 0, 1).mul_(-1);
    }
  }
  return std::make_tuple(det, u, sigma, v);
}

Tensor det(const Tensor& self) {
  return std::get<0>(self._det_with_svd());
}

Tensor cat(TensorList tensors, int64_t dim) {
  if (tensors.size() == 0 or not tensors[0].is_sparse()) {
    return at::_dense_cat(tensors, dim);
  }
  maybe_wrap_dim(dim, tensors[0].dim());
  return sparseCat(tensors, dim);
}

Tensor & cat_out(Tensor & self, TensorList tensors, int64_t dim) {
  return at::_dense_cat_out(self, tensors, dim);
}

Tensor stack(TensorList tensors, int64_t dim) {
  if (tensors.size() == 0) {
    throw std::runtime_error("stack expects a non-empty TensorList");
  }
  dim = maybe_wrap_dim(dim, tensors[0].dim() + 1);

  std::vector<Tensor> inputs(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    inputs[i] = tensors[i].unsqueeze(dim);
  }
  return at::cat(inputs, dim);
}

static Tensor maybeSqueeze(const Tensor & tensor, int64_t dim_tensor1, int64_t dim_tensor2) {
  if (dim_tensor1 == 1) {
    return tensor.squeeze(-2);
  } else if (dim_tensor2 == 1) {
    return tensor.squeeze(-1);
  } else {
    return tensor;
  }
}

/*
Matrix product of two Tensors.
The behavior depends on the dimensionality of the Tensors as follows:
- If both Tensors are 1-dimensional, the dot product (scalar) is returned.
- If both arguments are 2-dimensional, the matrix-matrix product is returned.
- If the first argument is 1-dimensional and the second argument is 2-dimensional,
  a 1 is prepended to its dimension for the purpose of the matrix multiply.
  After the matrix multiply, the prepended dimension is removed.
- If the first argument is 2-dimensional and the second argument is 1-dimensional,
  the matrix-vector product is returned.
- If both arguments are at least 1-dimensional and at least one argument is
  N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
  argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
  batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
  1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
  The non-matrix (i.e. batch) dimensions are broadcasted (and thus
  must be broadcastable).  For example, if tensor1 is a (j x 1 x n x m) Tensor
  and tensor2 is a (k x m x p) Tensor, the returned tensor will be an (j x k x n x p) Tensor.
*/
Tensor matmul(const Tensor & tensor1, const Tensor & tensor2) {
  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    return tensor1.dot(tensor2);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    return tensor1.mv(tensor2);
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    return tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    return tensor1.mm(tensor2);
  } else if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
    // optimization: use mm instead of bmm by folding tensor1's batch into
    // its leading matrix dimension.

    Tensor t2 = dim_tensor2 == 1 ? tensor2.unsqueeze(-1) : tensor2;
    auto size1 = tensor1.sizes();
    auto size2 = t2.sizes();
    std::vector<int64_t> output_size;
    output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
    output_size.insert(output_size.end(), size2.end() - 1, size2.end());

    // fold the batch into the first dimension
    Tensor t1 = tensor1.contiguous().view({-1, size1[size1.size() - 1]});

    auto output = t1.mm(t2).view(output_size);
    if (dim_tensor2 == 1) {
      output = output.squeeze(-1);
    }
    return output;
  } else if ((dim_tensor1 >= 1 && dim_tensor2 >= 1) && (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
    // We are multiplying b1 x n x m1 by x2 x m2 x p (where b1 can be a list);
    // we track m1 vs m2 separately even though they must match for nicer error messages
    int64_t n = dim_tensor1 > 1 ? tensor1.size(-2) : 1;
    int64_t m1 = tensor1.size(-1);
    IntList batch_tensor1(tensor1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0));
    int64_t m2 = dim_tensor2 > 1 ? tensor2.size(-2) : 1;
    int64_t p = tensor2.size(-1);
    IntList batch_tensor2(tensor2.sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0));

    // expand the batch portion (i.e. cut off matrix dimensions and expand rest)
    std::vector<int64_t> expand_batch_portion = infer_size(batch_tensor1, batch_tensor2);

    std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
    tensor1_expand_size.insert(tensor1_expand_size.end(), {n, m1});

    std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
    tensor2_expand_size.insert(tensor2_expand_size.end(), {m2, p});

    int expand_batch_product = std::accumulate(expand_batch_portion.begin(), expand_batch_portion.end(),
                                               1, std::multiplies<int64_t>());

    std::vector<int64_t> tensor1_bmm_view({expand_batch_product});
    tensor1_bmm_view.insert(tensor1_bmm_view.end(), {n, m1});

    std::vector<int64_t> tensor2_bmm_view({expand_batch_product});
    tensor2_bmm_view.insert(tensor2_bmm_view.end(), {m2, p});

    // flatten expanded batches
    Tensor tensor1_expanded = tensor1.expand(tensor1_expand_size).contiguous().view(tensor1_bmm_view);
    Tensor tensor2_expanded = tensor2.expand(tensor2_expand_size).contiguous().view(tensor2_bmm_view);

    Tensor output = tensor1_expanded.bmm(tensor2_expanded);

    // reshape batches back into result
    std::vector<int64_t> total_expansion(expand_batch_portion);
    total_expansion.insert(total_expansion.end(), {n, p});
    return maybeSqueeze(output.view(total_expansion), dim_tensor1, dim_tensor2);
  }

  runtime_error("both arguments to matmul need to be at least 1D, but they are %dD and %dD",
                dim_tensor1, dim_tensor2);

}

bool allclose(const Tensor& self, const Tensor& other, double rtol, double atol) {
  if (!self.sub(other).abs().le(other.abs().mul(rtol).add(atol)).all()) {
    return false;
  }

  return true;
}

std::tuple<at::Tensor, at::Tensor> RoiPooling2d_forward_cpu(
	const Tensor& input,
	const Tensor& rois,
	int64_t pooledHeight,
	int64_t pooledWidth,
	double spatialScale)
{
  // Input is the output of the last convolutional layer in the Backbone network, so
  // it should be in the format of NCHW
  AT_ASSERT(input.ndimension() == 4, "Input to RoI Pooling should be a NCHW Tensor");

  // ROIs is the set of region proposals to process. It is a 2D Tensor where the first
  // dim is the # of proposals, and the second dim is the proposal itself in the form
  // [batch_index startW startH endW endH]
  AT_ASSERT(rois.ndimension() == 2, "RoI Proposals should be a 2D Tensor, (batch_sz x proposals)");
  AT_ASSERT(rois.size(1) == 5, "Proposals should be of the form [batch_index startW startH endW enH]");

  auto proposals = rois.size(0);
  auto inputChannels = input.size(1);
  auto inputHeight = input.size(2);
  auto inputWidth = input.size(3);

  // Output Tensor is (num_rois, C, pooledHeight, pooledWidth)
  auto output = input.type().tensor({proposals, inputChannels, pooledHeight, pooledWidth});

  // TODO: need some mechanism for determining train vs. test

  // During training, we need to store the argmaxes for the pooling operation, so
  // the argmaxes Tensor should be the same size as the output Tensor
  auto argmaxes = input.type().toScalarType(kInt).tensor({proposals, inputChannels, pooledHeight, pooledWidth});

  AT_ASSERT(input.is_contiguous(), "input must be contiguous");
  AT_ASSERT(rois.is_contiguous(), "rois must be contiguous");

  auto *rawInput = input.data<float>();
  auto inputChannelStride = inputHeight * inputWidth;
  auto inputBatchStride = inputChannels * inputChannelStride;
  auto *rawRois = rois.data<float>();
  auto roiProposalStride = rois.size(1);

  auto *rawOutput = output.data<float>();
  auto *rawArgmaxes = argmaxes.data<int>();
  auto outputChannelStride = pooledHeight * pooledWidth;

  // Now that our Tensors are properly sized, we can perform the pooling operation.
  // We iterate over each RoI and perform pooling on each channel in the input, to
  // generate a pooledHeight x pooledWidth output for each RoI
  for (auto i = 0; i < proposals; ++i) {
    auto n = static_cast<int>(rawRois[0]);
    auto startWidth = static_cast<int>(std::round(rawRois[1] * spatialScale));
    auto startHeight = static_cast<int>(std::round(rawRois[2] * spatialScale));
    auto endWidth = static_cast<int>(std::round(rawRois[3] * spatialScale));
    auto endHeight = static_cast<int>(std::round(rawRois[4] * spatialScale));

    // TODO: assertions for valid values?
    // TODO: fix malformed ROIs??

    auto roiHeight = endHeight - startHeight;
    auto roiWidth = endWidth - startWidth;

    // Because the Region of Interest can be of variable size, but our output
    // must always be (pooledHeight x pooledWidth), we need to split the RoI
    // into a pooledHeight x pooledWidth grid of tiles

    auto tileHeight = static_cast<float>(roiHeight) / static_cast<float>(pooledHeight);
    auto tileWidth = static_cast<float>(roiWidth) / static_cast<float>(pooledWidth);

    auto *rawInputBatch = rawInput + (n * inputBatchStride);

    // Compute pooling for each of the (pooledHeight x pooledWidth) tiles for each
    // channel in the input
    for (auto ch = 0; ch < inputChannels; ++ch) {
      for (auto ph = 0; ph < pooledHeight; ++ph) {
        for (auto pw = 0; pw < pooledWidth; ++pw) {
          auto tileHStart = static_cast<int64_t>(std::floor(ph * tileHeight));
          auto tileWStart =	static_cast<int64_t>(std::floor(pw * tileWidth));
          auto tileHEnd = static_cast<int64_t>(std::ceil((ph + 1) * tileHeight));
          auto tileWEnd = static_cast<int64_t>(std::ceil((pw + 1) * tileWidth));

          // Add tile offsets to RoI offsets, and clip to input boundaries
          tileHStart = std::min(std::max<int64_t>(tileHStart + startHeight, 0), inputHeight);
          tileWStart = std::min(std::max<int64_t>(tileWStart + startWidth, 0), inputWidth);
          tileHEnd = std::min(std::max<int64_t>(tileHEnd + startHeight, 0), inputHeight);
          tileWEnd = std::min(std::max<int64_t>(tileWEnd + startWidth, 0), inputWidth);

          auto poolIndex = (ph * pooledWidth) + pw;

          // If our pooling region is empty, we set the output to 0, otherwise to
          // the min float so we can calculate the max properly
          auto empty = tileHStart >= tileHEnd || tileWStart >= tileWEnd;
          rawOutput[poolIndex] = empty ? 0 : std::numeric_limits<float>::min();

          // Set to -1 so we don't try to backprop to anywhere
          // TODO: make optional for test
          rawArgmaxes[poolIndex] = -1;

          for (auto th = tileHStart; th < tileHEnd; ++th) {
            for (auto tw = tileWStart; tw < tileWEnd; ++tw) {
              auto index = (th * inputWidth) + tw;
              if (rawInputBatch[index] > rawOutput[poolIndex]) {
                rawOutput[poolIndex] = rawInputBatch[index];
                // TODO: make optional for test
                rawArgmaxes[poolIndex] = index;
              }
            }
          }
        }
      }
      // Increment raw pointers by channel stride
      rawInputBatch += inputChannelStride;
      rawOutput += outputChannelStride;
      // TODO: make optional for test
      rawArgmaxes += outputChannelStride;
    }
    // Increment RoI raw pointer
    rawRois += roiProposalStride;
  }

  return std::make_tuple(output, argmaxes);
}

Tensor RoiPooling2d_backward_cpu(
  const Tensor& input,
  const Tensor& rois,
  int64_t pooledHeight,
  int64_t pooledWidth,
  double spatialScale,
  const Tensor& gradOutput,
  const Tensor& argmaxes) {
  throw std::runtime_error("not implemented");
}

}
}
