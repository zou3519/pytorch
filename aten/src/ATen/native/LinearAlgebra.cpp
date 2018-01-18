#include "ATen/ATen.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include <functional>
#include <numeric>
#include <vector>
#include <map>

namespace at {
namespace native {

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
    ss << "det(" << self.type() << "{" << self.sizes() << "}): expected a 2D "
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
    if ((qr_det < 0).any().toCByte() ^ (det < 0).any().toCByte()) {  // if different sign
      u.narrow(1, 0, 1).mul_(-1);
      sigma.narrow(0, 0, 1).mul_(-1);
    }
  }
  return std::make_tuple(det, u, sigma, v);
}

Tensor det(const Tensor& self) {
  return std::get<0>(self._det_with_svd());
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

using SpMatMap = std::map<int64_t,std::map<int64_t,float>>;

static Tensor sp_mm(const Tensor & tensor1, const Tensor & tensor2) {
  int64_t nnz = tensor1._indices().size(1);
  auto tensor1_indices = tensor1._indices().contiguous();
  auto tensor2_indices = tensor2._indices().contiguous();

  auto tensor1_data = (int64_t*)tensor1_indices.data_ptr();
  auto tensor2_data = (int64_t*)tensor2_indices.data_ptr();

  auto tensor1_values_data = (float*)tensor1._values().data_ptr();
  auto tensor2_values_data = (float*)tensor2._values().data_ptr();

  SpMatMap t1_map;
  SpMatMap t2_map;

  // the strides are nnz, 1
  for (int64_t i = 0; i < nnz; i++) {
    auto row = tensor1_data[0*nnz + i];
    auto col = tensor1_data[1*nnz + i];
    t1_map[row][col] = tensor1_values_data[i];

    row = tensor2_data[0*nnz + i];
    col = tensor2_data[1*nnz + i];
    t2_map[col][row] = tensor2_values_data[i];
  }
  
  int64_t out_nnz = t1_map.size() * t2_map.size();

  auto result_indices = tensor1_indices.type().tensor({2, out_nnz});
  auto result_values = tensor1._values().type().tensor({out_nnz});

  auto result_indices_data = (int64_t*)result_indices.data_ptr();
  auto result_values_data = (float*)result_values.data_ptr();
  auto result_size = std::array<int64_t, 2>{{ tensor1.size(0), tensor2.size(1) }};

  auto inner_dim_size = tensor1.size(1);
  int64_t count = 0;
  for (auto& it_t1map : t1_map) {
    auto row = it_t1map.first;
    auto t1_inner_map = it_t1map.second;
    for (auto& it_t2map : t2_map) {
      auto col = it_t2map.first;
      auto t2_inner_map = it_t2map.second;

      // we have t1's row vector and t2's col vector. time to take the dot product.
      float sum = 0;
      for (int64_t i = 0; i < inner_dim_size; i++) {
        if (t1_inner_map.count(i) > 0 && t2_inner_map.count(i) > 0) {
          sum += t1_inner_map[i] * t2_inner_map[i];
        }
      }
      
      if (sum != 0) {
        result_indices_data[0*nnz + count] = row;
        result_indices_data[1*nnz + count] = col;
        result_values_data[count] = sum;
        ++count;
      }
    }  
  }
  
  // now, return new sparse tensor
  tensor1.type().sparse_coo_tensor(result_indices, result_values, result_size);
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
    if (tensor1.is_sparse()) {
      return sp_mm(tensor1, tensor2);
    }
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

}
}
