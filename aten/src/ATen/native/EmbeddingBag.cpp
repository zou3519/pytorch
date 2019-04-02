#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/NativeFunctions.h>

#include <TH/THBlasUtils.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
  const int MODE_SUM = 0;
  const int MODE_MEAN = 1;
  const int MODE_MAX = 2;
}

namespace at {
namespace native {

static void make_offset2bag(const Tensor &offsets, const Tensor &indices,
                            Tensor &offset2bag) {
  offset2bag.index_add_(
      0, offsets, at::ones_like(offsets)); // offset2bag = [1 0 1 0 1]
  offset2bag[0] -= 1;                     // offset2bag = [0 0 1 0 1]
  offset2bag = offset2bag.cumsum(0);     // offset2bag = [0 0 1 1 2]
}

// This function combines index_select (using select_indices as the index) and
// index_add (using add_indices as the index), without creating an intermediary
// tensor to hold the selected embeddings
template<typename T>
static void index_select_add(const Tensor &select_indices,
                             const Tensor &add_indices,
                             const Tensor &src,
                             Tensor &output) {
  auto add_indices_data = add_indices.data<int64_t>();
  auto select_indices_data = select_indices.data<int64_t>();
  auto src_data = src.data<T>();
  auto output_data = output.data<T>();
  auto numel = add_indices.numel();
  int64_t ddim = src.size(1);
  auto src_stride0 = src.stride(0);
  auto src_stride1 = src.stride(1);
  auto output_stride0 = output.stride(0);
  auto output_stride1 = output.stride(1);
  for (int64_t i = 0; i < numel; i++) {
    THBlas_axpy<T>(ddim, 1,
            src_data + src_stride0 * select_indices_data[i], src_stride1,
            output_data + output_stride0 * add_indices_data[i], output_stride1);
  }
}

static void make_bag_size(const Tensor &offsets, const Tensor &indices,
                          const int64_t mode, Tensor &bag_size) {
  if (mode == MODE_MEAN || mode == MODE_MAX) {
    // Compute this for MODE_MEAN and MODE_MAX (latter needed for backwards)
    if (offsets.size(0) != 1) {
      bag_size.slice(0, 0, bag_size.size(0) - 1, 1) =
          offsets.slice(0, 1, offsets.size(0), 1) -
          offsets.slice(0, 0, offsets.size(0) - 1, 1);
    }
    bag_size[-1] = indices.size(0) - offsets[-1];
  }
}

static Tensor apply_bag_size(const Tensor &offsets, const Tensor &indices,
                             const int64_t mode, Tensor &output,
                             const Tensor &bag_size) {
  if (mode == MODE_MEAN) {
    if (offsets.size(0) == 1) {
      auto bag_size_ = indices.size(0);
      output /= bag_size_;
    } else {
      // Avoid dividing by 0 for empty bags.
      // Instead we want empty bags to return all 0s
      auto bag_size_ = at::max(bag_size, at::ones_like(bag_size))
                           .to(output.options())
                           .unsqueeze(1)
                           .expand_as(output);
      output /= bag_size_;
    }
  }
  return output;
}

static Tensor apply_bag_size_backward(const Tensor &offsets,
                                      const Tensor &indices, const int64_t mode,
                                      Tensor &output, const Tensor &offset2bag,
                                      const Tensor &bag_size) {
  if (mode == MODE_MEAN) {
    if (offsets.size(0) == 1) {
      auto bag_size_ = indices.size(0);
      output /= bag_size_;
    } else {
      auto inv_bag_size_ = (1 / bag_size.to(output.options()))
                             .unsqueeze(1)
                             .index_select(0, offset2bag);
      output *= inv_bag_size_;
    }
  }
  return output;
}


template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor, Tensor> embedding_bag_cpu_max(
  const Tensor& weight, const Tensor &indices, const Tensor& offset2bag, const Tensor& output, const Tensor& bag_size, const Tensor& offsets) {

    auto max_indices = at::zeros({offsets.size(0), weight.size(1)}, indices.options());

    int64_t numel = indices.numel();
    int64_t dims = weight.size(1);
    auto indices_data = indices.data<int64_t>();
    auto offset2bag_data = offset2bag.data<int64_t>();

    auto max_indices_data = max_indices.data<int64_t>();
    auto max_indices_stride = max_indices.stride(0);

    auto weight_data = weight.data<scalar_t>();
    auto output_data = output.data<scalar_t>();
    auto weight_stride0 = weight.stride(0);
    auto weight_stride1 = weight.stride(1);
    auto output_stride = output.stride(0);

    for (int i = 0; i < numel; i++) {
      auto bag = offset2bag_data[i];
      auto word_idx = indices_data[i];


      for (int dim = 0; dim < dims; dim++) {
        auto& current_item = output_data[output_stride * bag + dim];
        auto weight_item = weight_data[weight_stride0 * word_idx + dim * weight_stride1];

        bool is_first_for_bag = (i == 0) || offset2bag_data[i - 1] != bag;

        if (is_first_for_bag || weight_item > current_item) {
          current_item = weight_item;
          max_indices_data[max_indices_stride * bag + dim] = word_idx;
        }
      }
    }

    return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, offset2bag, bag_size, max_indices);
}

// embedding_bag wrapper to enforce contiguity in tensors other than `weight`.
// This is created to save extra `.contiguous()` call in backward.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
embedding_bag(const Tensor &weight, const Tensor &indices,
              const Tensor &offsets, const bool scale_grad_by_freq,
              const int64_t mode, bool sparse) {
  return at::_embedding_bag(weight, indices.contiguous(), offsets.contiguous(),
                            scale_grad_by_freq, mode, sparse);
  };

// Assumes all input tensors except for `weight` are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
_embedding_bag_cpu(const Tensor &weight, const Tensor &indices,
                  const Tensor &offsets, const bool scale_grad_by_freq,
                  const int64_t mode, bool sparse) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding_bag", indices_arg, kLong);
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarType("embedding_bag", indices_arg, kLong);
  auto weight_arg = TensorArg(weight, "weight", 1);
  checkScalarTypes("embedding_bag", weight_arg, {kFloat, kDouble});

  auto bag_size = at::zeros(offsets.sizes(), indices.options());
  make_bag_size(offsets, indices, mode, bag_size);

  // If the last entries are empty, that the last offsets are irrelevant as they
  // won't change anything in the assignment of ID -> bag, but index_add would
  // throw out of bounds error. So to keep it simple we just add one more
  // entry to the end then get rid of it after make_offset2bag.
  auto offset2bag = at::zeros(
     {indices.sizes()[0] + 1}, indices.options()); // offset2bag = [0 0 0 0 0]

  make_offset2bag(offsets, indices, offset2bag);

  offset2bag.resize_({indices.sizes()[0]});

  auto output = at::zeros({offsets.size(0), weight.size(1)}, weight.options());

  if (mode == MODE_MEAN || mode == MODE_SUM) {
    AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "embedding_bag_cpu", [&]() {
      index_select_add<scalar_t>(indices, offset2bag, weight, output);
    });
    auto ret = apply_bag_size(offsets, indices, mode, output, bag_size);
    return std::tuple<Tensor, Tensor, Tensor, Tensor>(ret, offset2bag, bag_size, bag_size);
  } else { // MODE_MAX
    return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weight.scalar_type(), "embedding_bag_cpu_max", [&]() {
        return embedding_bag_cpu_max<scalar_t>(weight, indices, offset2bag, output, bag_size, offsets);
      }
    );
  }
}

// Assumes all input tensors are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
Tensor _embedding_bag_backward(const Tensor &grad, const Tensor &indices,
                              const Tensor &offsets,
                              const Tensor &offset2bag,
                              const Tensor &bag_size_,
                              const Tensor &max_indices_,
                              int64_t num_weights,
                              bool scale_grad_by_freq, int64_t mode,
                              bool sparse) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding_bag", indices_arg, kLong);
  checkContiguous("embedding_bag", indices_arg);
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarType("embedding_bag", offsets_arg, kLong);
  checkContiguous("embedding_bag", offsets_arg);
  auto offset2bag_arg = TensorArg(offset2bag, "offset2bag", 1);
  checkScalarType("embedding_bag", offset2bag_arg, kLong);
  checkContiguous("embedding_bag", offset2bag_arg);

  if (sparse) {
    return at::_embedding_bag_sparse_backward(
        grad, indices, offsets, offset2bag, bag_size_, num_weights,
        scale_grad_by_freq, mode);
  } else {
    return at::_embedding_bag_dense_backward(
        grad, indices, offsets, offset2bag, bag_size_, max_indices_, num_weights,
        scale_grad_by_freq, mode);
  }
}

static Tensor _embedding_bag_dense_backward_cpu_max(
    const Tensor& grad,
    const Tensor& bag_size,
    const Tensor& max_indices,
    int64_t num_weights) {
  AT_ASSERT(max_indices.defined());
  auto index_grad_weight =
      at::zeros({num_weights, grad.size(1)}, grad.options());
  auto nonempty_max_indices = max_indices.index_select(0, bag_size.nonzero().view(-1));
  auto nonempty_grad = grad.index_select(0, bag_size.nonzero().view(-1));

  for (int64_t dim = 0; dim < grad.size(1); dim++) {
    index_grad_weight.select(1, dim).index_add_(
      0, nonempty_max_indices.select(1, dim), nonempty_grad.select(1, dim));
  }
  return index_grad_weight;
}

static std::vector<int64_t> compute_counts(
    int64_t num_weights,
    int64_t* indices_data,
    int64_t indices_length) {
  std::vector<int64_t> counts(num_weights, 0);
  for (int i = 0; i < indices_length; i++) {
    counts[indices_data[i]]++;
  }
  return counts;
}

// counts_uniq stores the index of the NEXT unique element
// of the (sorted) indices vector.
//
// For example:
// indices: [0, 0, 0, 1, 3, 3, 4]
// counts: [3, 1, 0, 2, 1, 0]
// counts_uniq: [3, 4, 6, 7]
//
// The unique indices can be found at index 0, 3, 4, 6.
static std::vector<int64_t> compute_counts_uniq(
    int64_t num_weights,
    int64_t* indices_data,
    int64_t indices_length,
    const std::vector<int64_t>& counts) {
  std::vector<int64_t> counts_uniq;
  counts_uniq.reserve(num_weights);
  int64_t o = 0;
  for (int64_t i = 0; i < indices_length; i += counts[indices_data[i]]) {
    counts_uniq.push_back(counts[indices_data[i]]);
    if (o > 0) {
      counts_uniq[o] += counts_uniq[o - 1];
    }
    o++;
  }
  return counts_uniq;
}

Tensor _embedding_bag_dense_backward_cpu(const Tensor &grad_, const Tensor &indices_,
                                  const Tensor &offsets_,
                                  const Tensor &offset2bag__,
                                  const Tensor &bag_size_,
                                  const Tensor& max_indices_, int64_t num_weights,
                                  bool scale_grad_by_freq, int64_t mode) {
  // indices_, offsets_ and offset2bag__ are assumed having correct dtypes and
  // contiguous here due to the checks in _embedding_bag_backward above.
  // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
  // for more details.

  auto grad = grad_.contiguous();
  auto grad_arg = TensorArg(grad, "grad_", 1);
  checkScalarTypes("embedding_bag", grad_arg, {kFloat, kDouble});

  if (mode == MODE_MAX) {
    return _embedding_bag_dense_backward_cpu_max(
        grad_, bag_size_, max_indices_, num_weights);
  }
  AT_ASSERT(mode == MODE_MEAN || mode == MODE_SUM);

  auto index_grad_weight =
      at::zeros({num_weights, grad.size(1)}, grad.options());

  Tensor &offset2bag_ = const_cast<Tensor &>(offset2bag__);

  auto ind_sort_ = indices_.sort();
  auto indices = std::get<0>(ind_sort_);
  auto ind_sort = std::get<1>(ind_sort_);
  auto offset2bag = offset2bag_.index_select(0, ind_sort);

  auto indices_data = indices.data<int64_t>();
  auto offsets_data = offsets_.data<int64_t>();
  auto offset2bag_data = offset2bag.data<int64_t>();
  int64_t numel = indices.numel();

  auto counts = compute_counts(num_weights, indices_data, numel);
  auto next_unique_index_idx =
      compute_counts_uniq(num_weights, indices_data, numel, counts);

  #pragma omp parallel for if (numel > 1000)
    for (int64_t i = 0; i < (int64_t)next_unique_index_idx.size(); i++) {
      int64_t start = i == 0 ? 0 : next_unique_index_idx[i - 1];
      int64_t index = indices_data[start];
      for (int64_t j = start; j < next_unique_index_idx[i]; j++) {
        int64_t source = offset2bag_data[j];
        double scale = 1.0;
        if (scale_grad_by_freq) {
          scale /= counts[indices_data[i]];
        }
        if (mode == 1) { // MODE_MEAN
          if (offsets_.size(0) == 1) {
            auto bag_size = indices.size(0);
            scale /= bag_size;
          } else {
            if (source == offsets_.size(0) - 1) {
              scale /= indices.size(0) - offsets_data[offsets_.size(0) - 1];
            } else {
              scale /= offsets_data[source + 1] - offsets_data[source];
            }
          }
        }
        int64_t ddim = grad.size(1);
        AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "embedding_bag_backward", [&] {
          auto igwd = index_grad_weight.data<scalar_t>();
          auto gd = grad.data<scalar_t>();
          THBlas_axpy<scalar_t>(ddim, (scalar_t)scale, gd + ddim * source, 1,
                      igwd + ddim * index, 1);
        });
      }
    }

  return index_grad_weight;
}

Tensor _embedding_bag_sparse_backward(
    const Tensor &grad_, const Tensor &indices, const Tensor &offsets,
    const Tensor &offset2bag, const Tensor &bag_size_, int64_t num_weights,
    bool scale_grad_by_freq, int64_t mode) {
  // indices, offsets and offset2bag are assumed having correct dtypes and
  // contiguous here due to the checks in _embedding_bag_backward above.
  // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
  // for more details.

  Tensor grad = grad_;
  Tensor index_grad = grad_.index_select(0, offset2bag);
  index_grad = apply_bag_size_backward(offsets, indices, mode, index_grad,
                                       offset2bag, bag_size_);
  return native::embedding_backward(index_grad, indices, num_weights, -1,
                                    scale_grad_by_freq, true);
}
}
} // namespace at::native
