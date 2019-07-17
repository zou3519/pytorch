#ifdef BUILD_NAMEDTENSOR

#include <ATen/NamedTensorUtils.h>
#include <sstream>

namespace at {

void internal_set_names_inplace(Tensor& tensor, optional<DimnameList> names) {
  impl::internal_set_names_inplace(tensor.unsafeGetTensorImpl(), names);
}

// Returns "Tensor['N', 'C', 'H', 'W']" for a tensor with names ('N', 'C', 'H', 'W').
static std::string toDimnameRepr(const Tensor& tensor) {
  std::ostringstream os;
  os << "Tensor";
  if (tensor.names() == nullopt) {
    os << "[";
    for (auto i = 0; i < tensor.dim(); i++) {
      if (i != 0) os << ", ";
      os << "None";
    }
    os << "]";
  } else {
    os << *tensor.names();
  }
  return os.str();
}

int64_t dimname_to_position(const Tensor& tensor, Dimname dim) {
  TORCH_CHECK(dim.type() != NameType::WILDCARD,
      "Please look up dimensions by name, got: name = None.");
  TORCH_CHECK(tensor.names().has_value(),
      "Name ", dim, " not found in ", toDimnameRepr(tensor), ".");
  const auto names = *tensor.names();

  const auto it = std::find_if(
      names.begin(), names.end(),
      [&dim](const Dimname& candidate) { return dim.can_refer_to(candidate); });
  TORCH_CHECK(it != names.end(),
      "Name ", dim, " not found in ", toDimnameRepr(tensor), ".");

  // Check that it can't refer to another dimension
  const auto dup = std::find_if(
      it + 1, names.end(),
      [&dim](const Dimname& candidate) { return dim.can_refer_to(candidate); });
  TORCH_CHECK(
      dup == names.end(),
      "Name ", dim, " could refer to multiple dimensions in ",
      toDimnameRepr(tensor), ". Please disambiguate by using a more ",
      "specific name like ", *it, " or ", dup, ".");
  return std::distance(names.begin(), it);
}

static void report_positional_error(
    const Dimname& name,
    const Dimname& other_name,
    DimnameList names,
    DimnameList other_names) {
  // TODO(zou3519): Can improve message by checking if names are alignable and suggesting workarounds
  TORCH_CHECK(false,
      "Names ", name, " and ", other_name, " do not match positionally ",
      "from the right in names ", names, " and ", other_names, ".");
}

static void check_for_misalignment(
    const Dimname& name,
    DimnameList names,
    DimnameList other_names) {
  if (name.is_wildcard()) {
    return;
  }
  auto it = std::find_if(other_names.begin(), other_names.end(),
      [&](const Dimname& candidate) { return name.can_refer_to(candidate); });
  // TODO(zou3519): Can improve message by checking if names are alignable and suggesting workarounds
  TORCH_CHECK(it == other_names.end(),
      "Names ", names, " and ", other_names, " are misaligned: name ", name,
      " appears in a different position from the right.");
}

// Assumption: A DimnameList can have no duplicate full names with
// the exception of wildcards
static std::vector<Dimname> unify_from_right(DimnameList names, DimnameList other_names) {
  const auto wildcard = Dimname::wildcard();
  const auto size = std::max(names.size(), other_names.size());
  auto result = std::vector<Dimname>(size, wildcard);

  auto names_it = names.rbegin();
  auto other_it = other_names.rbegin();
  auto result_it = result.rbegin();
  while (names_it != names.rend() || other_it != other_names.rend()) {
    // TODO(zou3519): Don't support tagged names for now. They're a little weird.
    if (names_it->is_tagged() || other_it->is_tagged()) {
      TORCH_INTERNAL_ASSERT("unify_from_right: NYI: tagged names.");
    }

    const auto& name = names_it == names.rend() ? wildcard : *names_it;
    const auto& other_name = other_it == other_names.rend() ? wildcard : *other_it;

    // Step 1: Check that the names match
    const auto maybeName = unify(name, other_name);
    if (!maybeName) {
      report_positional_error(name, other_name, names, other_names);
    }
    *result_it = *maybeName;

    // Step 2: Check that the names are not misaligned
    if (!names_it->is_normal() || !other_it->is_normal()) {
      // Let: N = max(len(names), len(other_names))
      //      K = # of special names among names and other_names.
      // This search (including the outer loop) is O(N*K) but typically # of dims is small.
      check_for_misalignment(name, names, other_names);
      check_for_misalignment(other_name, other_names, names);
    }

    if (names_it != names.rend()) {
      ++names_it;
    }
    if (other_it != other_names.rend()) {
      ++other_it;
    }
    ++result_it;
  }
  return result;
}

bool are_none_names_matched(DimnameList first, DimnameList second) {
  auto it_first = first.rbegin();
  auto it_second = second.rbegin();
  while (it_first != first.rend() && it_second != second.rend()) {
    if (it_first->is_wildcard() ^ it_second->is_wildcard()) {
      return false;
    }
    ++it_first;
    ++it_second;
  }
  return true;
}

bool is_subsequence(DimnameList shorter, DimnameList longer) {
  auto it_longer = longer.begin();
  auto it_shorter = shorter.begin();
  for(; it_longer != longer.end() && it_shorter != shorter.end(); ++it_longer) {
    TORCH_INTERNAL_ASSERT(
        !it_longer->is_tagged() && !it_shorter->is_tagged(),
        "Tagged names NYI");
    if (it_longer->full_name() == it_shorter->full_name()) {
      ++it_shorter;
    }
  }
  // If everything in `shorter` was found in `longer`
  return it_shorter == shorter.end();
}

DimnameList infer_alignment(DimnameList first, DimnameList second) {
  TORCH_CHECK(
      are_none_names_matched(first, second),
      "Cannot infer the alignment of Tensor", first, " and Tensor", second,
      ": please name unnamed dimensions. Otherwise, the unnamed dimensions should ",
      "occur in the same position from the right for both tensors.");

  size_t dims_first = first.size();
  size_t dims_second = second.size();

  DimnameList longer = dims_first > dims_second ? first : second;
  DimnameList shorter = dims_first > dims_second ? second : first;

  TORCH_CHECK(
      is_subsequence(shorter, longer),
      "Could not unambiguously align tensor a with dims (", first,
      ") and tensor b with dims (", second, ") because the dims of one "
      "tensor are not a subsequence of the dims of the other.");
  return longer;
}

Tensor align_to(const Tensor& tensor, DimnameList names) {
  auto tensor_sizes = tensor.sizes();
  TORCH_CHECK(
      tensor.is_named() || tensor_sizes.size() == 0,
      "align_to: input tensor must have named dimensions.");
  TORCH_CHECK(
      names.size() >= tensor.dim(),
      "Cannot align tensor with dims (", tensor.names().value(),
      ") to a shorter set of dims (", names, ").");

  std::vector<int64_t> expanded_sizes(names.size(), 1);
  auto tensor_names = *tensor.names();
  ptrdiff_t dim = (ptrdiff_t)tensor.dim() - 1;
  ptrdiff_t idx = (ptrdiff_t)names.size() - 1;
  for (; idx >= 0 && dim >= 0; --idx) {
    TORCH_INTERNAL_ASSERT(
        !tensor_names[dim].is_tagged() && !names[idx].is_tagged(),
        "Tagged names NYI");
    if (tensor_names[dim].full_name() != names[idx].full_name()) {
      continue;
    }
    expanded_sizes[idx] = tensor_sizes[dim];
    --dim;
  }
  TORCH_CHECK(
      dim == -1,
      "Could not align tensor `a` with dims (", tensor_names,
      ") to `names` (", names, ") because `a.names` is not a subsequence of `names`.");

  auto result = tensor.view(expanded_sizes);
  at::internal_set_names_inplace(result, names);
  return result;
}

// Assumption: A DimnameList can have no duplicate full names with
// the exception of wildcards
CAFFE2_API optional<std::vector<Dimname>>
unify_from_right(optional<DimnameList> names, optional<DimnameList> other_names) {
  if (!names && !other_names) {
    return nullopt;
  }
  if (!names) {
    return other_names.value().vec();
  }
  if (!other_names) {
    return names.value().vec();
  }
  return unify_from_right(*names, *other_names);
}


namespace namedinference {

optional<std::vector<Dimname>> erase_name(optional<DimnameList> self_names, int64_t dim) {
  if (self_names == nullopt) {
    return nullopt;
  }
  auto outnames = self_names->vec();
  outnames.erase(outnames.begin() + dim);
  return outnames;
}

void propagate_names(Tensor& result, const Tensor& src) {
  at::internal_set_names_inplace(result, src.names());
}

void propagate_names(TensorImpl* result, TensorImpl* src) {
  const auto names = at::impl::internal_get_names(src);
  at::impl::internal_set_names_inplace(result, names);
}

std::tuple<Tensor,Tensor,optional<DimnameList>>
align_names(const Tensor& tensor, const Tensor& other) {
  optional<DimnameList> outnames = nullopt;
  if (!tensor.is_named() and !other.is_named()) {
    // optimization
    return std::tuple<Tensor,Tensor,optional<DimnameList>>(tensor, other, outnames);
  }
  if (tensor.names().has_value() ^ other.names().has_value()) {
    // TODO(rzou): Don't materialize names
    auto& troublesome = tensor.names() ? other : tensor;
    auto& not_troublesome = tensor.names() ? tensor : other;
    auto materialized_names = std::vector<Dimname>(troublesome.dim(), Dimname::wildcard());
    outnames = infer_alignment(materialized_names, *not_troublesome.names());
  } else {
    outnames = infer_alignment(*tensor.names(), *other.names()); 
  }
  auto tensor_ = tensor.dim() == outnames->size() ? tensor : align_to(tensor, *outnames);
  auto other_ = other.dim() == outnames->size() ? other : align_to(other, *outnames);
  return std::tuple<Tensor,Tensor,optional<DimnameList>>(tensor_, other_, outnames);
}

} // namespace namedinference
} // namespace at
#endif
