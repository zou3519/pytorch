#include <ATen/core/EnableNamedTensor.h>
#include <ATen/TensorNames.h>
#include <ATen/WrapDimUtils.h>
#include <cstring>

namespace at { namespace namedinference {

#ifdef BUILD_NAMEDTENSOR

namespace {

// While broadcasting dims:
//   [ batch: 3, channel: 4, height: 4, width: 5]
//   [                        batch: 3, width: 5]
//                           ^^^^^^^^^
// Expected the names of these dimensions to match but they do not. 

// Helper struct to store a (name, size) tuple.
struct DimMeta {
  DimMeta(Dimname name, int64_t size) : name(name), size(size) {};
  Dimname name;
  int64_t size;
};

int64_t printlen(Dimname arg) {
  if (arg.isWildcard()) return 0;
  return strlen(arg.symbol().toUnqualString());
}

// Returns number of chars written
int64_t dumpDim(
    std::ostream& os,
    const DimMeta& dim,
    int64_t dimname_padding = 0,
    int64_t size_padding = 0) {
  // Add an extra space
  dimname_padding++;
  size_padding++;

  std::stringstream ss;
  if (dim.name.isWildcard()) {
    ss << std::string(dimname_padding, ' ') << /*to match ":" below*/" "
       << std::string(size_padding, ' ') << dim.size;
  } else {
    ss << std::string(dimname_padding, ' ') << dim.name.symbol().toUnqualString() << ":"
       << std::string(size_padding, ' ') << dim.size;
  }
  auto str = ss.str();
  os << str;
  return str.length();
}

std::tuple<int64_t,int64_t> computePadding(Dimname first, Dimname second) {
  int64_t first_len = printlen(first);
  int64_t second_len = printlen(second);
  int64_t max = std::max(first_len, second_len);
  return {max - first_len, max - second_len};
}

std::tuple<int64_t,int64_t> computePadding(int64_t first, int64_t second) {
  int64_t ndigits_first = floor(log10(first));
  int64_t ndigits_second = floor(log10(second));
  int64_t max = std::max(ndigits_first, ndigits_second);
  return {max - ndigits_first, max - ndigits_second};
}

void formatDim(
    std::ostream& first_line,
    optional<DimMeta> first_dim,
    std::ostream& second_line,
    optional<DimMeta> second_dim,
    std::ostream& third_line,
    bool show_err_marker) {
  int64_t chars_written = 0;
  if (first_dim && second_dim) {
    auto dimname_padding = computePadding(first_dim->name, second_dim->name);
    auto size_padding = computePadding(first_dim->size, second_dim->size);
    chars_written = dumpDim(
        first_line, *first_dim, std::get<0>(dimname_padding), std::get<0>(size_padding));
    dumpDim(
        second_line, *second_dim, std::get<1>(dimname_padding), std::get<1>(size_padding));
  } else if (first_dim && !second_dim) {
    chars_written = dumpDim(first_line, *first_dim);
    std::cout << "chars_written: " << chars_written << std::endl;
    second_line << std::string(chars_written, ' ');
  } else if (!first_dim && second_dim) {
    chars_written = dumpDim(second_line, *second_dim);
    first_line << std::string(chars_written, ' ');
  }
  if (show_err_marker) {
    third_line << std::string(chars_written, '^');
  } else {
    third_line << std::string(chars_written, ' ');
  }
}

}

// wrong_idx should be negative...
std::string unifyErrorMessage(
    DimnameList names1,
    IntArrayRef sizes1,
    DimnameList names2,
    IntArrayRef sizes2,
    int64_t wrong_idx) {
  std::stringstream tensor1_line;
  std::stringstream tensor2_line;
  std::stringstream error_line;

  tensor1_line << "[";
  tensor2_line << "[";
  error_line << " ";

  int64_t max_dim = std::max(names1.size(), names2.size());
  int64_t tensor1_offset = max_dim - names1.size();
  int64_t tensor2_offset = max_dim - names2.size();
  for (int64_t idx = 0; idx < max_dim; idx++) {
      int64_t tensor1_idx = idx - tensor1_offset;
      int64_t tensor2_idx = idx - tensor2_offset;
      if (tensor1_idx > 0 && tensor2_idx > 0) {
        tensor1_line << ",";
        tensor2_line << ",";
      } else if (tensor1_idx > 0 && tensor2_idx <= 0) {
        tensor1_line << ",";
        tensor2_line << " ";
      } else if (tensor1_idx <= 0 && tensor2_idx > 0) {
        tensor2_line << " ";
        tensor1_line << ",";
      }
      error_line << " ";
      std::cout << "tensor1_idx, tensor2_idx " << tensor1_idx << ", " << tensor2_idx << std::endl;
      formatDim(
          tensor1_line,
          tensor1_idx >= 0 ? make_optional(DimMeta(names1[tensor1_idx], sizes1[tensor1_idx]))
                           : nullopt,
          tensor2_line,
          tensor2_idx >= 0 ? make_optional(DimMeta(names2[tensor2_idx], sizes2[tensor2_idx]))
                           : nullopt,
          error_line,
          max_dim + wrong_idx == idx);
  }
  std::cout << "first: " << tensor1_line.str() << std::endl;
  std::cout << "second: " << tensor2_line.str() << std::endl;
  std::cout << "third: " << error_line.str() << std::endl;
  return "While broadcasting tensors with shapes:\n" + tensor1_line.str() + "]\n" + tensor2_line.str() + "]\n" + error_line.str();
}


Dimname TensorName::toDimname() const {
  return name_;
}

const TensorName& TensorName::unify(const TensorName& other, const char* op_name) const {
  // unify(None, None)
  if (name_.isWildcard() && other.name_.isWildcard()) {
    return *this;
  }

  // unify(A, A)
  if (name_ == other.name_) {
    return *this;
  }

  // unify(A, None)
  if (other.name_.isWildcard()) {
    const auto it = std::find(other.origin_.begin(), other.origin_.end(), name_);
    TORCH_CHECK(it == other.origin_.end(),
        op_name, ":",
        " Cannot match ", *this, " with ", other,
        " because the latter names already have ", name_, ".",
        " Are your tensors misaligned?");
    return *this;
  }

  // unify(None, A)
  if (name_.isWildcard()) {
    return other.unify(*this, op_name);
  }

  // unify(A, B)
  TORCH_CHECK(name_ == other.name_,
      op_name, ":",
      " Expected ", *this,
      " to match ", other,
      " but they do not match.");
  return *this;
}

TensorNames::TensorNames(ArrayRef<Dimname> names) {
  names_.reserve(names.size());
  for (int64_t idx = 0; idx < names.size(); ++idx) {
    names_.emplace_back(names, idx);
  }
}

TensorNames::TensorNames(ArrayRef<Dimname> names, int64_t start, int64_t end) {
  start = maybe_wrap_dim(start, names.size());
  end = maybe_wrap_dim(end, names.size());
  names_.reserve(end - start);
  for (int64_t idx = start; idx < end; ++idx) {
    names_.emplace_back(names, idx);
  }
}

TensorNames& TensorNames::unifyFromRightInplace(const TensorNames& other, const char* op_name) {
  int64_t size_diff = std::labs(names_.size() - other.names_.size());

  if (names_.size() > other.names_.size()) {
    for (int64_t idx = size_diff; idx < names_.size(); ++idx) {
      names_[idx] = names_[idx].unify(other.names_[idx - size_diff], op_name);
    }
  } else {
    // pad names_ to the same length as other.names_ before unification
    names_.insert(
        names_.begin(),
        other.names_.begin(),
        other.names_.begin() + size_diff);
    for (int64_t idx = size_diff; idx < names_.size(); ++idx) {
      names_[idx] = names_[idx].unify(other.names_[idx], op_name);
    }
  }

  return *this;
}

void TensorNames::append(TensorName&& name) {
  names_.emplace_back(name);
}

void TensorNames::checkUnique(const char* op_name) const {
  // O(N^2), but named tensors can have at most N = 64 dimensions, so this
  // doesn't matter unless benchmarking tells us it does. The alternative is
  // to create some sort of set data structure but the overhead of that
  // might dominate for small sizes.
  for (auto it = names_.begin(); it != names_.end(); ++it) {
    const auto name = it->toDimname();
    if (name.isWildcard()) continue;

    auto dup = std::find_if(it + 1, names_.end(),
        [&](const TensorName& other) { return other.toDimname() == name; });
    TORCH_CHECK(dup == names_.end(),
        op_name, ": ",
        "Attempted to propagate dims ", *it, " and ", *dup, " to the output, ",
        "but that would create a tensor with duplicate names [", toDimnameVec(),
        "]. Please rename your inputs with Tensor.rename to prevent this.");
  }
}

// Let's say the TensorName represents 'C' in ['N', 'C', 'H, 'W'].
// It should print like:
// 'C' (index 1 of ['N', 'C', 'H', 'W'])
std::ostream& operator<<(std::ostream& out, const TensorName& tensorname) {
  out << tensorname.name_ << " (index ";
  out << tensorname.origin_idx_ << " of ";
  out << tensorname.origin_ << ")";
  return out;
}

std::vector<Dimname> TensorNames::toDimnameVec() const {
  std::vector<Dimname> result;
  result.reserve(names_.size());
  for (const auto& tensor_name : names_) {
    result.emplace_back(tensor_name.toDimname());
  }
  return result;
}

#endif

}} // namespace at::namedinference
