#include <torch/csrc/autograd/named.h>
#include <ATen/ATen.h>

namespace torch { namespace autograd { namespace named {

static bool is_alpha_or_underscore(char ch) {
  return ch == '_' || ('a' <= ch && ch <= 'z') ||
      ('A' <= ch && ch <= 'Z');
}

static bool is_valid_name(const std::string& name) {
  if (name.length() == 0) {
    return false;
  }
  for (auto it = name.begin(); it != name.end(); ++it) {
    if (not is_alpha_or_underscore(*it)) {
      return false;
    }
  }
  return true;
}

std::pair<NameType,c10::optional<std::string>>
check_valid_name(const std::string& name) {
  std::string delimiter = "."; 
  auto it = name.find(delimiter);
  if (it == std::string::npos) {
    AT_CHECK(is_valid_name(name), "Invalid name '", name, "': ", 
        "Normal names may only contain alphabetical characters and underscore.");
    return { NameType::NORMAL, c10::nullopt };
  }

  auto name_without_tag = name.substr(0, it);
  auto tag = name.substr(it + 1);
  AT_CHECK(is_valid_name(name_without_tag),
      "Invalid name '", name, "': for name '", name_without_tag, "': ", 
      "Names may only contain alphabetical characters and underscore.");
  AT_CHECK(is_valid_name(tag), "Invalid name '", name, "': invalid tag '", tag, "': ", 
      "Tags may only contain alphabetical characters and underscore.");
  return { NameType::TAGGED, name_without_tag };
}

c10::optional<DimName> unify(DimName dimname, DimName other) {
  if (other.is_wildcard()) {
    return dimname;
  }
  if (dimname.is_wildcard()) {
    return unify(other, dimname);
  }
  if (dimname.name() == other.name()) {
    return dimname;
  }
  if (dimname.basename() == other.basename()) {
    return DimName(dimname.basename());
  }
  return c10::nullopt;
}

bool match(DimName dimname, DimName other) {
  return unify(dimname, other).has_value();
}

}}}
