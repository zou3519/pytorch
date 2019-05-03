#pragma once
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <c10/util/Optional.h>
#include <c10/core/TensorImpl.h>

#include <ATen/ATen.h>

namespace torch { namespace autograd { namespace named {

using InternedName = uint64_t;

enum class NameType: uint8_t { INVALID, NORMAL, WILDCARD, TAGGED }; 

struct DimName {
  explicit DimName(): name_(0), type_(NameType::WILDCARD) {}
  DimName(InternedName name) : name_(name), type_(NameType::NORMAL) {}
  DimName(InternedName name, InternedName name_without_tag) : 
    name_(name), name_without_tag_(name_without_tag), type_(NameType::TAGGED) {}

  operator InternedName() const { return this->name_; }
  NameType type() const { return this->type_; }
  InternedName name() const { return this->name_; }
  bool is_wildcard() const { return this->type_ == NameType::WILDCARD; }
  InternedName basename() const {
    if (this->type_ == NameType::TAGGED) {
      return this->name_without_tag_.value();
    }
    return this->name_;
  }

 private:
  InternedName name_;
  c10::optional<InternedName> name_without_tag_;
  NameType type_;
};

std::pair<NameType,c10::optional<std::string>>
check_valid_name(const std::string& name);

c10::optional<DimName> unify(DimName name, DimName other);
bool match(DimName name, DimName other);

struct NamedMeta : public c10::NamedMetaInterface {
  std::vector<DimName> names;

  bool is_named() const override {
    return !std::all_of(
        names.begin(), names.end(), [](const DimName& n) { return n.is_wildcard(); });
  }
};

bool all_unnamed(at::TensorList tensors);

}}}
