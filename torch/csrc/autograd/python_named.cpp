#include <torch/csrc/autograd/python_named.h>
#include <ATen/ATen.h>
#include <torch/csrc/utils/python_strings.h>

namespace torch { namespace autograd { namespace named {

DimName DimNamesTable::offer(PyObject* name) {
  std::lock_guard<std::mutex> lock(this->records_mutex_);
  const auto it = this->records_.find(reinterpret_cast<InternedName>(name));
  if (it == this->records_.end()) {
    return it->second.second;
  }
  return this->register_pyname(name);
}

std::pair<std::string,DimName> DimNamesTable::pyobj2dimname(PyObject* name) {
  if (name == Py_None) {
    return { "*", DimName() };
  }
  AT_CHECK(THPUtils_checkString(name), "Expected None or string.");

  NameType type;
  c10::optional<std::string> maybe_name_without_tag;

  auto namestr = THPUtils_unpackString(name);
  std::tie(type, maybe_name_without_tag) = check_valid_name(THPUtils_unpackString(name));
  auto interned = reinterpret_cast<InternedName>(name);
  if (type == NameType::NORMAL) {
    return { std::move(namestr), DimName(interned) };
  }
  if (type == NameType::TAGGED) {
    AT_ASSERT(maybe_name_without_tag);
    const auto it = this->records_interned_.find(maybe_name_without_tag.value());
    AT_CHECK(it == this->records_interned_.end(),
        "TODO: get python to intern the name first");
    InternedName interned_without_tag = it->second;
    return { std::move(namestr), DimName(interned, interned_without_tag) };
  }
  AT_ASSERT(false);
}

DimName DimNamesTable::register_pyname(PyObject* name) {
  // Assumes we have records_mutex already
  auto interned = reinterpret_cast<InternedName>(name);
  auto result = this->pyobj2dimname(name);
  this->records_[interned] = result;
  this->records_interned_[result.first] = interned;
  return result.second;
}

const std::string& DimNamesTable::lookup_string(InternedName name) {
  std::lock_guard<std::mutex> lock(this->records_mutex_);

  const auto it = this->records_.find(name);
  AT_CHECK(it != this->records_.end(), "DimNameTable::lookup_string: name not found");

  return it->second.first;
}

}}}
