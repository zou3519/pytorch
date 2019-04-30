#include <torch/csrc/python_headers.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/named.h>

namespace torch { namespace autograd { namespace named {

struct DimNamesTable {
  DimName offer(PyObject* name);
  const std::string& lookup_string(InternedName name);

 private:
  DimName register_pyname(PyObject* name);
  std::pair<std::string,DimName> pyobj2dimname(PyObject* name);

  std::unordered_map<InternedName, std::pair<std::string,DimName>> records_;
  std::unordered_map<std::string,InternedName> records_interned_;
  std::mutex records_mutex_;
};

}}}
