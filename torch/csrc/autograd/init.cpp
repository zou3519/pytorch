#include <torch/csrc/python_headers.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <ATen/profiler.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/function.h>

PyObject * THPAutograd_initExtension(PyObject *_unused)
{
  auto tensor_module = THPObjectPtr(PyImport_ImportModule("torch.tensor"));
  if (!tensor_module) throw python_error();

  // NOTE: "leaks" THPVariableClass
  THPVariableClass = PyObject_GetAttrString(tensor_module, "Tensor");
  if (!THPVariableClass) throw python_error();

  auto autograd_module = THPObjectPtr(PyImport_ImportModule("torch.autograd"));
  if (!autograd_module) throw python_error();

  // NOTE: "leaks" Function
  THPFunctionClass = PyObject_GetAttrString(autograd_module, "Function");
  if (!THPFunctionClass) throw python_error();

  auto m = py::handle(autograd_module).cast<py::module>();

  py::class_<at::profiler::Event>(m, "ProfilerEvent")
      .def("kind", &at::profiler::Event::kind)
      .def(
          "name",
          [](const at::profiler::Event& e) { return e.name(); })
      .def("thread_id", &at::profiler::Event::thread_id)
      .def("device", &at::profiler::Event::device)
      .def("cpu_elapsed_us", &at::profiler::Event::cpu_elapsed_us)
      .def(
          "cuda_elapsed_us", &at::profiler::Event::cuda_elapsed_us)
      .def("has_cuda", &at::profiler::Event::has_cuda);
  py::enum_<at::profiler::ProfilerState>(m,"ProfilerState")
  .value("Disabled", at::profiler::ProfilerState::Disabled)
  .value("CPU", at::profiler::ProfilerState::CPU)
  .value("CUDA", at::profiler::ProfilerState::CUDA)
  .value("NVTX", at::profiler::ProfilerState::NVTX);

  m.def("_enable_profiler", at::profiler::enableProfiler);
  m.def("_disable_profiler", at::profiler::disableProfiler);

  m.def("_push_range", [](std::string name) {
      at::profiler::pushRange(std::move(name));
  });
  m.def("_pop_range", []() { at::profiler::popRange(); });

  Py_RETURN_TRUE;
}

namespace torch { namespace autograd {

static PyObject * set_grad_enabled(PyObject* _unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  if (!PyBool_Check(arg)) {
    throw TypeError("enabled must be a bool (got %s)", Py_TYPE(arg)->tp_name);
  }
  GradMode::set_enabled(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * is_grad_enabled(PyObject* _unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  if (GradMode::is_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject * set_anomaly_mode_enabled(PyObject* _unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  if (!PyBool_Check(arg)) {
    throw TypeError("enabled must be a bool (got %s)", Py_TYPE(arg)->tp_name);
  }
  AnomalyMode::set_enabled(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * is_anomaly_mode_enabled(PyObject* _unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  if (AnomalyMode::is_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// autograd methods on torch._C
static PyMethodDef methods[] = {
  {"set_grad_enabled", (PyCFunction)set_grad_enabled, METH_O, nullptr},
  {"is_grad_enabled", (PyCFunction)is_grad_enabled, METH_NOARGS, nullptr},
  {"set_anomaly_enabled", (PyCFunction)set_anomaly_mode_enabled, METH_O, nullptr},
  {"is_anomaly_enabled", (PyCFunction)is_anomaly_mode_enabled, METH_NOARGS, nullptr},
  {nullptr, nullptr, 0, nullptr}
};

PyMethodDef* python_functions() {
  return methods;
}

}} // namespace torch::autograd
