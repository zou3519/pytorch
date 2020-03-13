#include <ATen/NestedTensor.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace at {

static void NestedTensor_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK(false, "NestedTensor NYI: ", op.schema(), ".");
}

Tensor NestedTensor_conv2d(const Tensor& input, const Tensor& weight,
                           const Tensor& bias, IntArrayRef stride,
                           IntArrayRef padding, IntArrayRef dilation,
                           int64_t groups) {
  // Simple for-loop for now.
  auto* nested = static_cast<NestedTensorImpl*>(input.unsafeGetTensorImpl());
  std::vector<Tensor> results;
  for (const auto& tensor : nested->rep_) {
    auto result = at::conv2d(tensor, weight, bias, stride, padding, dilation, groups);
    results.push_back(std::move(result));
  }
  return native::_make_nested(results);
}

std::vector<Tensor> NestedTensor_unbind(const Tensor& self, int64_t dim) {
  TORCH_INTERNAL_ASSERT(dim == 0);
  auto* nested = static_cast<NestedTensorImpl*>(self.unsafeGetTensorImpl());
  return nested->rep_;  // copy
}

// Operator registrations
auto NestedTensor_registry = c10::Dispatcher::singleton().registerBackendFallbackKernel(
    DispatchKey::NestedTensorId,
    KernelFunction::makeFromBoxedFunction<&NestedTensor_fallback>()
);

static auto NestedTensor_registry2 = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
      .schema("aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor")
      .impl_unboxedOnlyKernel<Tensor (const Tensor&, const Tensor&, const Tensor&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), &NestedTensor_conv2d>(DispatchKey::NestedTensorId))
  .op(torch::RegisterOperators::options()
      .schema("aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]")
      .impl_unboxedOnlyKernel<std::vector<Tensor> (const Tensor&, int64_t), &NestedTensor_unbind>(DispatchKey::NestedTensorId))
  // TODO: this should really be dispatch-key agnostic
  .op(torch::RegisterOperators::options()
      .schema("aten::_make_batched(Tensor self, int? batch_dim, int level) -> Tensor")
      .kernel(DispatchKey::NestedTensorId, &at::native::_make_batched))
  .op(torch::RegisterOperators::options()
      .schema("aten::size.int(Tensor self, int dim) -> int")
      .kernel(DispatchKey::NestedTensorId, [] (const Tensor& self, int64_t dim) -> int64_t {
        dim = maybe_wrap_dim(dim, self.dim());
        auto* nested = static_cast<NestedTensorImpl*>(self.unsafeGetTensorImpl());
        if (dim == 0) {
          return nested->rep_.size();
        } else {
          return nested->rep_[0].size(dim - 1);
        }
      }))
  ;


}
