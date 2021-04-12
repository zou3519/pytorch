#include <torch/library.h>
#include <ATen/DynamicLayer.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/autograd/variable.h>
#include <ATen/TensorWrapper.h>

namespace at {

// Initial autograd layer, because autograd is always "on"
thread_local std::vector<DynamicLayer> dynamicLayerStack = { DynamicLayer(DispatchKey::Autograd, 1) };

DynmetaData kDynMetaDataSingleton;

DynmetaData& getGlobalDynmetaData() {
  return kDynMetaDataSingleton;
}

optional<DynamicLayer> maybeCurrentDynamicLayer() {
  // NB: Exception for regular autograd, maybe tweak this
  if (dynamicLayerStack.size() <= 1) {
    return {};
  }
  return dynamicLayerStack.back();
}

const std::vector<DynamicLayer>& getDynamicLayerStack() {
  return dynamicLayerStack;
}

void setDynamicLayerStack(const std::vector<DynamicLayer>& stack) {
  dynamicLayerStack = stack;
}

static DynamicLayer popDynamicLayer() {
  TORCH_INTERNAL_ASSERT(dynamicLayerStack.size() > 0);
  auto result = dynamicLayerStack.back();
  TORCH_INTERNAL_ASSERT(result.key() != DispatchKey::Undefined);
  dynamicLayerStack.pop_back();

  if (dynamicLayerStack.size() == 0) {
    // std::cout << "DynamicLayer off" << std::endl;
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerFront, false);
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerBack, false);
  }

  return result;
}

static int64_t pushDynamicLayer(DispatchKey key) {
  TORCH_INTERNAL_ASSERT(key != DispatchKey::Undefined);
  auto layerId = 1 + dynamicLayerStack.size();
  dynamicLayerStack.emplace_back(key, layerId);

  if (layerId == 2) {
    // std::cout << "DynamicLayer on" << std::endl;
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerFront, true);
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerBack, true);
  }

  return layerId;
}

int64_t initAndPushDynamicLayer(DispatchKey key) {
  auto layerId = pushDynamicLayer(key);
  auto& data = getGlobalDynmetaData();
  TORCH_INTERNAL_ASSERT(data.find(layerId) == data.end());
  data[layerId] = std::make_shared<bool>(true);
  return layerId;
}

DynamicLayer popDynamicLayerAndDeleteMetadata() {
  auto result = popDynamicLayer();
  auto level = result.layerId();

  // TODO: is this lock safe? No one else should be writing to the same bucket
  if (c10::show_dispatch_trace_enabled()) {
    std::cout << "deleting metadata" << std::endl;
  }
  auto& data = getGlobalDynmetaData();
  auto it = data.find(level);
  if (it == data.end()) {
    return result;
  }
  if (c10::show_dispatch_trace_enabled()) {
    std::cout << "deleted metadata for level " << level << std::endl;
  }
  // invalidate the thing
  *(it->second) = false;
  data.erase(level);
  return result;
}

static Tensor materializeGradWrappers(const Tensor& tensor, const std::vector<DynamicLayer>& dynlayerStack) {
  if (!tensor.defined()) {
    return tensor;
  }
  // TODO: First entry in the stack is a default autograd key.
  // We should clean up the logic
  if (dynlayerStack.size() <= 1) {
    return tensor;
  }
  if (dynlayerStack.back().key() != DispatchKey::Autograd) {
    return tensor;
  }
  auto cur_level = dynlayerStack.back().layerId();
  auto* wrapper = maybeGetTensorWrapper(tensor);
  if (!wrapper) {
    return makeTensorWrapper(tensor, cur_level);
  }
  TORCH_INTERNAL_ASSERT(wrapper->level() <= cur_level, "escaped?");
  if (wrapper->level() == cur_level) {
    TORCH_INTERNAL_ASSERT(tensor.defined());
    return tensor;
  }
  return makeTensorWrapper(tensor, cur_level);
}

static Tensor unwrapIfDead(const Tensor& tensor) {
  auto* wrapped = maybeGetTensorWrapper(tensor);
  if (!wrapped) {
    return tensor;
  }
  if (wrapped->is_alive()) {
    return tensor;
  }
  return wrapped->value();
}

static void foreachTensorInplace(std::vector<IValue>& args, int64_t begin, int64_t end,
    std::function<Tensor(const Tensor&)> func) {
  TORCH_INTERNAL_ASSERT(begin >= 0);
  TORCH_INTERNAL_ASSERT(end >= 0);
  TORCH_INTERNAL_ASSERT(begin <= end);
  for (int64_t idx = begin; idx < end; idx++) {
    auto ivalue = args[idx];
    if (ivalue.isTensorList()) {
      auto list = ivalue.toTensorList();
      for (int64_t list_idx = 0; list_idx < list.size(); list_idx++) {
        list[list_idx] = func(list[list_idx]);
      }
      args[idx] = list;
    }
    TORCH_INTERNAL_ASSERT(!ivalue.isGenericDict(), "No operators can accept GenericDict");
    if (!ivalue.isTensor()) {
      continue;
    }
    Tensor value = ivalue.toTensor();
    Tensor replacement = func(value);
    args[idx] = std::move(replacement);
    // sanity checks
    if (ivalue.toTensor().defined()) {
      TORCH_INTERNAL_ASSERT(args[idx].toTensor().defined());
    }
  }
}

constexpr DispatchKeySet all_dynlayer_keyset = DispatchKeySet({
  DispatchKey::DynamicLayerFront,
  DispatchKey::DynamicLayerBack,
  DispatchKey::TensorWrapper,
  DispatchKey::Batched,
  DispatchKey::InplaceOrView
}) | autograd_dispatch_keyset;

static void sanityCheckStack(torch::jit::Stack* stack) {
  if (stack->size() > 0) {
    auto last_ivalue = (*stack)[stack->size() - 1];
    if (last_ivalue.isTensor()) {
      auto tensor = last_ivalue.toTensor();
      auto* wrapper = maybeGetTensorWrapper(tensor);
      TORCH_INTERNAL_ASSERT(wrapper == nullptr);
      TORCH_INTERNAL_ASSERT(tensor.has_storage());
    }
  }
}

void dynamicLayerFrontFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  if (c10::show_dispatch_trace_enabled()) {
    std::cout << "DLS size: " << dynamicLayerStack.size() << std::endl;
  }
  if (dynamicLayerStack.size() == 0) {
    sanityCheckStack(stack);
    c10::impl::ExcludeDispatchKeyGuard guard(all_dynlayer_keyset);
    op.callBoxed(stack);
    return;
  }

  // Unwrap dead GradWrappers, materialize live ones
  auto maybeTransformGradWrappers = [](const Tensor& tensor) {
    auto result = unwrapIfDead(tensor);
    return materializeGradWrappers(result, getDynamicLayerStack());
  };
  auto num_args = op.schema().arguments().size();
  foreachTensorInplace(*stack, stack->size() - num_args, stack->size(), maybeTransformGradWrappers);

  auto layer = dynamicLayerStack.back();

  DispatchKeySet exclude = DispatchKeySet::FULL;
  exclude = exclude.remove(DispatchKey::DynamicLayerBack);
  if (layer.key() == DispatchKey::Autograd) {
    exclude = exclude - autograd_dispatch_keyset;
    exclude = exclude.remove(DispatchKey::InplaceOrView);
  } else {
    exclude = exclude.remove(layer.key());
  }
  c10::impl::ExcludeDispatchKeyGuard guard(exclude);

  // Re-dispatch
  op.callBoxed(stack);
}

struct WithoutTop {
  WithoutTop(): layer_(popDynamicLayer()) {
  }
  ~WithoutTop() {
    pushDynamicLayer(layer_.key());
  }

  bool prev_grad_enabled_;
  DynamicLayer layer_;
};

void dynamicLayerBackFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto cur_level = getDynamicLayerStack().back().layerId();
  auto cur_key = getDynamicLayerStack().back().key();

  auto unwrap = [&](const Tensor& tensor) {
    if (!tensor.defined()) {
      return tensor;
    }
    auto* maybe_tensor_wrapper = maybeGetTensorWrapper(tensor);
    if (!maybe_tensor_wrapper) {
      return tensor;
    }
    if (maybe_tensor_wrapper->level() == cur_level) {
      return maybe_tensor_wrapper->value();
    }
    if (c10::show_dispatch_trace_enabled()) {
      std::cout << "unwrap " << cur_level << std::endl;
    }
    return tensor;
  };
  auto wrap = [&](const Tensor& tensor) {
    if (!tensor.defined()) {
      return tensor;
    }
    if (cur_level == 1) {
      return tensor;
    }
    if (c10::show_dispatch_trace_enabled()) {
      std::cout << "wrap " << cur_level << std::endl;
    }
    return makeTensorWrapper(tensor, cur_level);
  };

  // Hack for autograd key: Unwrap everything
  if (cur_key == DispatchKey::Autograd) {
    auto args_size = op.schema().arguments().size();
    foreachTensorInplace(*stack, stack->size() - args_size, stack->size(), unwrap);
  }

  // pop the top layer. Put it back on dtor.
  WithoutTop guard;

  // "reset exclude set"
  // TODO: Still a problem with composabiilty and AutoNonVariableTypeGuard.
  auto keyset = c10::impl::PODLocalDispatchKeySet();
  c10::impl::_force_tls_local_dispatch_key_set(keyset);
  c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerFront, true);
  c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerBack, true);


  // Re-dispatch
  op.callBoxed(stack);

  // Hack for autograd key: Rewrap everything
  if (cur_key == DispatchKey::Autograd) {
    auto ret_size = op.schema().returns().size();
    foreachTensorInplace(*stack, stack->size() - ret_size, stack->size(), wrap);
  }
}

TORCH_LIBRARY_IMPL(_, DynamicLayerFront, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dynamicLayerFrontFallback>());
}

TORCH_LIBRARY_IMPL(_, DynamicLayerBack, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dynamicLayerBackFallback>());
}

TORCH_LIBRARY_IMPL(aten, DynamicLayerFront, m) {
  m.impl("_unwrap_for_grad", native::_unwrap_for_grad);
  m.impl("dump_tensor", native::dump_tensor);
  m.impl("dlevel", native::dlevel);
}

} // namespace at
