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
std::mutex kDynMetaDataSingletonMutex;

DynmetaData& getGlobalDynmetaData() {
  return kDynMetaDataSingleton;
}

std::mutex& getGlobalDynmetaDataMutex() {
  return kDynMetaDataSingletonMutex;
}


bool gradLayerAtTop() {
  return dynamicLayerStack.back().key() == DispatchKey::Autograd;
}

optional<DynamicLayer> maybeCurrentDynamicLayer() {
  // NB: Exception for regular autograd, maybe tweak this
  if (dynamicLayerStack.size() <= 1) {
    return {};
  }
  return dynamicLayerStack.back();
}

std::vector<DynamicLayer>& getDynamicLayerStack() {
  return dynamicLayerStack;
}

int64_t pushDynamicLayer(DispatchKey key) {
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

DynamicLayer popDynamicLayer() {
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

DynamicLayer popDynamicLayerAndDeleteMetadata() {
  auto result = popDynamicLayer();
  auto level = result.layerId();

  // There is unfortunately a deadlock somewhere :/
  // std::lock_guard<std::mutex> guard(getGlobalDynmetaDataMutex());
  auto& data = getGlobalDynmetaData();
  auto it = data.find(level);
  if (it == data.end()) {
    return result;
  }
  for (auto& ptr : it->second) {
    auto val = ptr.lock();
    if (!val) continue;
    // Clear the unique_ptr inside the shared_ptr.
    (*val).reset();
  }
  // Clear the queue of weak_ptrs
  data[level].clear();

  return result;
}

static Tensor fullyMaterializeWrappers(const Tensor& tensor, std::vector<DynamicLayer>& dynlayerStack) {
  if (!tensor.defined()) {
    return tensor;
  }
  // First entry in the stack is a default autograd key
  if (dynlayerStack.size() <= 1) {
    return tensor;
  }
  // std::cout << "fullyMaterializeWrappers " << dynlayerStack.size() << std::endl;
  auto* wrapper = maybeGetTensorWrapper(tensor);
  Tensor result = tensor;
  if (!wrapper) {
    for (int64_t idx = 1; idx < dynlayerStack.size(); idx++) {
      if (dynlayerStack[idx].key() == DispatchKey::Autograd) {
        std::cout << "materializing " << dynlayerStack[idx].layerId() << std::endl;
        result = makeTensorWrapper(result, dynlayerStack[idx].layerId());
      } else {
        TORCH_INTERNAL_ASSERT(false);
      }
    }
    TORCH_INTERNAL_ASSERT(result.defined());
    dumpTensorCout(result);
    return result;
  }
  if (wrapper->level() == dynlayerStack.back().layerId()) {
    TORCH_INTERNAL_ASSERT(tensor.defined());
    return tensor;
  }
  for (int64_t idx = 1; idx < dynlayerStack.size(); idx++) {
    if (wrapper->level() >= dynlayerStack[idx].layerId()) {
      continue;
    }
    TORCH_INTERNAL_ASSERT(dynlayerStack[idx].key() == DispatchKey::Autograd);
    std::cout << "materializing " << dynlayerStack[idx].layerId() << std::endl;
    result = makeTensorWrapper(result, dynlayerStack[idx].layerId());
  }
  TORCH_INTERNAL_ASSERT(result.defined());
  dumpTensorCout(result);
  return result;
}

void foreachTensorInplace(std::vector<IValue>& args, int64_t begin, int64_t end,
    std::function<Tensor(const Tensor&)> func) {
  TORCH_INTERNAL_ASSERT(begin >= 0);
  TORCH_INTERNAL_ASSERT(end >= 0);
  TORCH_INTERNAL_ASSERT(begin <= end);
  for (int64_t idx = begin; idx < end; idx++) {
    auto ivalue = args[idx];
    if (ivalue.isTensorList()) {
      TORCH_INTERNAL_ASSERT(false, "NYI: TensorList");
    }
    if (!ivalue.isTensor()) {
      continue;
    }
    // std::cout << "replaced " << idx << std::endl;
    Tensor value = ivalue.toTensor();
    Tensor replacement = func(value);
    args[idx] = replacement; // TODO: std::move?
    // sanity checks
    if (ivalue.toTensor().defined()) {
      TORCH_INTERNAL_ASSERT(args[idx].toTensor().defined());
    }
    // if (ivalue.toTensor().data_ptr()) {
    //   TORCH_INTERNAL_ASSERT(args[idx].toTensor().data_ptr());
    // }
  }
}

void dynamicLayerFrontFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  std::cout << "DLS size: " << dynamicLayerStack.size() << std::endl;
  if (dynamicLayerStack.size() == 0) {
    // TODO: temp code
    if (stack->size() > 0) {
      auto last_ivalue = (*stack)[stack->size() - 1];
      if (last_ivalue.isTensor()) {
        auto tensor = last_ivalue.toTensor();
        auto* wrapper = maybeGetTensorWrapper(tensor);
        TORCH_INTERNAL_ASSERT(wrapper == nullptr);
        TORCH_INTERNAL_ASSERT(tensor.has_storage());
      }
    }
    // std::cout << "dynamicLayerFrontFallback " << op.operator_name() << " terminal" << std::endl;
    DispatchKeySet exclude;
    exclude = exclude.add(DispatchKey::DynamicLayerFront);
    exclude = exclude.add(DispatchKey::Batched);
    exclude = exclude.add(DispatchKey::Autograd);
    exclude = exclude.add(DispatchKey::AutogradOther);
    exclude = exclude.add(DispatchKey::AutogradCPU);
    exclude = exclude.add(DispatchKey::AutogradCUDA);
    exclude = exclude.add(DispatchKey::InplaceOrView);
    exclude = exclude.add(DispatchKey::DynamicLayerBack);

    c10::impl::ExcludeDispatchKeyGuard guard(exclude);
    op.callBoxed(stack);
    // std::cout << "dynamicLayerFrontFallback " << op.operator_name() << " end terminal" << std::endl;
    return;
  }

  auto materialize = [](const Tensor& tensor) {
    return fullyMaterializeWrappers(tensor, getDynamicLayerStack());
  };
  auto num_args = op.schema().arguments().size();
  foreachTensorInplace(*stack, stack->size() - num_args, stack->size(), materialize);

  auto layer = dynamicLayerStack.back();

  DispatchKeySet exclude = DispatchKeySet::FULL;
  exclude = exclude.remove(DispatchKey::DynamicLayerBack);
  // NB: Alias dispatch key doesn't work in exclude set :(
  if (layer.key() == DispatchKey::Autograd) {
    // std::cout << "enabling some autograd keys..." << std::endl;
    exclude = exclude.remove(DispatchKey::Autograd);
    exclude = exclude.remove(DispatchKey::AutogradOther);
    exclude = exclude.remove(DispatchKey::AutogradCPU);
    exclude = exclude.remove(DispatchKey::AutogradCUDA);
    exclude = exclude.remove(DispatchKey::InplaceOrView);
  } else {
    exclude = exclude.remove(layer.key());
  }
  c10::impl::ExcludeDispatchKeyGuard guard(exclude);
  // Exclude all keys except for layer.key and DynamicLayerBack
  // auto keyset = c10::impl::PODLocalDispatchKeySet();
  // keyset.set_excluded(exclude);
  // c10::impl::_force_tls_local_dispatch_key_set(keyset);

  // std::cout << "dynamicLayerFrontFallback " << op.operator_name() << " " << layer.key() << " " << layer.layerId() << std::endl;

  // Re-dispatch
  op.callBoxed(stack);

  // Clear TLS
  // keyset = c10::impl::PODLocalDispatchKeySet();
  // c10::impl::_force_tls_local_dispatch_key_set(keyset);
  // c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerFront, true);
  // c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerBack, true);
}

struct WithoutTop {
  WithoutTop(): layer_(popDynamicLayer()) {
    // prev_grad_enabled_ = GradMode::is_enabled();
    // if (!prev_grad_enabled_) {
    //   GradMode::set_enabled(true);
    // }
  }
  ~WithoutTop() {
    pushDynamicLayer(layer_.key()); 
    // GradMode::set_enabled(prev_grad_enabled_);
  }

  bool prev_grad_enabled_;
  DynamicLayer layer_;
};

void dynamicLayerBackFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // std::cout << "dynamicLayerBackFallback" << std::endl;
  //
  auto cur_level = getDynamicLayerStack().back().layerId();

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
    std::cout << "unwrap " << cur_level << std::endl;
    return tensor;
  };
  auto wrap = [&](const Tensor& tensor) {
    if (!tensor.defined()) {
      return tensor;
    }
    if (cur_level == 1) {
      return tensor;
    }
    std::cout << "wrap " << cur_level << std::endl;
    return makeTensorWrapper(tensor, cur_level);
  };

  // Unwrap everything
  auto args_size = op.schema().arguments().size();
  foreachTensorInplace(*stack, stack->size() - args_size, stack->size(), unwrap);

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

  // Rewrap everything
  auto ret_size = op.schema().returns().size();
  foreachTensorInplace(*stack, stack->size() - ret_size, stack->size(), wrap);
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
}

} // namespace at
