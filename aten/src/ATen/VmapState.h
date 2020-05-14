#include <ATen/BatchedTensorImpl.h>

namespace at {

using VmapLevel = int64_t;

struct VmapState {
  VmapLevel addLevel(int64_t batch_size) {
    cur_level_++;
    stack_.push_back({cur_level_, batch_size});
    valid_level_refs_.push_back(std::make_shared<int64_t>(cur_level_));
    return cur_level_;
  }

  std::pair<VmapLevel,int64_t> popLevel() {
    TORCH_INTERNAL_ASSERT(cur_level_ > 0);
    auto result = stack_[stack_.size() - 1];
    stack_.pop_back();
    valid_level_refs_.pop_back();
    cur_level_--;
    return result;
  }

  std::weak_ptr<int64_t> getLevelRef() {
    TORCH_INTERNAL_ASSERT(cur_level_ > 0);
    return valid_level_refs_.back();
  }

  const std::vector<std::pair<VmapLevel,int64_t>>& stack() {
    return stack_;
  }

 private:
  VmapLevel cur_level_;
  std::vector<std::pair<VmapLevel,int64_t>> stack_;
  std::vector<std::shared_ptr<int64_t>> valid_level_refs_;
};

VmapState* getVmapState();

TORCH_API VmapLevel enterVmapLevel(int64_t batch_size);
TORCH_API int64_t exitVmapLevel();

} // namespace at
