#pragma once

#include <atomic>

namespace ur::offload {
struct handle_base {};
} // namespace ur::offload

struct RefCounted : ur::offload::handle_base {
  std::atomic_uint32_t RefCount = 1;
};
