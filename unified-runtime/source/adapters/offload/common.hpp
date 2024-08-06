#pragma once

#include <atomic>

struct RefCounted {
  std::atomic_uint32_t RefCount = 1;
};
