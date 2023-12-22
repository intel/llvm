#include "common.hpp"

extern "C" __attribute__((weak)) void __asan_init(void);

namespace ur_sanitizer_layer {

bool IsInASanContext() { return __asan_init != nullptr; }

} // namespace ur_sanitizer_layer