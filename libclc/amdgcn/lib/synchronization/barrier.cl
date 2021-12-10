#include <clc/clc.h>

_CLC_DEF _CLC_OVERLOAD void barrier(cl_mem_fence_flags flags) {
  // Call spir-v implementation of the barrier.
  // Set semantics to not None, so it performs mem fence.
  __spirv_ControlBarrier(0, flags, 1);
}
