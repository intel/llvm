#include <clc/clc.h>

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT void
__spirv_ControlBarrier(unsigned int scope, unsigned int memory,
                       unsigned int semantics);

_CLC_DEF _CLC_OVERLOAD void barrier(cl_mem_fence_flags flags) {
  // Call spir-v implementation of the barrier.
  // Set semantics to not None, so it performs mem fence.
  __spirv_ControlBarrier(0, flags, 1);
}
