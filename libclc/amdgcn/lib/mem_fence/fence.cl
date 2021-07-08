#include <clc/clc.h>

_CLC_OVERLOAD _CLC_DEF void __spirv_MemoryBarrier(unsigned int memory,
                                                  unsigned int semantics);

// We don't have separate mechanism for read and write fences
_CLC_DEF _CLC_OVERLOAD void read_mem_fence(cl_mem_fence_flags flags) {
  __spirv_MemoryBarrier(flags, 1);
}

_CLC_DEF _CLC_OVERLOAD void write_mem_fence(cl_mem_fence_flags flags) {
  __spirv_MemoryBarrier(flags, 1);
}
