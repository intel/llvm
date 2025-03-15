#include <clc/clc.h>
#include <libspirv/spirv.h>

#ifdef cl_khr_int64_extended_atomics

#define IMPL(TYPE, AS, NAME)                                            \
  _CLC_OVERLOAD _CLC_DEF TYPE atom_min(volatile AS TYPE *p, TYPE val) { \
    return NAME((AS TYPE *)p, Device, SequentiallyConsistent, val);     \
  }

IMPL(long, global, __spirv_AtomicSMin)
IMPL(unsigned long, global, __spirv_AtomicUMin)
IMPL(long, local, __spirv_AtomicSMin)
IMPL(unsigned long, local, __spirv_AtomicUMin)
#undef IMPL

#endif
