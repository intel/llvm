#include <clc/clc.h>
#include <libspirv/spirv.h>

#ifdef cl_khr_int64_extended_atomics

#define IMPL(TYPE, AS)                                                          \
  _CLC_OVERLOAD _CLC_DEF TYPE atom_or(volatile AS TYPE *p, TYPE val) {          \
    return __spirv_AtomicOr((AS TYPE *)p, Device, SequentiallyConsistent, val); \
  }

IMPL(long, global)
IMPL(unsigned long, global)
IMPL(long, local)
IMPL(unsigned long, local)
#undef IMPL

#endif
