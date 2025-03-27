#include <clc/clc.h>
#include <libspirv/spirv.h>

#define IMPL(TYPE, AS, OP)                                                \
  _CLC_OVERLOAD _CLC_DEF TYPE atomic_max(volatile AS TYPE *p, TYPE val) { \
    return OP((AS TYPE *)p, Device, SequentiallyConsistent, val);         \
  }

IMPL(int, global, __spirv_AtomicSMax)
IMPL(unsigned int, global, __spirv_AtomicUMax)
IMPL(int, local, __spirv_AtomicSMax)
IMPL(unsigned int, local, __spirv_AtomicUMax)
#undef IMPL
