#include <clc/clc.h>
#include <libspirv/spirv.h>

#define IMPL(TYPE, AS)                                          \
  _CLC_OVERLOAD _CLC_DEF TYPE atomic_dec(volatile AS TYPE *p) { \
    return __spirv_AtomicIDecrement(                            \
        (AS TYPE *)p, Device, SequentiallyConsistent);          \
  }

IMPL(int, global)
IMPL(unsigned int, global)
IMPL(int, local)
IMPL(unsigned int, local)
#undef IMPL
