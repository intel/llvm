#include <clc/clc.h>
#include <libspirv/spirv.h>

#define IMPL(TYPE, TYPE_MANGLED, AS, AS_MANGLED)                                                                          \
  _CLC_OVERLOAD _CLC_DEF TYPE atomic_inc(volatile AS TYPE *p) {                                                           \
    /* TODO: Stop manually mangling this name. Need C++ namespaces to get the                                             \
     * exact mangling. */                                                                                                 \
    return _Z24__spirv_AtomicIIncrementPU3##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE( \
        p, Device, SequentiallyConsistent);                                                                               \
  }

IMPL(int, i, global, AS1)
IMPL(unsigned int, j, global, AS1)
IMPL(int, i, local, AS3)
IMPL(unsigned int, j, local, AS3)
#undef IMPL
