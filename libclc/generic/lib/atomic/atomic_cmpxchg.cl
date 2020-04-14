#include <clc/clc.h>
#include <spirv/spirv.h>

#define IMPL(TYPE, TYPE_MANGLED, AS, AS_MANGLED)                                                                                                              \
  _CLC_OVERLOAD _CLC_DEF TYPE atomic_cmpxchg(volatile AS TYPE *p, TYPE cmp,                                                                                   \
                                             TYPE val) {                                                                                                      \
    /* TODO: Stop manually mangling this name. Need C++ namespaces to get the                                                                                 \
     * exact mangling. */                                                                                                                                     \
    return _Z29__spirv_AtomicCompareExchangePU3##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_##TYPE_MANGLED##TYPE_MANGLED( \
        p, Device, SequentiallyConsistent, SequentiallyConsistent, val, cmp);                                                                                 \
  }

IMPL(int, i, global, AS1)
IMPL(unsigned int, j, global, AS1)
IMPL(int, i, local, AS3)
IMPL(unsigned int, j, local, AS3)
#undef IMPL
