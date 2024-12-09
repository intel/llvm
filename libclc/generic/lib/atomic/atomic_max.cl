#include <clc/clc.h>
#include <libspirv/spirv.h>

#define IMPL(TYPE, TYPE_MANGLED, AS, AS_MANGLED, OP)                                                                  \
  _CLC_OVERLOAD _CLC_DEF TYPE atomic_max(volatile AS TYPE *p, TYPE val) {                                             \
    /* TODO: Stop manually mangling this name. Need C++ namespaces to get the                                         \
     * exact mangling. */                                                                                             \
    return _Z18##OP##PU3##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
        p, Device, SequentiallyConsistent, val);                                                                      \
  }

IMPL(int, i, global, AS1, __spirv_AtomicSMax)
IMPL(unsigned int, j, global, AS1, __spirv_AtomicUMax)
IMPL(int, i, local, AS3, __spirv_AtomicSMax)
IMPL(unsigned int, j, local, AS3, __spirv_AtomicUMax)
#undef IMPL
