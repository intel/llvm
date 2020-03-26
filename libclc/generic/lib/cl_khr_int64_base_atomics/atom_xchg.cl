#include <clc/clc.h>
#include <spirv/spirv.h>

#ifdef cl_khr_int64_base_atomics

#define IMPL(TYPE, TYPE_MANGLED, AS, AS_MANGLED)                                                                                      \
  _CLC_OVERLOAD _CLC_DEF TYPE atom_xchg(volatile AS TYPE *p, TYPE val) {                                                              \
    return _Z22__spirv_AtomicExchangePU3##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
        p, Device, SequentiallyConsistent, val);                                                                                      \
  }

IMPL(long, l, global, AS1)
IMPL(unsigned long, m, global, AS1)
IMPL(long, l, local, AS3)
IMPL(unsigned long, m, local, AS3)
#undef IMPL

#endif
