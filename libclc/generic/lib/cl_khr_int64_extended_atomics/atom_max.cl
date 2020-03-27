#include <clc/clc.h>
#include <spirv/spirv.h>

#ifdef cl_khr_int64_extended_atomics

#define IMPL(TYPE, TYPE_MANGLED, AS, AS_MANGLED, NAME)                                                                  \
  _CLC_OVERLOAD _CLC_DEF TYPE atom_max(volatile AS TYPE *p, TYPE val) {                                                 \
    return _Z18##NAME##PU3##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
        p, Device, SequentiallyConsistent, val);                                                                        \
  }

IMPL(long, l, global, AS1, __spirv_AtomicSMax)
IMPL(unsigned long, m, global, AS1, __spirv_AtomicUMax)
IMPL(long, l, local, AS3, __spirv_AtomicSMax)
IMPL(unsigned long, m, local, AS3, __spirv_AtomicUMax)
#undef IMPL

#endif
