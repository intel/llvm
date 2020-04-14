#include <clc/clc.h>
#include <spirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF float atomic_xchg(volatile global float *p, float val) {
  /* TODO: Stop manually mangling this name. Need C++ namespaces to get the exact mangling. */
  return _Z22__spirv_AtomicExchangePU3AS1fN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEf(
      p, Device, SequentiallyConsistent, val);
}

_CLC_OVERLOAD _CLC_DEF float atomic_xchg(volatile local float *p, float val) {
  /* TODO: Stop manually mangling this name. Need C++ namespaces to get the exact mangling. */
  return _Z22__spirv_AtomicExchangePU3AS3fN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEf(
      p, Device, SequentiallyConsistent, val);
}

#define IMPL(TYPE, TYPE_MANGLED, AS, AS_MANGLED)                                                                                      \
  _CLC_OVERLOAD _CLC_DEF TYPE atomic_xchg(volatile AS TYPE *p, TYPE val) {                                                            \
    /* TODO: Stop manually mangling this name. Need C++ namespaces to get the                                                         \
     * exact mangling. */                                                                                                             \
    return _Z22__spirv_AtomicExchangePU3##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
        p, Device, SequentiallyConsistent, val);                                                                                      \
  }

IMPL(int, i, global, AS1)
IMPL(unsigned int, j, global, AS1)
IMPL(int, i, local, AS3)
IMPL(unsigned int, j, local, AS3)
#undef IMPL
