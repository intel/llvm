// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -fdeclare-spirv-builtins -fsyntax-only -verify %s
// expected-no-diagnostics

#define AS_GLOBAL __attribute__((opencl_global))
#define AS_LOCAL __attribute__((opencl_local))
#define AS_PRIVATE __attribute__((opencl_private))
#define AS_GENERIC __attribute__((opencl_generic))

void test_signed() {
  long long AS_GLOBAL *a = nullptr;
  long long AS_LOCAL *b = nullptr;
  long long AS_PRIVATE *c = nullptr;

  __spirv_AtomicLoad(a, 1, 16);
  __spirv_AtomicStore(b, 2, 8, 0LL);
  __spirv_AtomicExchange(c, 4, 4, 0LL);
  __spirv_AtomicCompareExchange(a, 2, 0, 0, 1LL, 0LL);
  __spirv_AtomicCompareExchangeWeak(a, 1, 16, 0, 1LL, 0LL);
  __spirv_AtomicSMin(b, 2, 8, 0LL);
  __spirv_AtomicSMax(c, 4, 4, 0LL);
  __spirv_AtomicIAdd(a, 2, 0, 0LL);
  __spirv_AtomicISub(a, 1, 16, 0LL);
  __spirv_AtomicAnd(b, 2, 8, 0LL);
  __spirv_AtomicOr(c, 4, 4, 0LL);
  __spirv_AtomicXor(a, 2, 0, 0LL);
}

void test_unsigned() {
  unsigned long long AS_GLOBAL *a = nullptr;
  unsigned long long AS_LOCAL *b = nullptr;
  unsigned long long AS_PRIVATE *c = nullptr;

  __spirv_AtomicLoad(a, 1, 16);
  __spirv_AtomicStore(b, 2, 8, 0ULL);
  __spirv_AtomicExchange(c, 4, 4, 0ULL);
  __spirv_AtomicCompareExchange(a, 2, 0, 0, 1ULL, 0ULL);
  __spirv_AtomicCompareExchangeWeak(a, 1, 16, 0, 1ULL, 0ULL);
  __spirv_AtomicUMin(b, 2, 8, 0ULL);
  __spirv_AtomicUMax(c, 4, 4, 0ULL);
  __spirv_AtomicIAdd(a, 2, 0, 0ULL);
  __spirv_AtomicISub(a, 1, 16, 0ULL);
  __spirv_AtomicAnd(b, 2, 8, 0ULL);
  __spirv_AtomicOr(c, 4, 4, 0ULL);
  __spirv_AtomicXor(a, 2, 0, 0ULL);
}