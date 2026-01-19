// RUN: %clang_cc1 -triple spir64 -fdeclare-spirv-builtins -emit-llvm %s -o - | FileCheck %s

#define AS_GLOBAL __attribute__((opencl_global))
#define AS_LOCAL __attribute__((opencl_local))
#define AS_PRIVATE __attribute__((opencl_private))
#define AS_GENERIC __attribute__((opencl_generic))

void test_flag(int AS_GLOBAL *a, int AS_LOCAL *b, int AS_PRIVATE *c,
               int AS_GENERIC *d) {
  __spirv_AtomicFlagTestAndSet(a, 1, 16);
  __spirv_AtomicFlagTestAndSet(b, 2, 8);
  __spirv_AtomicFlagTestAndSet(c, 4, 4);
  __spirv_AtomicFlagTestAndSet(d, 2, 0);

  __spirv_AtomicFlagClear(a, 1, 16);
  __spirv_AtomicFlagClear(b, 2, 4);
  __spirv_AtomicFlagClear(c, 4, 0);
  __spirv_AtomicFlagClear(d, 2, 0);
}

template <class T>
void test_signed(T AS_GLOBAL *a, T AS_LOCAL *b, T AS_PRIVATE *c,
                 T AS_GENERIC *d) {
  __spirv_AtomicLoad(a, 1, 16);
  __spirv_AtomicLoad(b, 2, 8);
  __spirv_AtomicLoad(c, 4, 4);
  __spirv_AtomicLoad(d, 2, 0);

  __spirv_AtomicStore(a, 1, 16, (T)0);
  __spirv_AtomicStore(b, 2, 8, (T)0);
  __spirv_AtomicStore(c, 4, 4, (T)0);
  __spirv_AtomicStore(d, 2, 0, (T)0);

  __spirv_AtomicExchange(a, 1, 16, (T)0);
  __spirv_AtomicExchange(b, 2, 8, (T)0);
  __spirv_AtomicExchange(c, 4, 4, (T)0);
  __spirv_AtomicExchange(d, 2, 0, (T)0);

  __spirv_AtomicCompareExchange(a, 1, 16, 0, (T)1, (T)0);
  __spirv_AtomicCompareExchange(b, 2, 8, 0, (T)1, (T)0);
  __spirv_AtomicCompareExchange(c, 4, 4, 0, (T)1, (T)0);
  __spirv_AtomicCompareExchange(d, 2, 0, 0, (T)1, (T)0);

  __spirv_AtomicCompareExchangeWeak(a, 1, 16, 0, (T)1, (T)0);
  __spirv_AtomicCompareExchangeWeak(b, 2, 8, 0, (T)1, (T)0);
  __spirv_AtomicCompareExchangeWeak(c, 4, 4, 0, (T)1, (T)0);
  __spirv_AtomicCompareExchangeWeak(d, 2, 0, 0, (T)1, (T)0);

  __spirv_AtomicIIncrement(a, 1, 16);
  __spirv_AtomicIIncrement(b, 2, 8);
  __spirv_AtomicIIncrement(c, 4, 4);
  __spirv_AtomicIIncrement(d, 2, 0);

  __spirv_AtomicIDecrement(a, 1, 16);
  __spirv_AtomicIDecrement(b, 2, 8);
  __spirv_AtomicIDecrement(c, 4, 4);
  __spirv_AtomicIDecrement(d, 2, 0);

  __spirv_AtomicSMin(a, 1, 16, (T)0);
  __spirv_AtomicSMin(b, 2, 8, (T)0);
  __spirv_AtomicSMin(c, 4, 4, (T)0);
  __spirv_AtomicSMin(d, 2, 0, (T)0);

  __spirv_AtomicSMax(a, 1, 16, (T)0);
  __spirv_AtomicSMax(b, 2, 8, (T)0);
  __spirv_AtomicSMax(c, 4, 4, (T)0);
  __spirv_AtomicSMax(d, 2, 0, (T)0);

  __spirv_AtomicIAdd(a, 1, 16, (T)0);
  __spirv_AtomicIAdd(b, 2, 8, (T)0);
  __spirv_AtomicIAdd(c, 4, 4, (T)0);
  __spirv_AtomicIAdd(d, 2, 0, (T)0);

  __spirv_AtomicISub(a, 1, 16, (T)0);
  __spirv_AtomicISub(b, 2, 8, (T)0);
  __spirv_AtomicISub(c, 4, 4, (T)0);
  __spirv_AtomicISub(d, 2, 0, (T)0);

  __spirv_AtomicAnd(a, 1, 16, (T)0);
  __spirv_AtomicAnd(b, 2, 8, (T)0);
  __spirv_AtomicAnd(c, 4, 4, (T)0);
  __spirv_AtomicAnd(d, 2, 0, (T)0);

  __spirv_AtomicOr(a, 1, 16, (T)0);
  __spirv_AtomicOr(b, 2, 8, (T)0);
  __spirv_AtomicOr(c, 4, 4, (T)0);
  __spirv_AtomicOr(d, 2, 0, (T)0);

  __spirv_AtomicXor(a, 1, 16, (T)0);
  __spirv_AtomicXor(b, 2, 8, (T)0);
  __spirv_AtomicXor(c, 4, 4, (T)0);
  __spirv_AtomicXor(d, 2, 0, (T)0);
}

template <class T>
void test_unsigned(T AS_GLOBAL *a, T AS_LOCAL *b, T AS_PRIVATE *c,
                   T AS_GENERIC *d) {

  __spirv_AtomicUMin(a, 1, 16, (T)0);
  __spirv_AtomicUMin(b, 2, 8, (T)0);
  __spirv_AtomicUMin(c, 4, 4, (T)0);
  __spirv_AtomicUMin(d, 2, 0, (T)0);

  __spirv_AtomicUMax(a, 1, 16, (T)0);
  __spirv_AtomicUMax(b, 2, 8, (T)0);
  __spirv_AtomicUMax(c, 4, 4, (T)0);
  __spirv_AtomicUMax(d, 2, 0, (T)0);
}

template <class T>
void test_float(T AS_GLOBAL *a, T AS_LOCAL *b, T AS_PRIVATE *c,
                T AS_GENERIC *d) {
  __spirv_AtomicFMaxEXT(a, 1, 16, (T)0);
  __spirv_AtomicFMaxEXT(b, 2, 8, (T)0);
  __spirv_AtomicFMaxEXT(c, 4, 4, (T)0);
  __spirv_AtomicFMaxEXT(d, 2, 0, (T)0);

  __spirv_AtomicFMinEXT(a, 1, 16, (T)0);
  __spirv_AtomicFMinEXT(b, 2, 8, (T)0);
  __spirv_AtomicFMinEXT(c, 4, 4, (T)0);
  __spirv_AtomicFMinEXT(d, 2, 0, (T)0);

  __spirv_AtomicFAddEXT(a, 1, 16, (T)0);
  __spirv_AtomicFAddEXT(b, 2, 8, (T)0);
  __spirv_AtomicFAddEXT(c, 4, 4, (T)0);
  __spirv_AtomicFAddEXT(d, 2, 0, (T)0);
}

void foo() {
  int AS_GLOBAL *a;
  int AS_LOCAL *b;
  int AS_PRIVATE *c;
  int AS_GENERIC *d;
  test_flag(a, b, c, d);

  test_signed<int>(a, b, c, d);

  unsigned int AS_GLOBAL *ua;
  unsigned int AS_LOCAL *ub;
  unsigned int AS_PRIVATE *uc;
  unsigned int AS_GENERIC *ud;
  test_unsigned<unsigned int>(ua, ub, uc, ud);

  float AS_GLOBAL *fa;
  float AS_LOCAL *fb;
  float AS_PRIVATE *fc;
  float AS_GENERIC *fd;
  test_float<float>(fa, fb, fc, fd);
}

// CHECK: call spir_func noundef zeroext i1 @_Z28__spirv_AtomicFlagTestAndSetPU3AS1iii(
// CHECK: call spir_func noundef zeroext i1 @_Z28__spirv_AtomicFlagTestAndSetPU3AS3iii(
// CHECK: call spir_func noundef zeroext i1 @_Z28__spirv_AtomicFlagTestAndSetPiii(
// CHECK: call spir_func noundef zeroext i1 @_Z28__spirv_AtomicFlagTestAndSetPU3AS4iii(
// CHECK: call spir_func void @_Z23__spirv_AtomicFlagClearPU3AS1iii(
// CHECK: call spir_func void @_Z23__spirv_AtomicFlagClearPU3AS3iii(
// CHECK: call spir_func void @_Z23__spirv_AtomicFlagClearPiii(
// CHECK: call spir_func void @_Z23__spirv_AtomicFlagClearPU3AS4iii(

// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicLoadPU3AS1iii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicLoadPU3AS3iii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicLoadPiii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicLoadPU3AS4iii(
// CHECK: call spir_func void @_Z19__spirv_AtomicStorePU3AS1iiii(
// CHECK: call spir_func void @_Z19__spirv_AtomicStorePU3AS3iiii(
// CHECK: call spir_func void @_Z19__spirv_AtomicStorePiiii(
// CHECK: call spir_func void @_Z19__spirv_AtomicStorePU3AS4iiii(
// CHECK: call spir_func noundef i32 @_Z22__spirv_AtomicExchangePU3AS1iiii(
// CHECK: call spir_func noundef i32 @_Z22__spirv_AtomicExchangePU3AS3iiii(
// CHECK: call spir_func noundef i32 @_Z22__spirv_AtomicExchangePiiii(
// CHECK: call spir_func noundef i32 @_Z22__spirv_AtomicExchangePU3AS4iiii(
// CHECK: call spir_func noundef i32 @_Z29__spirv_AtomicCompareExchangePU3AS1iiiiii(
// CHECK: call spir_func noundef i32 @_Z29__spirv_AtomicCompareExchangePU3AS3iiiiii(
// CHECK: call spir_func noundef i32 @_Z29__spirv_AtomicCompareExchangePiiiiii(
// CHECK: call spir_func noundef i32 @_Z29__spirv_AtomicCompareExchangePU3AS4iiiiii(
// CHECK: call spir_func noundef i32 @_Z33__spirv_AtomicCompareExchangeWeakPU3AS1iiiiii(
// CHECK: call spir_func noundef i32 @_Z33__spirv_AtomicCompareExchangeWeakPU3AS3iiiiii(
// CHECK: call spir_func noundef i32 @_Z33__spirv_AtomicCompareExchangeWeakPiiiiii(
// CHECK: call spir_func noundef i32 @_Z33__spirv_AtomicCompareExchangeWeakPU3AS4iiiiii(
// CHECK: call spir_func noundef i32 @_Z24__spirv_AtomicIIncrementPU3AS1iii(
// CHECK: call spir_func noundef i32 @_Z24__spirv_AtomicIIncrementPU3AS3iii(
// CHECK: call spir_func noundef i32 @_Z24__spirv_AtomicIIncrementPiii(
// CHECK: call spir_func noundef i32 @_Z24__spirv_AtomicIIncrementPU3AS4iii(
// CHECK: call spir_func noundef i32 @_Z24__spirv_AtomicIDecrementPU3AS1iii(
// CHECK: call spir_func noundef i32 @_Z24__spirv_AtomicIDecrementPU3AS3iii(
// CHECK: call spir_func noundef i32 @_Z24__spirv_AtomicIDecrementPiii(
// CHECK: call spir_func noundef i32 @_Z24__spirv_AtomicIDecrementPU3AS4iii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicSMinPU3AS1iiii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicSMinPU3AS3iiii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicSMinPiiii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicSMinPU3AS4iiii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicSMaxPU3AS1iiii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicSMaxPU3AS3iiii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicSMaxPiiii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicSMaxPU3AS4iiii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicIAddPU3AS1iiii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicIAddPU3AS3iiii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicIAddPiiii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicIAddPU3AS4iiii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicISubPU3AS1iiii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicISubPU3AS3iiii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicISubPiiii(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicISubPU3AS4iiii(
// CHECK: call spir_func noundef i32 @_Z17__spirv_AtomicAndPU3AS1iiii(
// CHECK: call spir_func noundef i32 @_Z17__spirv_AtomicAndPU3AS3iiii(
// CHECK: call spir_func noundef i32 @_Z17__spirv_AtomicAndPiiii(
// CHECK: call spir_func noundef i32 @_Z17__spirv_AtomicAndPU3AS4iiii(
// CHECK: call spir_func noundef i32 @_Z16__spirv_AtomicOrPU3AS1iiii(
// CHECK: call spir_func noundef i32 @_Z16__spirv_AtomicOrPU3AS3iiii(
// CHECK: call spir_func noundef i32 @_Z16__spirv_AtomicOrPiiii(
// CHECK: call spir_func noundef i32 @_Z16__spirv_AtomicOrPU3AS4iiii(
// CHECK: call spir_func noundef i32 @_Z17__spirv_AtomicXorPU3AS1iiii(
// CHECK: call spir_func noundef i32 @_Z17__spirv_AtomicXorPU3AS3iiii(
// CHECK: call spir_func noundef i32 @_Z17__spirv_AtomicXorPiiii(
// CHECK: call spir_func noundef i32 @_Z17__spirv_AtomicXorPU3AS4iiii(

// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicUMinPU3AS1jiij(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicUMinPU3AS3jiij(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicUMinPjiij(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicUMinPU3AS4jiij(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicUMaxPU3AS1jiij(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicUMaxPU3AS3jiij(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicUMaxPjiij(
// CHECK: call spir_func noundef i32 @_Z18__spirv_AtomicUMaxPU3AS4jiij(

// CHECK: call spir_func noundef float @_Z21__spirv_AtomicFMaxEXTPU3AS1fiif(
// CHECK: call spir_func noundef float @_Z21__spirv_AtomicFMaxEXTPU3AS3fiif(
// CHECK: call spir_func noundef float @_Z21__spirv_AtomicFMaxEXTPfiif(
// CHECK: call spir_func noundef float @_Z21__spirv_AtomicFMaxEXTPU3AS4fiif(
// CHECK: call spir_func noundef float @_Z21__spirv_AtomicFMinEXTPU3AS1fiif(
// CHECK: call spir_func noundef float @_Z21__spirv_AtomicFMinEXTPU3AS3fiif(
// CHECK: call spir_func noundef float @_Z21__spirv_AtomicFMinEXTPfiif(
// CHECK: call spir_func noundef float @_Z21__spirv_AtomicFMinEXTPU3AS4fiif(
// CHECK: call spir_func noundef float @_Z21__spirv_AtomicFAddEXTPU3AS1fiif(
// CHECK: call spir_func noundef float @_Z21__spirv_AtomicFAddEXTPU3AS3fiif(
// CHECK: call spir_func noundef float @_Z21__spirv_AtomicFAddEXTPfiif(
// CHECK: call spir_func noundef float @_Z21__spirv_AtomicFAddEXTPU3AS4fiif(
