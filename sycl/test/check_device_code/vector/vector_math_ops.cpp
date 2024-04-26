// RUN: %clangxx -I %sycl_include -S -emit-llvm -fno-sycl-instrument-device-code -fsycl-device-only %s -o - | FileCheck %s

// This test checks the device code for various math operations on sycl::vec.
#include <sycl/sycl.hpp>

using namespace sycl;

// For testing binary operations.
#define CHECKBINOP(T, SUFFIX, OP)                                              \
  SYCL_EXTERNAL auto CheckDevCodeBINOP##SUFFIX(T Ref1 = (T)5, T Ref2 = (T)6) { \
    vec<T, 2> InVec##SUFFIX##2A {Ref1, Ref2};                                  \
    vec<T, 2> InVec##SUFFIX##2B {Ref2, Ref1};                                  \
    return InVec##SUFFIX##2A OP InVec##SUFFIX##2B;                             \
  }

// For testing unary operators.
#define CHECKUOP(T, SUFFIX, OP, REF)                                           \
  SYCL_EXTERNAL auto CheckDevCodeUOP##SUFFIX(int Ref1 = REF) {                 \
    vec<T, 2> InVec##SUFFIX##2 {static_cast<T>(Ref1),                          \
                                static_cast<T>(Ref1 + 1)};                     \
    return OP InVec##SUFFIX##2;                                                \
  }

/********************** Binary Ops **********************/

// CHECK: {{.*}} add <2 x i32> %{{.*}}
CHECKBINOP(int, INTA, +)

// CHECK: {{.*}} add <2 x i8> %{{.*}}
CHECKBINOP(std::byte, BYTEA, +)

// CHECK: {{.*}} add nuw nsw <2 x i8> {{.*}}
// CHECK: for.body{{.*}}
// CHECK: {{.*}} = icmp ne i8 %{{.*}}, 0
CHECKBINOP(bool, BOOLA, +)

// CHECK: {{.*}} fadd <2 x half> %{{.*}}
CHECKBINOP(sycl::half, HALFA, +)

// CHECK: for.body{{.*}}
// CHECK: {{.*}}ConvertBF16ToFINTEL{{.*}}
// CHECK: {{.*}}ConvertBF16ToFINTEL{{.*}}
// CHECK: %add{{.*}} = fadd float %{{.*}}, %{{.*}}
// CHECK: {{.*}}ConvertFToBF16INTEL{{.*}}
CHECKBINOP(ext::oneapi::bfloat16, BFA, +)

// CHECK: icmp sgt <2 x i32> %{{.*}}
// CHECK: sext <2 x i1> %{{.*}} to <2 x i32>
CHECKBINOP(int, INTL, >)

// CHECK: {{.*}} icmp sgt <2 x i8> %{{.*}}
// CHECK: {{.*}} sext <2 x i1> %{{.*}} to <2 x i8>
CHECKBINOP(std::byte, BYTEL, >)

// CHECK: {{.*}} icmp ugt <2 x i8> %{{.*}}
// CHECK: {{.*}} sext <2 x i1> %{{.*}} to <2 x i8>
CHECKBINOP(bool, BOOLL, >)

// CHECK: {{.*}} fcmp ogt <2 x half> {{.*}}
// CHECK: {{.*}} sext <2 x i1> {{.*}} to <2 x i16>
CHECKBINOP(sycl::half, HALFL, >)

// FIXME: Why do we treat BF16 as i16 when doing logical ops
// but convert to float for arithmetic ops?
// CHECK: {{.*}} icmp ugt <2 x i16> {{.*}}
// CHECK: {{.*}} sext <2 x i1> {{.*}} to <2 x i16>
CHECKBINOP(ext::oneapi::bfloat16, BFL, >)

/********************** Unary Ops **********************/

// CHECK: {{.*}} icmp eq <2 x i32> %{{.*}}, zeroinitializer
// CHECK: {{.*}} sext <2 x i1> %{{.*}} to <2 x i32>
CHECKUOP(int, INTNEG, !, 1)

// CHECK: {{.*}} sub <2 x i32> zeroinitializer, %{{.*}}
CHECKUOP(int, INTSUB, -, 1)

// CHECK: %{{.*}} = icmp eq <2 x i8> %{{.*}}, zeroinitializer
// CHECK: %{{.*}} = sext <2 x i1> %{{.*}} to <2 x i8>
CHECKUOP(std::byte, BYTENEG, !, 1)

// CHECK: {{.*}} sub <2 x i8> zeroinitializer, %{{.*}}
CHECKUOP(std::byte, BYTESUB, -, -1)

// CHECK: {{.*}} icmp eq <2 x i8> {{.*}}, zeroinitializer
// CHECK: %{{.*}} sext <2 x i1> %{{.*}} to <2 x i8>
CHECKUOP(bool, BOOLNEG, !, 1)

// CHECK: {{.*}} fcmp oeq <2 x half> %{{.*}}, zeroinitializer
// CHECK: {{.*}} sext <2 x i1> %{{.*}} to <2 x i16>
CHECKUOP(sycl::half, HALFNEG, !, 1)

// CHECK: {{.*}} fneg <2 x half> %{{.*}}
CHECKUOP(sycl::half, HALFSUB, -, 1)

// CHECK: for.cond{{.*}}
// CHECK: {{.*}} fcmp oeq float %{{.*}}, 0.000000e+00
// CHECK: {{.*}} uitofp i1 %{{.*}} to float
CHECKUOP(ext::oneapi::bfloat16, BFNEG, !, 1)

// CHECK: for{{.*}}
// CHECK: {{.*}} fneg float %{{.*}}
CHECKUOP(ext::oneapi::bfloat16, BFSUB, -, 1)
