// RUN: %clangxx -fsycl-device-only -S -O3 -Xclang -emit-llvm -o - %s | FileCheck %s

// This test checks the device code for various math operations on sycl::vec.
#include <sycl/sycl.hpp>

using namespace sycl;

template <typename T>
using rel_t = std::conditional_t<
    sizeof(T) == 1, int8_t,
    std::conditional_t<
        sizeof(T) == 2, int16_t,
        std::conditional_t<sizeof(T) == 4, int32_t,
                           std::conditional_t<sizeof(T) == 8, int64_t, void>>>>;

// For testing binary operations
#define CHECKBINOP(Q, T, N, IS_RELOP, OP, SUFFIX)                              \
  {                                                                            \
    using VecT = sycl::vec<T, N>;                                              \
    using ResT = sycl::vec<std::conditional_t<IS_RELOP, rel_t<T>, T>, N>;      \
    VecT In##SUFFIX##VecA{static_cast<T>(5)};                                  \
    VecT In##SUFFIX##VecB{static_cast<T>(6)};                                  \
    {                                                                          \
      VecT OutVecsDevice[1];                                                   \
      sycl::buffer<VecT, 1> OutVecsBuff{OutVecsDevice, 1};                     \
      Q.submit([&](sycl::handler &CGH) {                                       \
        sycl::accessor OutVecsAcc{OutVecsBuff, CGH, sycl::read_write};         \
        CGH.single_task([=]() {                                                \
          auto OutVec1 = In##SUFFIX##VecA OP In##SUFFIX##VecB;                 \
          static_assert(std::is_same_v<decltype(OutVec1), ResT>);              \
          OutVecsAcc[0] = OutVec1;                                             \
        });                                                                    \
      });                                                                      \
    }                                                                          \
  }

// For testing unary operators
#define CHECKUOP(Q, T, N, IS_RELOP, OP, SUFFIX, REF)                           \
  {                                                                            \
    using VecT = sycl::vec<T, N>;                                              \
    using ResT = sycl::vec<std::conditional_t<IS_RELOP, rel_t<T>, T>, N>;      \
    VecT In##SUFFIX##VecA{static_cast<T>(REF)};                                \
    {                                                                          \
      VecT OutVecsDevice[1];                                                   \
      sycl::buffer<VecT, 1> OutVecsBuff{OutVecsDevice, 1};                     \
      Q.submit([&](sycl::handler &CGH) {                                       \
        sycl::accessor OutVecsAcc{OutVecsBuff, CGH, sycl::read_write};         \
        CGH.single_task([=]() {                                                \
          auto OutVec1 = OP In##SUFFIX##VecA;                                  \
          static_assert(std::is_same_v<decltype(OutVec1), ResT>);              \
          OutVecsAcc[0] = OutVec1;                                             \
        });                                                                    \
      });                                                                      \
    }                                                                          \
  }

int main() {

  queue Q;

  // Check device code for binary arithmetic operations.
  // int, char, and float are handled in the same manner.
  {
    // CHECK: %add{{.*}} = add <2 x i32> %{{.*}}
    CHECKBINOP(Q, int, 2, false, +, INT)

    // CHECK: %add{{.*}} = add <2 x i8> %{{.*}}
    CHECKBINOP(Q, std::byte, 2, false, +, BYTE)

    // CHECK: %add{{.*}} = add <2 x i8> %{{.*}}
    // CHECK: for.body{{.*}}
    // CHECK: {{.*}} = icmp ne i8 %{{.*}}, 0
    CHECKBINOP(Q, bool, 2, false, +, BOOL)

    // CHECK: %add{{.*}} = fadd <2 x half> %{{.*}}
    CHECKBINOP(Q, sycl::half, 2, false, +, HALF)

    // CHECK: for.body{{.*}}
    // CHECK: {{.*}}ConvertBF16ToFINTEL{{.*}}
    // CHECK: {{.*}}ConvertBF16ToFINTEL{{.*}}
    // CHECK: %add{{.*}} = fadd float %{{.*}}, %{{.*}}
    // CHECK: {{.*}}ConvertFToBF16INTEL{{.*}}
    CHECKBINOP(Q, ext::oneapi::bfloat16, 2, false, +, BF)
  }

  // Check device code for binary logical operations.
  {
    // CHECK: icmp sgt <2 x i32> %{{.*}}
    // CHECK: sext <2 x i1> %{{.*}} to <2 x i32>
    CHECKBINOP(Q, int, 2, true, >, INT)

    // TODO: std::byte and bool implementation for logical ops
    // can be optimized. For some reason, we have an
    // extra vector "%ref.tmp.i = alloca {{.*}}:vec.3"
    // that serves no purpose.
    // CHECK: {{.*}} icmp sgt <2 x i8> %{{.*}}
    // CHECK: {{.*}} sext <2 x i1> %{{.*}} to <2 x i8>
    CHECKBINOP(Q, std::byte, 2, true, >, BYTE)

    // CHECK: {{.*}} icmp sgt <2 x i8> %{{.*}}
    // CHECK: {{.*}} sext <2 x i1> %{{.*}} to <2 x i8>
    CHECKBINOP(Q, bool, 2, true, >, BOOL)

    // CHECK: {{.*}} fcmp ogt <2 x half> {{.*}}
    // CHECK: {{.*}} sext <2 x i1> {{.*}} to <2 x i16>
    CHECKBINOP(Q, sycl::half, 2, true, >, HALF)

    // FIXME: Why do we treat BF16 as i16 when doing logical ops
    // but convert to float for arithmetic ops?
    // CHECK: {{.*}} load <2 x i16>, ptr %_arg_InBFVecA {{.*}}
    // CHECK: {{.*}} icmp ugt <2 x i16> {{.*}}
    // CHECK: {{.*}} sext <2 x i1> {{.*}} to <2 x i16>
    CHECKBINOP(Q, ext::oneapi::bfloat16, 2, true, >, BF)
  }

  // Check device code for unary operators
  {
    // CHECK: {{.*}} icmp eq <2 x i32> %{{.*}}, zeroinitializer
    // CHECK: {{.*}} sext <2 x i1> %{{.*}} to <2 x i32>
    CHECKUOP(Q, int, 2, true, !, INTUOP, 1)

    // CHECK: {{.*}} sub <2 x i32> zeroinitializer, %{{.*}}
    CHECKUOP(Q, int, 2, false, -, INTUOP, 1)

    // CHECK: %{{.*}} = icmp eq <2 x i8> %{{.*}}, zeroinitializer
    // CHECK: %{{.*}} = sext <2 x i1> %{{.*}} to <2 x i8>
    CHECKUOP(Q, std::byte, 2, true, !, BYTEUOP, 1)

    // FIXME: Why is this getting optimized out?
    //CHECKUOP(Q, std::byte, 2, false, +, BYTEUOP, -1)

    // CHECK: {{.*}} icmp eq <2 x i8> {{.*}}, zeroinitializer
    // CHECK: %{{.*}} sext <2 x i1> %{{.*}} to <2 x i8>
    CHECKUOP(Q, bool, 2, true, !, BOOLUOP, 1)

    // CHECK: {{.*}} sub <2 x i8> zeroinitializer, %{{.*}}
    CHECKUOP(Q, bool, 2, false, -, BOOLUOP, 1)

    // CHECK: {{.*}} fcmp oeq <2 x half> %{{.*}}, zeroinitializer
    // CHECK: {{.*}} sext <2 x i1> %{{.*}} to <2 x i16>
    CHECKUOP(Q, sycl::half, 2, true, !, HALFUOP, 1)

    // CHECK: {{.*}} fneg <2 x half> %{{.*}}
    CHECKUOP(Q, sycl::half, 2, false, -, HALFUOP, 1)

    // CHECK: for.cond{{.*}}
    // CHECK: {{.*}} fcmp oeq float %{{.*}}, 0.000000e+00
    // CHECK: {{.*}} uitofp i1 %{{.*}} to float
    CHECKUOP(Q, ext::oneapi::bfloat16, 2, true, !, BFUOP, 1)

    // CHECK: for.cond{{.*}}
    // CHECK: {{.*}} fneg float %{{.*}}
    CHECKUOP(Q, ext::oneapi::bfloat16, 2, false, -, BFUOP, 1)
  }

  return 0;
};
