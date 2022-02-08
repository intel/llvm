//==---------------- ext_math.cpp  - DPC++ ESIMD extended math test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to unimplemented 'half' type
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test checks extended math operations. Combinations of
// - argument type - half, float
// - math function - sin, cos, ..., div_ieee, pow
// - SYCL vs ESIMD APIs

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/builtins_esimd.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <cmath>
#include <iostream>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental;
using namespace sycl::ext::intel::experimental::esimd;

// --- Data initialization functions

// Initialization data for trigonometric functions' input.
// H/w supports only limited range of sin/cos arguments with decent accuracy:
// absolute error <= 0.0008 for the range of +/- 32767*pi (+/- 102941).

constexpr int accuracy_limit = 32767 * 3.14 - 1;

template <class T> struct InitDataFuncTrig {
  void operator()(T *In, T *Out, size_t Size) const {
    for (auto I = 0; I < Size; ++I) {
      In[I] = (I + 1) % accuracy_limit;
      Out[I] = (T)0;
    }
  }
};

template <class T> struct InitDataFuncWide {
  void operator()(T *In, T *Out, size_t Size) const {
    for (auto I = 0; I < Size; ++I) {
      In[I] = I + 1.0;
      Out[I] = (T)0;
    }
  }
};

template <class T> struct InitDataFuncNarrow {
  void operator()(T *In, T *Out, size_t Size) const {
    for (auto I = 0; I < Size; ++I) {
      In[I] = 2.0f + 16.0f * ((T)I / (T)(Size - 1)); // in [2..18] range
      Out[I] = (T)0;
    }
  }
};

template <class T> struct InitDataInRange0_5 {
  void operator()(T *In, T *Out, size_t Size) const {
    for (auto I = 0; I < Size; ++I) {
      In[I] = 5.0f * ((T)I / (T)(Size - 1)); // in [0..5] range
      Out[I] = (T)0;
    }
  }
};

template <class T> struct InitDataBinFuncNarrow {
  void operator()(T *In1, T *In2, T *Out, size_t Size) const {
    for (auto I = 0; I < Size; ++I) {
      In1[I] = I % 17 + 1;
      In2[I] = 4.0f * ((T)I / (T)(Size - 1)); // in [0..4] range
      Out[I] = (T)0;
    }
  }
};

// --- Math operation identification

enum class MathOp {
  sin,
  cos,
  exp,
  sqrt,
  sqrt_ieee,
  inv,
  log,
  rsqrt,
  trunc,
  exp2,
  log2,
  div_ieee,
  pow
};

// --- Template functions calculating given math operation on host and device

template <class T, int VL, MathOp Op> struct FuncESIMD;
template <class T, int VL, MathOp Op> struct BinFuncESIMD;
template <class T, int VL, MathOp Op> struct FuncSYCL;
template <class T, MathOp Op> struct HostFunc;

#define DEFINE_HOST_OP(Op, HostOp)                                             \
  template <class T> struct HostFunc<T, MathOp::Op> {                          \
    T operator()(T X) { return HostOp; }                                       \
  };

DEFINE_HOST_OP(sin, std::sin(X));
DEFINE_HOST_OP(cos, std::cos(X));
DEFINE_HOST_OP(exp, std::exp(X));
DEFINE_HOST_OP(log, std::log(X));
DEFINE_HOST_OP(inv, 1.0f / X);
DEFINE_HOST_OP(sqrt, std::sqrt(X));
DEFINE_HOST_OP(sqrt_ieee, std::sqrt(X));
DEFINE_HOST_OP(rsqrt, 1.0f / std::sqrt(X));
DEFINE_HOST_OP(trunc, std::trunc(X));
DEFINE_HOST_OP(exp2, std::exp2(X));
DEFINE_HOST_OP(log2, std::log2(X));

#define DEFINE_HOST_BIN_OP(Op, HostOp)                                         \
  template <class T> struct HostFunc<T, MathOp::Op> {                          \
    T operator()(T X, T Y) { return HostOp; }                                  \
  };

DEFINE_HOST_BIN_OP(div_ieee, X / Y);
DEFINE_HOST_BIN_OP(pow, std::pow(X, Y));

// --- Specializations per each extended math operation

#define DEFINE_ESIMD_DEVICE_OP(Op)                                             \
  template <class T, int VL> struct FuncESIMD<T, VL, MathOp::Op> {             \
    simd<T, VL> operator()(const simd<T, VL> &X) const SYCL_ESIMD_FUNCTION {   \
      return esimd::Op<T, VL>(X);                                              \
    }                                                                          \
  };

DEFINE_ESIMD_DEVICE_OP(sin);
DEFINE_ESIMD_DEVICE_OP(cos);
DEFINE_ESIMD_DEVICE_OP(exp);
DEFINE_ESIMD_DEVICE_OP(log);
DEFINE_ESIMD_DEVICE_OP(inv);
DEFINE_ESIMD_DEVICE_OP(sqrt);
DEFINE_ESIMD_DEVICE_OP(sqrt_ieee);
DEFINE_ESIMD_DEVICE_OP(rsqrt);
DEFINE_ESIMD_DEVICE_OP(trunc);
DEFINE_ESIMD_DEVICE_OP(exp2);
DEFINE_ESIMD_DEVICE_OP(log2);

#define DEFINE_ESIMD_DEVICE_BIN_OP(Op)                                         \
  template <class T, int VL> struct BinFuncESIMD<T, VL, MathOp::Op> {          \
    simd<T, VL> operator()(const simd<T, VL> &X,                               \
                           const simd<T, VL> &Y) const SYCL_ESIMD_FUNCTION {   \
      return esimd::Op<T, VL>(X, Y);                                           \
    }                                                                          \
  };

DEFINE_ESIMD_DEVICE_BIN_OP(div_ieee);
DEFINE_ESIMD_DEVICE_BIN_OP(pow);

#define DEFINE_SYCL_DEVICE_OP(Op)                                              \
  template <class T, int VL> struct FuncSYCL<T, VL, MathOp::Op> {              \
    simd<T, VL> operator()(const simd<T, VL> &X) const SYCL_ESIMD_FUNCTION {   \
      /* T must be float for SYCL, so not a template parameter for sycl::Op*/  \
      return sycl::Op<VL>(X);                                                  \
    }                                                                          \
  };

DEFINE_SYCL_DEVICE_OP(sin);
DEFINE_SYCL_DEVICE_OP(cos);
DEFINE_SYCL_DEVICE_OP(exp);
DEFINE_SYCL_DEVICE_OP(log);

// --- Generic kernel calculating an extended math operation on array elements

template <class T, int VL, MathOp Op,
          template <class, int, MathOp> class Kernel, typename AccIn,
          typename AccOut>
struct UnaryDeviceFunc {
  AccIn In;
  AccOut Out;

  UnaryDeviceFunc(AccIn &In, AccOut &Out) : In(In), Out(Out) {}

  void operator()(id<1> I) const SYCL_ESIMD_KERNEL {
    unsigned int Offset = I * VL * sizeof(T);
    simd<T, VL> Vx;
    Vx.copy_from(In, Offset);
    Kernel<T, VL, Op> DevF{};
    Vx = DevF(Vx);
    Vx.copy_to(Out, Offset);
  };
};

template <class T, int VL, MathOp Op,
          template <class, int, MathOp> class Kernel, typename AccIn,
          typename AccOut>
struct BinaryDeviceFunc {
  AccIn In1;
  AccIn In2;
  AccOut Out;

  BinaryDeviceFunc(AccIn &In1, AccIn &In2, AccOut &Out)
      : In1(In1), In2(In2), Out(Out) {}

  void operator()(id<1> I) const SYCL_ESIMD_KERNEL {
    unsigned int Offset = I * VL * sizeof(T);
    simd<T, VL> V1(In1, Offset);
    simd<T, VL> V2(In2, Offset);
    Kernel<T, VL, Op> DevF{};
    simd<T, VL> V = DevF(V1, V2);
    V.copy_to(Out, Offset);
  };
};

// --- Generic test function for an extended math operation

template <class T, int VL, MathOp Op,
          template <class, int, MathOp> class Kernel,
          typename InitF = InitDataFuncNarrow<T>>
bool test(queue &Q, const std::string &Name,
          InitF InitData = InitDataFuncNarrow<T>{}, float delta = 0.0f) {

  constexpr size_t Size = 1024 * 128;
  constexpr bool IsBinOp = (Op == MathOp::div_ieee) || (Op == MathOp::pow);

  T *A = new T[Size];
  T *B = new T[Size];
  T *C = new T[Size];
  if constexpr (IsBinOp) {
    InitData(A, B, C, Size);
  } else {
    InitData(A, B, Size);
  }
  const char *kind = std::is_same_v<Kernel<T, VL, Op>, FuncESIMD<T, VL, Op>>
                         ? "ESIMD"
                         : "SYCL";
  std::cout << "  " << Name << " test, kind=" << kind << "...\n";

  try {
    buffer<T, 1> BufA(A, range<1>(Size));
    buffer<T, 1> BufB(B, range<1>(Size));
    buffer<T, 1> BufC(C, range<1>(Size));

    // number of workgroups
    cl::sycl::range<1> GlobalRange{Size / VL};

    // threads (workitems) in each workgroup
    cl::sycl::range<1> LocalRange{1};

    auto E = Q.submit([&](handler &CGH) {
      auto PA = BufA.template get_access<access::mode::read>(CGH);
      auto PC = BufC.template get_access<access::mode::write>(CGH);
      if constexpr (IsBinOp) {
        auto PB = BufB.template get_access<access::mode::read>(CGH);
        BinaryDeviceFunc<T, VL, Op, Kernel, decltype(PA), decltype(PC)> F(
            PA, PB, PC);
        CGH.parallel_for(nd_range<1>{GlobalRange, LocalRange}, F);
      } else {
        UnaryDeviceFunc<T, VL, Op, Kernel, decltype(PA), decltype(PC)> F(PA,
                                                                         PC);
        CGH.parallel_for(nd_range<1>{GlobalRange, LocalRange}, F);
      }
    });
    E.wait();
  } catch (sycl::exception &Exc) {
    std::cout << "    *** ERROR. SYCL exception caught: << " << Exc.what()
              << "\n";
    return false;
  }

  int ErrCnt = 0;

  for (unsigned I = 0; I < Size; ++I) {
    T Gold;

    if constexpr (IsBinOp) {
      Gold = HostFunc<T, Op>{}((T)A[I], (T)B[I]);
    } else {
      Gold = HostFunc<T, Op>{}((T)A[I]);
    }
    T Test = C[I];

    if (delta == 0.0f) {
      delta = sizeof(T) > 2 ? 0.0001 : 0.01;
    }

    if (abs(Test - Gold) > delta) {
      if (++ErrCnt < 10) {
        std::cout << "    failed at index " << I << ", " << Test
                  << " != " << Gold << " (gold)\n";
      }
    }
  }
  delete[] A;
  delete[] B;
  delete[] C;

  if (ErrCnt > 0) {
    std::cout << "    pass rate: "
              << ((float)(Size - ErrCnt) / (float)Size) * 100.0f << "% ("
              << (Size - ErrCnt) << "/" << Size << ")\n";
  }

  std::cout << (ErrCnt > 0 ? "    FAILED\n" : "    Passed\n");
  return ErrCnt == 0;
}

// --- Tests all extended math operations with given vector length

template <class T, int VL> bool testESIMD(queue &Q) {
  bool Pass = true;

  std::cout << "--- TESTING ESIMD functions, T=" << typeid(T).name()
            << ", VL = " << VL << "...\n";

  Pass &=
      test<T, VL, MathOp::sqrt, FuncESIMD>(Q, "sqrt", InitDataFuncWide<T>{});
  Pass &= test<T, VL, MathOp::inv, FuncESIMD>(Q, "inv");
  Pass &= test<T, VL, MathOp::rsqrt, FuncESIMD>(Q, "rsqrt");
  Pass &= test<T, VL, MathOp::sin, FuncESIMD>(Q, "sin", InitDataFuncTrig<T>{});
  Pass &= test<T, VL, MathOp::cos, FuncESIMD>(Q, "cos", InitDataFuncTrig<T>{});
  Pass &=
      test<T, VL, MathOp::exp, FuncESIMD>(Q, "exp", InitDataInRange0_5<T>{});
  Pass &= test<T, VL, MathOp::log, FuncESIMD>(Q, "log", InitDataFuncWide<T>{});
  Pass &=
      test<T, VL, MathOp::exp2, FuncESIMD>(Q, "exp2", InitDataInRange0_5<T>{});
  Pass &=
      test<T, VL, MathOp::log2, FuncESIMD>(Q, "log2", InitDataFuncWide<T>{});
  Pass &=
      test<T, VL, MathOp::trunc, FuncESIMD>(Q, "trunc", InitDataFuncWide<T>{});
  return Pass;
}

template <class T, int VL> bool testESIMDSqrtIEEE(queue &Q) {
  bool Pass = true;
  std::cout << "--- TESTING ESIMD sqrt_ieee, T=" << typeid(T).name()
            << ", VL = " << VL << "...\n";
  Pass &= test<T, VL, MathOp::sqrt_ieee, FuncESIMD>(Q, "sqrt_ieee",
                                                    InitDataFuncWide<T>{});
  return Pass;
}

template <class T, int VL> bool testESIMDDivIEEE(queue &Q) {
  bool Pass = true;
  std::cout << "--- TESTING ESIMD div_ieee, T=" << typeid(T).name()
            << ", VL = " << VL << "...\n";
  Pass &= test<T, VL, MathOp::div_ieee, BinFuncESIMD>(
      Q, "div_ieee", InitDataBinFuncNarrow<T>{});
  return Pass;
}

template <class T, int VL> bool testESIMDPow(queue &Q) {
  bool Pass = true;
  std::cout << "--- TESTING ESIMD pow, T=" << typeid(T).name()
            << ", VL = " << VL << "...\n";
  Pass &= test<T, VL, MathOp::pow, BinFuncESIMD>(
      Q, "pow", InitDataBinFuncNarrow<T>{}, 0.1);
  return Pass;
}

template <class T, int VL> bool testSYCL(queue &Q) {
  bool Pass = true;
  // TODO SYCL currently supports only these 4 functions, extend the test when
  // more are available.
  std::cout << "--- TESTING SYCL functions, T=" << typeid(T).name()
            << ", VL = " << VL << "...\n";
  // SYCL functions will have good accuracy for any argument, unlike bare h/w
  // ESIMD versions, so init with "wide" data set.
  Pass &= test<T, VL, MathOp::sin, FuncSYCL>(Q, "sin", InitDataFuncWide<T>{});
  Pass &= test<T, VL, MathOp::cos, FuncSYCL>(Q, "cos", InitDataFuncWide<T>{});
  Pass &= test<T, VL, MathOp::exp, FuncSYCL>(Q, "exp", InitDataInRange0_5<T>{});
  Pass &= test<T, VL, MathOp::log, FuncSYCL>(Q, "log", InitDataFuncWide<T>{});
  return Pass;
}

// --- The entry point

int main(void) {
  queue Q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<info::device::name>() << "\n";
  bool Pass = true;
  Pass &= testESIMD<half, 8>(Q);
  Pass &= testESIMD<float, 16>(Q);
  Pass &= testESIMD<float, 32>(Q);
  Pass &= testSYCL<float, 8>(Q);
  Pass &= testSYCL<float, 32>(Q);
  Pass &= testESIMDSqrtIEEE<float, 16>(Q);
  Pass &= testESIMDSqrtIEEE<double, 32>(Q);
  Pass &= testESIMDDivIEEE<float, 8>(Q);
  Pass &= testESIMDDivIEEE<double, 32>(Q);
  Pass &= testESIMDPow<float, 8>(Q);
  Pass &= testESIMDPow<half, 32>(Q);
  std::cout << (Pass ? "Test Passed\n" : "Test FAILED\n");
  return Pass ? 0 : 1;
}
