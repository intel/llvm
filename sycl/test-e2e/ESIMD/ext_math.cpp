//==---------------- ext_math.cpp  - DPC++ ESIMD extended math test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES-INTEL-DRIVER: lin: 27012, win: 101.4576
// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}
// RUN: %{build} -fsycl-device-code-split=per_kernel %{mathflags} -o %t.out
// RUN: %{run} %t.out

// This test checks extended math operations. Combinations of
// - argument type - half, float
// - math function - sin, cos, ..., div_ieee, pow
// - SYCL vs ESIMD APIs

#include "esimd_test_utils.hpp"

#include <sycl/builtins_esimd.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <iostream>

using namespace sycl;
using namespace sycl::ext::intel;

#ifdef SATURATION_ON
#define ESIMD_SATURATION_TAG                                                   \
  esimd::saturation_on_tag {}
#define ESIMD_SATURATE(T, x) esimd::saturate<T>(x)
#define HOST_SATURATE(x) std::max(0.0f, std::min((x), 1.0f))
#else
#define ESIMD_SATURATION_TAG                                                   \
  esimd::saturation_off_tag {}
#define ESIMD_SATURATE(T, x) (x)
#define HOST_SATURATE(x) (x)
#endif

// --- Data initialization functions

// Initialization data for trigonometric functions' input.
// H/w supports only limited range of sin/cos arguments with decent accuracy:
// absolute error <= 0.0008 for the range of +/- 32767*pi (+/- 102941).

constexpr int accuracy_limit = 32767 * 3.14 - 1;

template <class T> struct InitTrig {
  void operator()(T *In, T *Out, size_t Size) const {
    for (auto I = 0; I < Size; ++I) {
      In[I] = (I + 1) % accuracy_limit;
      Out[I] = (T)0;
    }
  }
};

template <class T> struct InitWide {
  void operator()(T *In, T *Out, size_t Size) const {
    for (auto I = 0; I < Size; ++I) {
      In[I] = I + 1.0;
      Out[I] = (T)0;
    }
  }
};

template <class T> struct InitNarrow {
  void operator()(T *In, T *Out, size_t Size) const {
    for (auto I = 0; I < Size; ++I) {
      In[I] = 2.0f + 16.0f * ((T)I / (T)(Size - 1)); // in [2..18] range
      Out[I] = (T)0;
    }
  }
};

template <class T> struct InitInRange0_5 {
  void operator()(T *In, T *Out, size_t Size) const {
    for (auto I = 0; I < Size; ++I) {
      In[I] = 5.0f * ((T)I / (T)(Size - 1)); // in [0..5] range
      Out[I] = (T)0;
    }
  }
};

template <class T> struct InitBin {
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
  floor,
  ceil,
  trunc,
  exp2,
  log2,
  div_ieee,
  pow
};

// --- Template functions calculating given math operation on host and device

enum ArgKind { AllVec, AllSca, Sca1Vec2, Sca2Vec1 };

template <class T, int N, MathOp Op, int Args = AllVec> struct ESIMDf;
template <class T, int N, MathOp Op, int Args = AllVec> struct BinESIMDf;
template <class T, int N, MathOp Op, int Args = AllVec> struct SYCLf;

template <class T, MathOp Op> struct HostFunc;

#define DEFINE_HOST_OP(Op, HostOp)                                             \
  template <class T> struct HostFunc<T, MathOp::Op> {                          \
    T operator()(T X) { return HOST_SATURATE(HostOp); }                        \
  };

DEFINE_HOST_OP(sin, std::sin(X));
DEFINE_HOST_OP(cos, std::cos(X));
DEFINE_HOST_OP(exp, std::exp(X));
DEFINE_HOST_OP(log, std::log(X));
DEFINE_HOST_OP(inv, 1.0f / X);
DEFINE_HOST_OP(sqrt, std::sqrt(X));
DEFINE_HOST_OP(sqrt_ieee, std::sqrt(X));
DEFINE_HOST_OP(rsqrt, 1.0f / std::sqrt(X));
DEFINE_HOST_OP(floor, std::floor(X));
DEFINE_HOST_OP(ceil, std::ceil(X));
DEFINE_HOST_OP(trunc, std::trunc(X));
DEFINE_HOST_OP(exp2, std::exp2(X));
DEFINE_HOST_OP(log2, std::log2(X));

#define DEFINE_HOST_BIN_OP(Op, HostOp)                                         \
  template <class T> struct HostFunc<T, MathOp::Op> {                          \
    T operator()(T X, T Y) { return HOST_SATURATE(HostOp); }                   \
  };

DEFINE_HOST_BIN_OP(div_ieee, X / Y);
DEFINE_HOST_BIN_OP(pow, std::pow(X, Y));

// --- Specializations per each extended math operation

#define DEFINE_ESIMD_DEVICE_OP(Op)                                             \
  template <class T, int N> struct ESIMDf<T, N, MathOp::Op, AllVec> {          \
    esimd::simd<T, N>                                                          \
    operator()(esimd::simd<T, N> X) const SYCL_ESIMD_FUNCTION {                \
      return esimd::Op<T, N>(X, ESIMD_SATURATION_TAG);                         \
    }                                                                          \
  };                                                                           \
  template <class T, int N> struct ESIMDf<T, N, MathOp::Op, AllSca> {          \
    esimd::simd<T, N> operator()(T X) const SYCL_ESIMD_FUNCTION {              \
      return esimd::Op<T, N>(X, ESIMD_SATURATION_TAG);                         \
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
DEFINE_ESIMD_DEVICE_OP(floor);
DEFINE_ESIMD_DEVICE_OP(ceil);
DEFINE_ESIMD_DEVICE_OP(trunc);
DEFINE_ESIMD_DEVICE_OP(exp2);
DEFINE_ESIMD_DEVICE_OP(log2);

#define DEFINE_ESIMD_DEVICE_BIN_OP(Op)                                         \
  template <class T, int N> struct BinESIMDf<T, N, MathOp::Op, AllSca> {       \
    esimd::simd<T, N> operator()(T X, T Y) const SYCL_ESIMD_FUNCTION {         \
      return esimd::Op<T, N>(X, Y, ESIMD_SATURATION_TAG);                      \
    }                                                                          \
  };                                                                           \
  template <class T, int N> struct BinESIMDf<T, N, MathOp::Op, AllVec> {       \
    esimd::simd<T, N>                                                          \
    operator()(esimd::simd<T, N> X,                                            \
               esimd::simd<T, N> Y) const SYCL_ESIMD_FUNCTION {                \
      return esimd::Op<T, N>(X, Y, ESIMD_SATURATION_TAG);                      \
    }                                                                          \
  };                                                                           \
  template <class T, int N> struct BinESIMDf<T, N, MathOp::Op, Sca1Vec2> {     \
    esimd::simd<T, N>                                                          \
    operator()(T X, esimd::simd<T, N> Y) const SYCL_ESIMD_FUNCTION {           \
      return esimd::Op<T, N>(X, Y, ESIMD_SATURATION_TAG);                      \
    }                                                                          \
  };                                                                           \
  template <class T, int N> struct BinESIMDf<T, N, MathOp::Op, Sca2Vec1> {     \
    esimd::simd<T, N> operator()(esimd::simd<T, N> X,                          \
                                 T Y) const SYCL_ESIMD_FUNCTION {              \
      return esimd::Op<T, N>(X, Y, ESIMD_SATURATION_TAG);                      \
    }                                                                          \
  };

DEFINE_ESIMD_DEVICE_BIN_OP(div_ieee);
DEFINE_ESIMD_DEVICE_BIN_OP(pow);

#define DEFINE_SYCL_DEVICE_OP(Op)                                              \
  template <class T, int N> struct SYCLf<T, N, MathOp::Op, AllVec> {           \
    esimd::simd<T, N>                                                          \
    operator()(esimd::simd<T, N> X) const SYCL_ESIMD_FUNCTION {                \
      /* T must be float for SYCL, so not a template parameter for sycl::Op*/  \
      return ESIMD_SATURATE(T, sycl::Op<N>(X));                                \
    }                                                                          \
  };                                                                           \
  template <class T, int N> struct SYCLf<T, N, MathOp::Op, AllSca> {           \
    esimd::simd<T, N> operator()(T X) const SYCL_ESIMD_FUNCTION {              \
      return ESIMD_SATURATE(T, sycl::Op<N>(X));                                \
    }                                                                          \
  };

DEFINE_SYCL_DEVICE_OP(sin);
DEFINE_SYCL_DEVICE_OP(cos);
DEFINE_SYCL_DEVICE_OP(exp);
DEFINE_SYCL_DEVICE_OP(log);

// --- Generic kernel calculating an extended math operation on array elements

template <class T, int N, MathOp Op,
          template <class, int, MathOp, int> class Kernel, typename AccIn,
          typename AccOut>
struct UnaryDeviceFunc {
  AccIn In;
  AccOut Out;

  UnaryDeviceFunc(AccIn &In, AccOut &Out) : In(In), Out(Out) {}

  void operator()(id<1> I) const SYCL_ESIMD_KERNEL {
    unsigned int Offset = I * N * sizeof(T);
    esimd::simd<T, N> Vx;
    Vx.copy_from(In, Offset);

    if (I.get(0) % 2 == 0) {
      for (int J = 0; J < N; J++) {
        Kernel<T, N, Op, AllSca> DevF{};
        T Val = Vx[J];
        esimd::simd<T, N> V = DevF(Val); // scalar arg
        Vx[J] = V[J];
      }
    } else {
      Kernel<T, N, Op, AllVec> DevF{};
      Vx = DevF(Vx); // vector arg
    }
    Vx.copy_to(Out, Offset);
  };
};

template <class T, int N, MathOp Op,
          template <class, int, MathOp, int> class Kernel, typename AccIn,
          typename AccOut>
struct BinaryDeviceFunc {
  AccIn In1;
  AccIn In2;
  AccOut Out;

  BinaryDeviceFunc(AccIn &In1, AccIn &In2, AccOut &Out)
      : In1(In1), In2(In2), Out(Out) {}

  void operator()(id<1> I) const SYCL_ESIMD_KERNEL {
    unsigned int Offset = I * N * sizeof(T);
    esimd::simd<T, N> V1(In1, Offset);
    esimd::simd<T, N> V2(In2, Offset);
    esimd::simd<T, N> V;

    if (I.get(0) % 2 == 0) {
      int Ind = 0;
      {
        Kernel<T, N, Op, AllSca> DevF{};
        T Val2 = V2[Ind];
        esimd::simd<T, N> Vv = DevF(V1[Ind], Val2); // both arguments are scalar
        V[Ind] = Vv[Ind];
      }
      Ind++;
      {
        Kernel<T, N, Op, Sca1Vec2> DevF{};
        T Val1 = V1[Ind];
        esimd::simd<T, N> Vv = DevF(Val1, V2); // scalar, vector
        V[Ind] = Vv[Ind];
      }
      Ind++;
      {
        for (int J = Ind; J < N; ++J) {
          Kernel<T, N, Op, Sca2Vec1> DevF{};
          T Val2 = V2[J];
          esimd::simd<T, N> Vv = DevF(V1, Val2); // scalar 2nd arg
          V[J] = Vv[J];
        }
      }
    } else {
      Kernel<T, N, Op, AllVec> DevF{};
      V = DevF(V1, V2); // vec 2nd arg
    }
    V.copy_to(Out, Offset);
  };
};

// --- Generic test function for an extended math operation

template <class T, int N, MathOp Op,
          template <class, int, MathOp, int> class Kernel,
          typename InitF = InitNarrow<T>>
bool test(queue &Q, const std::string &Name, InitF Init = InitNarrow<T>{},
          float delta = 0.0f) {

  constexpr size_t Size =
      std::is_same_v<T, sycl::half> ? (16 * 128) : (1024 * 128);
  constexpr bool IsBinOp = (Op == MathOp::div_ieee) || (Op == MathOp::pow);

  T *A = new T[Size];
  T *B = new T[Size];
  T *C = new T[Size];
  if constexpr (IsBinOp) {
    Init(A, B, C, Size);
  } else {
    Init(A, B, Size);
  }
  const char *kind =
      std::is_same_v<Kernel<T, N, Op, AllVec>, ESIMDf<T, N, Op, AllVec>>
          ? "ESIMD"
          : "SYCL";
  std::cout << "  " << Name << " test, kind=" << kind << "...\n";

  try {
    buffer<T, 1> BufA(A, range<1>(Size));
    buffer<T, 1> BufB(B, range<1>(Size));
    buffer<T, 1> BufC(C, range<1>(Size));

    // number of workgroups
    sycl::range<1> GlobalRange{Size / N};

    // threads (workitems) in each workgroup
    sycl::range<1> LocalRange{1};

    auto E = Q.submit([&](handler &CGH) {
      auto PA = BufA.template get_access<access::mode::read>(CGH);
      auto PC = BufC.template get_access<access::mode::write>(CGH);
      if constexpr (IsBinOp) {
        auto PB = BufB.template get_access<access::mode::read>(CGH);
        BinaryDeviceFunc<T, N, Op, Kernel, decltype(PA), decltype(PC)> F(PA, PB,
                                                                         PC);
        CGH.parallel_for(nd_range<1>{GlobalRange, LocalRange}, F);
      } else {
        UnaryDeviceFunc<T, N, Op, Kernel, decltype(PA), decltype(PC)> F(PA, PC);
        CGH.parallel_for(nd_range<1>{GlobalRange, LocalRange}, F);
      }
    });
    E.wait();
  } catch (sycl::exception &Exc) {
    std::cout << "    *** ERROR. SYCL exception caught: << " << Exc.what()
              << "\n";
    delete[] A;
    delete[] B;
    delete[] C;
    return false;
  }

  int ErrCnt = 0;

  for (unsigned I = 0; I < Size; ++I) {
    // functions like std::isinf/isfinite/isnan do not work correctly with
    // sycl::half, thus we'll use 'float' instead.
    using CheckT = std::conditional_t<std::is_same_v<T, sycl::half>, float, T>;

    CheckT Gold;

    if constexpr (IsBinOp) {
      Gold = HostFunc<T, Op>{}((T)A[I], (T)B[I]);
    } else {
      Gold = HostFunc<T, Op>{}((T)A[I]);
    }
    CheckT Test = C[I];

    if (delta == 0.0f)
      delta = 0.0001;
    if constexpr (sizeof(T) <= 2)
      delta = delta + delta;

    bool BothFinite = true;
#ifndef TEST_FAST_MATH
    BothFinite = std::isfinite(Test) && std::isfinite(Gold);
#endif
    if (BothFinite && std::abs(Test - Gold) > delta) {
      if (++ErrCnt < 10) {
        std::cout << "    failed at index " << I << ", " << Test
                  << " != " << Gold << " (gold)\n";
        std::cout << "    A = " << (T)A[I] << ", B = " << (T)B[I]
                  << ", diff = " << std::abs(Test - Gold)
                  << ", max-delta = " << delta << "\n";
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

template <class T, int N> bool testESIMD(queue &Q) {
  bool Pass = true;

  std::cout << "--- TESTING ESIMD functions, T=" << typeid(T).name()
            << ", N = " << N << "...\n";

  Pass &= test<T, N, MathOp::sqrt, ESIMDf>(Q, "sqrt", InitWide<T>{});
  Pass &= test<T, N, MathOp::inv, ESIMDf>(Q, "inv");
  Pass &= test<T, N, MathOp::rsqrt, ESIMDf>(Q, "rsqrt");
  Pass &= test<T, N, MathOp::sin, ESIMDf>(Q, "sin", InitTrig<T>{});
  Pass &= test<T, N, MathOp::cos, ESIMDf>(Q, "cos", InitTrig<T>{});
  Pass &= test<T, N, MathOp::exp, ESIMDf>(Q, "exp", InitInRange0_5<T>{});
  Pass &= test<T, N, MathOp::log, ESIMDf>(Q, "log", InitWide<T>{});
  Pass &= test<T, N, MathOp::exp2, ESIMDf>(Q, "exp2", InitInRange0_5<T>{});
  Pass &= test<T, N, MathOp::log2, ESIMDf>(Q, "log2", InitWide<T>{});
  Pass &= test<T, N, MathOp::floor, ESIMDf>(Q, "floor", InitWide<T>{});
  Pass &= test<T, N, MathOp::ceil, ESIMDf>(Q, "ceil", InitWide<T>{});
  Pass &= test<T, N, MathOp::trunc, ESIMDf>(Q, "trunc", InitWide<T>{});
  return Pass;
}

template <class T, int N> bool testESIMDSqrtIEEE(queue &Q) {
  bool Pass = true;
  std::cout << "--- TESTING ESIMD sqrt_ieee, T=" << typeid(T).name()
            << ", N = " << N << "...\n";
  Pass &= test<T, N, MathOp::sqrt_ieee, ESIMDf>(Q, "sqrt_ieee", InitWide<T>{});
  return Pass;
}

template <class T, int N> bool testESIMDDivIEEE(queue &Q) {
  bool Pass = true;
  std::cout << "--- TESTING ESIMD div_ieee, T=" << typeid(T).name()
            << ", N = " << N << "...\n";
  Pass &= test<T, N, MathOp::div_ieee, BinESIMDf>(Q, "div_ieee", InitBin<T>{});
  return Pass;
}

template <class T, int N> bool testESIMDPow(queue &Q) {
  bool Pass = true;
  std::cout << "--- TESTING ESIMD pow, T=" << typeid(T).name() << ", N = " << N
            << "...\n";
  Pass &= test<T, N, MathOp::pow, BinESIMDf>(Q, "pow", InitBin<T>{}, 0.1);
  return Pass;
}

template <class T, int N> bool testSYCL(queue &Q) {
  bool Pass = true;
  // TODO SYCL currently supports only these 4 functions, extend the test when
  // more are available.
  std::cout << "--- TESTING SYCL functions, T=" << typeid(T).name()
            << ", N = " << N << "...\n";
  // SYCL functions will have good accuracy for any argument, unlike bare h/w
  // ESIMD versions, so init with "wide" data set.
  Pass &= test<T, N, MathOp::sin, SYCLf>(Q, "sin", InitWide<T>{});
  Pass &= test<T, N, MathOp::cos, SYCLf>(Q, "cos", InitWide<T>{});
  Pass &= test<T, N, MathOp::exp, SYCLf>(Q, "exp", InitInRange0_5<T>{});
  Pass &= test<T, N, MathOp::log, SYCLf>(Q, "log", InitWide<T>{});
  return Pass;
}

// --- The entry point

int main(void) {
  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  esimd_test::printTestLabel(Q);
  auto Dev = Q.get_device();

  bool Pass = true;
#ifdef TEST_IEEE_DIV_REM
  Pass &= testESIMDSqrtIEEE<float, 16>(Q);
  Pass &= testESIMDDivIEEE<float, 8>(Q);
  if (Dev.has(sycl::aspect::fp64)) {
    Pass &= testESIMDSqrtIEEE<double, 32>(Q);
    Pass &= testESIMDDivIEEE<double, 32>(Q);
  }
#else // !TEST_IEEE_DIV_REM
  Pass &= testESIMD<half, 8>(Q);
  Pass &= testESIMD<float, 16>(Q);
  Pass &= testESIMD<float, 32>(Q);
#ifndef TEST_FAST_MATH
  // TODO: GPU Driver does not yet support ffast-math versions of tested APIs.
  Pass &= testSYCL<float, 8>(Q);
  Pass &= testSYCL<float, 32>(Q);
#endif
  Pass &= testESIMDPow<float, 8>(Q);
  Pass &= testESIMDPow<half, 32>(Q);
#endif // !TEST_IEEE_DIV_REM
  std::cout << (Pass ? "Test Passed\n" : "Test FAILED\n");
  return Pass ? 0 : 1;
}
