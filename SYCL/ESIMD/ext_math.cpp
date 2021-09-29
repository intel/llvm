//==---------------- ext_math.cpp  - DPC++ ESIMD extended math test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test checks extended math operations.

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/builtins_esimd.hpp>
#include <iostream>
#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

// --- Data initialization functions

struct InitDataFuncWide {
  void operator()(float *In, float *Out, size_t Size) const {
    for (auto I = 0; I < Size; ++I) {
      In[I] = I + 1.0;
      Out[I] = (float)0.0;
    }
  }
};

struct InitDataFuncNarrow {
  void operator()(float *In, float *Out, size_t Size) const {
    for (auto I = 0; I < Size; ++I) {
      In[I] = 2.0f + 16.0f * ((float)I / (float)(Size - 1)); // in [2..18] range
      Out[I] = (float)0.0;
    }
  }
};

struct InitDataInRange0_5 {
  void operator()(float *In, float *Out, size_t Size) const {
    for (auto I = 0; I < Size; ++I) {
      In[I] = 5.0f * ((float)I / (float)(Size - 1)); // in [0..5] range
      Out[I] = (float)0.0;
    }
  }
};

// --- Math operation identification

enum class MathOp { sin, cos, exp, sqrt, inv, log, rsqrt, trunc };

// --- Template functions calculating given math operation on host and device

template <int VL, MathOp Op> struct DeviceMathFunc;
template <MathOp Op> float HostMathFunc(float X);

// --- Specializations per each extended math operation

#define DEFINE_ESIMD_OP(Op, HostOp)                                            \
  template <> float HostMathFunc<MathOp::Op>(float X) { return HostOp(X); }    \
  template <int VL> struct DeviceMathFunc<VL, MathOp::Op> {                    \
    simd<float, VL>                                                            \
    operator()(const simd<float, VL> &X) const SYCL_ESIMD_FUNCTION {           \
      return esimd_##Op<VL>(X);                                                \
    }                                                                          \
  }

// The same as above but adds explicit template parameter for esimd_##Op.
#define DEFINE_ESIMD_RT_OP(Op, HostOp)                                         \
  template <> float HostMathFunc<MathOp::Op>(float X) { return HostOp(X); }    \
  template <int VL> struct DeviceMathFunc<VL, MathOp::Op> {                    \
    simd<float, VL>                                                            \
    operator()(const simd<float, VL> &X) const SYCL_ESIMD_FUNCTION {           \
      return esimd_##Op<float, VL>(X);                                         \
    }                                                                          \
  }

#define DEFINE_SIMD_OVERLOADED_STD_SYCL_OP(Op, HostOp)                         \
  template <> float HostMathFunc<MathOp::Op>(float X) { return HostOp(X); }    \
  template <int VL> struct DeviceMathFunc<VL, MathOp::Op> {                    \
    simd<float, VL>                                                            \
    operator()(const simd<float, VL> &X) const SYCL_ESIMD_FUNCTION {           \
      return sycl::Op<VL>(X);                                                  \
    }                                                                          \
  }

DEFINE_SIMD_OVERLOADED_STD_SYCL_OP(sin, sin);
DEFINE_SIMD_OVERLOADED_STD_SYCL_OP(cos, cos);
DEFINE_SIMD_OVERLOADED_STD_SYCL_OP(exp, exp);
DEFINE_SIMD_OVERLOADED_STD_SYCL_OP(log, log);
DEFINE_ESIMD_OP(inv, 1.0f /);
DEFINE_ESIMD_OP(sqrt, sqrt);
DEFINE_ESIMD_OP(rsqrt, 1.0f / sqrt);
DEFINE_ESIMD_RT_OP(trunc, trunc);

// --- Generic kernel calculating an extended math operation on array elements

template <MathOp Op, int VL, typename AccIn, typename AccOut>
struct DeviceFunc {
  AccIn In;
  AccOut Out;

  DeviceFunc(AccIn &In, AccOut &Out) : In(In), Out(Out) {}

  void operator()(id<1> I) const SYCL_ESIMD_KERNEL {
    unsigned int Offset = I * VL * sizeof(float);
    simd<float, VL> Vx;
    Vx.copy_from(In, Offset);
    DeviceMathFunc<VL, Op> DevF{};
    Vx = DevF(Vx);
    Vx.copy_to(Out, Offset);
  };
};

// --- Generic test function for an extended math operation

template <MathOp Op, int VL, typename InitF = InitDataFuncNarrow>
bool test(queue &Q, const std::string &Name,
          InitF InitData = InitDataFuncNarrow{}) {

  constexpr size_t Size = 1024 * 128;

  float *A = new float[Size];
  float *B = new float[Size];
  InitData(A, B, Size);
  std::cout << "  Running " << Name << " test, VL=" << VL << "...\n";

  try {
    buffer<float, 1> BufA(A, range<1>(Size));
    buffer<float, 1> BufB(B, range<1>(Size));

    // number of workgroups
    cl::sycl::range<1> GlobalRange{Size / VL};

    // threads (workitems) in each workgroup
    cl::sycl::range<1> LocalRange{1};

    auto E = Q.submit([&](handler &CGH) {
      auto PA = BufA.get_access<access::mode::read>(CGH);
      auto PB = BufB.get_access<access::mode::write>(CGH);
      DeviceFunc<Op, VL, decltype(PA), decltype(PB)> Kernel(PA, PB);
      CGH.parallel_for(nd_range<1>{GlobalRange, LocalRange}, Kernel);
    });
    E.wait();
  } catch (sycl::exception &Exc) {
    std::cout << "    *** ERROR. SYCL exception caught: << " << Exc.what()
              << "\n";
    return false;
  }

  int ErrCnt = 0;

  for (unsigned I = 0; I < Size; ++I) {
    float Gold = A[I];
    Gold = HostMathFunc<Op>(Gold);
    float Test = B[I];

    if (abs(Test - Gold) > 0.0001) {
      if (++ErrCnt < 10) {
        std::cout << "    failed at index " << I << ", " << Test
                  << " != " << Gold << " (gold)\n";
      }
    }
  }
  delete[] A;
  delete[] B;

  if (ErrCnt > 0) {
    std::cout << "    pass rate: "
              << ((float)(Size - ErrCnt) / (float)Size) * 100.0f << "% ("
              << (Size - ErrCnt) << "/" << Size << ")\n";
  }

  std::cout << (ErrCnt > 0 ? "    FAILED\n" : "    Passed\n");
  return ErrCnt == 0;
}

// --- Tests all extended math operations with given vector length

template <int VL> bool test(queue &Q) {
  bool Pass = true;

  Pass &= test<MathOp::sqrt, VL>(Q, "sqrt", InitDataFuncWide{});
  Pass &= test<MathOp::inv, VL>(Q, "inv");
  Pass &= test<MathOp::rsqrt, VL>(Q, "rsqrt");
  Pass &= test<MathOp::sin, VL>(Q, "sin", InitDataFuncWide{});
  Pass &= test<MathOp::cos, VL>(Q, "cos", InitDataFuncWide{});
  Pass &= test<MathOp::exp, VL>(Q, "exp", InitDataInRange0_5{});
  Pass &= test<MathOp::log, VL>(Q, "log", InitDataFuncWide{});
  Pass &= test<MathOp::trunc, VL>(Q, "trunc", InitDataFuncWide{});
  return Pass;
}

// --- The entry point

int main(void) {
  queue Q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<info::device::name>() << "\n";
  bool Pass = true;
  Pass &= test<8>(Q);
  Pass &= test<16>(Q);
  Pass &= test<32>(Q);
  std::cout << (Pass ? "Test Passed\n" : "Test FAILED\n");
  return Pass ? 0 : 1;
}
