//==---------------- bfn.cpp - DPC++ ESIMD binary function test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-dg2 || arch-intel_gpu_pvc
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// This test checks binary function (bfn) operations. Combinations of
// - argument type - uint16_t, uint32_t.
// - binary function - several binary functins with three operands (~, &, |, ^).

#include "esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel;

// --- Initialization function for source operands of binary functions.

template <class T> struct InitOps {
  void operator()(T *In0, T *In1, T *In2, T *Out, size_t Size) const {
    for (auto I = 0; I < Size; ++I) {
      In0[I] = I * 3;
      In1[I] = I * 3 + 1;
      In2[I] = I * 3 + 2;
      Out[I] = (T)0;
    }
  }
};

// --- Test boolean control functions.

using bfn_t = esimd::bfn_t;

constexpr esimd::bfn_t F1 = bfn_t::x | bfn_t::y | bfn_t::z;
constexpr esimd::bfn_t F2 = bfn_t::x & bfn_t::y & bfn_t::z;
constexpr esimd::bfn_t F3 = ~bfn_t::x | bfn_t::y ^ bfn_t::z;

// --- Template functions calculating given boolean operation on host and device

enum ArgKind {
  AllVec,
  AllSca,
};

template <class T, esimd::bfn_t Op> struct HostFunc;

#define DEFINE_HOST_OP(FUNC_CTRL)                                              \
  template <class T> struct HostFunc<T, FUNC_CTRL> {                           \
    T operator()(T X0, T X1, T X2) {                                           \
      T res = 0;                                                               \
      for (unsigned i = 0; i < sizeof(X0) * 8; i++) {                          \
        T mask = T(0x1) << i;                                                  \
        res = (res & ~mask) |                                                  \
              ((static_cast<uint8_t>(FUNC_CTRL) >>                             \
                    ((((X0 >> i) & T(0x1))) + (((X1 >> i) & T(0x1)) << 1) +    \
                     (((X2 >> i) & T(0x1)) << 2)) &                            \
                T(0x1))                                                        \
               << i);                                                          \
      }                                                                        \
      return res;                                                              \
    }                                                                          \
  };

DEFINE_HOST_OP(F1);
DEFINE_HOST_OP(F2);
DEFINE_HOST_OP(F3);

// --- Specializations per each boolean operation.

template <class T, int N, esimd::bfn_t Op, int Args = AllVec> struct ESIMDf;

#define DEFINE_ESIMD_DEVICE_OP(FUNC_CTRL)                                      \
  template <class T, int N> struct ESIMDf<T, N, FUNC_CTRL, AllVec> {           \
    esimd::simd<T, N>                                                          \
    operator()(esimd::simd<T, N> X0, esimd::simd<T, N> X1,                     \
               esimd::simd<T, N> X2) const SYCL_ESIMD_FUNCTION {               \
      return esimd::bfn<FUNC_CTRL, T, N>(X0, X1, X2);                          \
    }                                                                          \
  };                                                                           \
  template <class T, int N> struct ESIMDf<T, N, FUNC_CTRL, AllSca> {           \
    esimd::simd<T, N> operator()(T X0, T X1, T X2) const SYCL_ESIMD_FUNCTION { \
      return esimd::bfn<FUNC_CTRL, T, N>(X0, X1, X2);                          \
    }                                                                          \
  };

DEFINE_ESIMD_DEVICE_OP(F1);
DEFINE_ESIMD_DEVICE_OP(F2);
DEFINE_ESIMD_DEVICE_OP(F3);

// --- Generic kernel calculating a binary function operation on array elements.

template <class T, int N, esimd::bfn_t Op,
          template <class, int, esimd::bfn_t, int> class Kernel>
struct DeviceFunc {
  const T *In0, *In1, *In2;
  T *Out;

  DeviceFunc(const T *In0, const T *In1, const T *In2, T *Out)
      : In0(In0), In1(In1), In2(In2), Out(Out) {}

  void operator()(nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
    auto gid = ndi.get_global_id(0);
    unsigned int Offset = gid * N;
    esimd::simd<T, N> V0;
    esimd::simd<T, N> V1;
    esimd::simd<T, N> V2;
    V0.copy_from(In0 + Offset);
    V1.copy_from(In1 + Offset);
    V2.copy_from(In2 + Offset);

    if (gid % 2 == 0) {
      for (int J = 0; J < N; J++) {
        Kernel<T, N, Op, AllSca> DevF{};
        T Val0 = V0[J];
        T Val1 = V1[J];
        T Val2 = V2[J];
        esimd::simd<T, N> V = DevF(Val0, Val1, Val2); // scalar arg
        V0[J] = V[J];
      }
    } else {
      Kernel<T, N, Op, AllVec> DevF{};
      V0 = DevF(V0, V1, V2); // vector arg
    }
    V0.copy_to(Out + Offset);
  };
};

// --- Generic test function for boolean function.

template <class T, int N, esimd::bfn_t Op, int Range,
          template <class, int, esimd::bfn_t, int> class Kernel,
          typename InitF = InitOps<T>>
bool test(queue &Q, const std::string &Name, InitF Init = InitOps<T>{}) {
  constexpr size_t Size = Range * N;

  auto UA = esimd_test::usm_malloc_shared<T>(Q, Size);
  T *A = UA.get();
  auto UB = esimd_test::usm_malloc_shared<T>(Q, Size);
  T *B = UB.get();
  auto UC = esimd_test::usm_malloc_shared<T>(Q, Size);
  T *C = UC.get();
  auto UD = esimd_test::usm_malloc_shared<T>(Q, Size);
  T *D = UD.get();
  Init(A, B, C, D, Size);

  std::cout << "  " << Name << " test"
            << "...\n";

  try {
    // number of workgroups
    sycl::range<1> GlobalRange{Range};

    // threads (workitems) in each workgroup
    sycl::range<1> LocalRange{1};

    auto E = Q.submit([=](handler &CGH) {
      DeviceFunc<T, N, Op, Kernel> F(A, B, C, D);
      CGH.parallel_for(nd_range<1>{GlobalRange, LocalRange}, F);
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

    Gold = HostFunc<T, Op>{}((T)A[I], (T)B[I], (T)C[I]);
    T Test = D[I];

    if (Test != Gold) {
      if (++ErrCnt < 10) {
        std::cout << "\tfailed at index " << I << ", " << std::hex << Test
                  << " != " << Gold << " (gold); "
                  << "Input was: " << (T)A[I] << ", " << (T)B[I] << ", "
                  << (T)C[I] << "; "
                  << "FuncCtrl: " << int(Op) << std::dec << "\n";
      }
    }
  }

  if (ErrCnt > 0) {
    std::cout << "    pass rate: "
              << ((float)(Size - ErrCnt) / (float)Size) * 100.0f << "% ("
              << (Size - ErrCnt) << "/" << Size << ")\n";
  }

  std::cout << (ErrCnt > 0 ? "    FAILED\n" : "    Passed\n");
  return ErrCnt == 0;
}

// --- Tests all boolean operations with given vector length.

template <class T, int N, int Range> bool testESIMD(queue &Q) {
  bool Pass = true;

  std::cout << "--- TESTING ESIMD functions, T=" << typeid(T).name()
            << ", N = " << N << ", Range: " << Range << "...\n";

  Pass &= test<T, N, F1, Range, ESIMDf>(Q, "F1");
  Pass &= test<T, N, F2, Range, ESIMDf>(Q, "F2");
  Pass &= test<T, N, F3, Range, ESIMDf>(Q, "F3");
  return Pass;
}

template <class T, int N> bool testESIMDRanges(queue &Q) {
  bool Pass = true;
  // Test vector API.
  Pass &= testESIMD<T, N, 128>(Q);
  // Test scalar API with odd size.
  Pass &= testESIMD<T, N, 101>(Q);
  return Pass;
}

template <class T> bool testESIMDGroup(queue &Q) {
  bool Pass = true;
  Pass &= testESIMDRanges<T, 1>(Q);
  Pass &= testESIMDRanges<T, 5>(Q);
  Pass &= testESIMDRanges<T, 8>(Q);
  Pass &= testESIMDRanges<T, 16>(Q);
  Pass &= testESIMDRanges<T, 32>(Q);
  return Pass;
}

// --- The entry point.

int main(void) {
  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  auto Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<sycl::info::device::name>()
            << "\n";
  bool Pass = true;

  Pass &= testESIMDGroup<uint16_t>(Q);
  Pass &= testESIMDGroup<uint32_t>(Q);
  Pass &= testESIMDGroup<int16_t>(Q);
  Pass &= testESIMDGroup<int32_t>(Q);

  Pass &= testESIMDGroup<uint8_t>(Q);
  Pass &= testESIMDGroup<int8_t>(Q);
  Pass &= testESIMDGroup<uint64_t>(Q);
  Pass &= testESIMDGroup<int64_t>(Q);

  std::cout << (Pass ? "Test Passed\n" : "Test FAILED\n");
  return Pass ? 0 : 1;
}
