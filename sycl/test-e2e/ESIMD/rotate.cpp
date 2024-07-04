//==- rotate.cpp - Test to verify ror/rol functionality ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -fsycl-device-code-split=per_kernel -std=c++20 -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -fsycl-device-code-split=per_kernel -std=c++20 -o %t1.out -DEXP
// RUN: %{run} %t1.out

// This is a basic test to validate the ror/rol functions.

#include "esimd_test_utils.hpp"
#include <bit>
#ifdef EXP
#define NS sycl::ext::intel::experimental::esimd
#else
#define NS sycl::ext::intel::esimd
#endif

// https://stackoverflow.com/questions/776508
template <typename T> T rotl(T n, unsigned int c) {
  const unsigned int mask = (CHAR_BIT * sizeof(n) - 1);
  c &= mask;
  return (n << c) | (n >> ((-c) & mask));
}

template <typename T> T rotr(T n, unsigned int c) {
  const unsigned int mask = (CHAR_BIT * sizeof(n) - 1);
  c &= mask;
  return (n >> c) | (n << ((-c) & mask));
}

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <typename DataT>
using shared_allocator = sycl::usm_allocator<DataT, sycl::usm::alloc::shared>;
template <typename DataT>
using shared_vector = std::vector<DataT, shared_allocator<DataT>>;

template <typename T> bool test_rotate(sycl::queue &Queue) {
  std::cout << "T=" << esimd_test::type_name<T>() << std::endl;
  shared_allocator<T> Allocator(Queue);
  constexpr int32_t VL = 16;

  shared_vector<T> OutputRorScalar(VL, 0, Allocator);
  for (T I = 0; I < VL; I++)
    OutputRorScalar[I] = I;
  shared_vector<T> OutputRolScalar = OutputRorScalar;
  shared_vector<T> OutputRolVector = OutputRorScalar;
  shared_vector<T> OutputRorVector = OutputRorScalar;

  std::vector<T> ExpectedRorScalar(OutputRorScalar.begin(),
                                   OutputRorScalar.end());
  std::vector<T> ExpectedRolScalar(OutputRolScalar.begin(),
                                   OutputRolScalar.end());
  std::vector<T> ExpectedRorVector(OutputRorVector.begin(),
                                   OutputRorVector.end());
  std::vector<T> ExpectedRolVector(OutputRolVector.begin(),
                                   OutputRolVector.end());

  int ScalarRotateFactor = 3;
  shared_vector<T> VectorRotateFactor(VL, 0, Allocator);
  for (T I = 0; I < VL; I++)
    VectorRotateFactor[I] = VL - I;

  auto *ScalarRorPtr = OutputRorScalar.data();
  auto *ScalarRolPtr = OutputRolScalar.data();
  auto *VectorRorPtr = OutputRorVector.data();
  auto *VectorRolPtr = OutputRolVector.data();
  auto *VectorRotateFactorPtr = VectorRotateFactor.data();
  Queue.submit([&](sycl::handler &cgh) {
    auto Kernel = ([=]() [[intel::sycl_explicit_simd]] {
      simd<T, VL> va(ScalarRorPtr);
      va = NS::ror<T>(va, ScalarRotateFactor);
      va.copy_to(ScalarRorPtr);

      simd<T, VL> vb(ScalarRolPtr);
      vb = NS::rol<T>(vb, ScalarRotateFactor);
      vb.copy_to(ScalarRolPtr);

      simd<T, VL> SimdRotateFac(VectorRotateFactorPtr);
      simd<T, VL> vc(VectorRorPtr);
      vc = NS::ror<T>(vc, SimdRotateFac);
      vc.copy_to(VectorRorPtr);

      simd<T, VL> vd(VectorRolPtr);
      vd = NS::rol<T>(vd, SimdRotateFac);
      vd.copy_to(VectorRolPtr);
    });
    cgh.single_task(Kernel);
  });
  Queue.wait();

  for (int I = 0; I < VL; I++) {
    using OpT = std::make_unsigned_t<T>;
    ExpectedRorScalar[I] = rotr<OpT>(sycl::bit_cast<OpT>(ExpectedRorScalar[I]),
                                     ScalarRotateFactor);
    ExpectedRolScalar[I] = rotl<OpT>(sycl::bit_cast<OpT>(ExpectedRolScalar[I]),
                                     ScalarRotateFactor);
    ExpectedRorVector[I] = rotr<OpT>(sycl::bit_cast<OpT>(ExpectedRorVector[I]),
                                     VectorRotateFactor[I]);
    ExpectedRolVector[I] = rotl<OpT>(sycl::bit_cast<OpT>(ExpectedRolVector[I]),
                                     VectorRotateFactor[I]);
  }
  for (int I = 0; I < VL; I++) {
    if (ExpectedRorScalar[I] != OutputRorScalar[I]) {
      std::cout << "Scalar ROR: error at I = " << std::to_string(I) << ": "
                << std::to_string(ExpectedRorScalar[I])
                << " != " << std::to_string(OutputRorScalar[I]) << std::endl;
      return false;
    }
  }

  for (int I = 0; I < VL; I++) {
    if (ExpectedRolScalar[I] != OutputRolScalar[I]) {
      std::cout << "Scalar ROL: error at I = " << std::to_string(I) << ": "
                << std::to_string(ExpectedRolScalar[I])
                << " != " << std::to_string(OutputRolScalar[I]) << std::endl;
      return false;
    }
  }
  for (int I = 0; I < VL; I++) {
    if (ExpectedRorVector[I] != OutputRorVector[I]) {
      std::cout << "Vector ROR: error at I = " << std::to_string(I) << ": "
                << std::to_string(ExpectedRorVector[I])
                << " != " << std::to_string(OutputRorVector[I]) << std::endl;
      return false;
    }
  }

  for (int I = 0; I < VL; I++) {
    if (ExpectedRolVector[I] != OutputRolVector[I]) {
      std::cout << "Vector ROL: error at I = " << std::to_string(I) << ": "
                << std::to_string(ExpectedRolVector[I])
                << " != " << std::to_string(OutputRolVector[I]) << std::endl;
      return false;
    }
  }

  return true;
}

int main() {

  bool Pass = true;
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);
  Pass &= test_rotate<uint32_t>(Q);
  Pass &= test_rotate<int32_t>(Q);
  Pass &= test_rotate<uint16_t>(Q);
  Pass &= test_rotate<int16_t>(Q);
#ifdef TEST_PVC
  Pass &= test_rotate<uint64_t>(Q);
  Pass &= test_rotate<int64_t>(Q);
#endif
  if (Pass) {
    std::cout << "Pass" << std::endl;
  }
  return !Pass;
}
