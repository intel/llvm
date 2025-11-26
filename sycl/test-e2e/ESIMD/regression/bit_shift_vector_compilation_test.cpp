//==- bit_shift_vector_compilation_test.cpp - test vector by vector shifts -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------===//

// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// This is a basic test to validate the vector bit shifting functions.

#include "../esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <typename DataT>
using shared_allocator = sycl::usm_allocator<DataT, sycl::usm::alloc::shared>;
template <typename DataT>
using shared_vector = std::vector<DataT, shared_allocator<DataT>>;

template <typename T> bool test_shift(sycl::queue &Queue) {
  std::cout << "T=" << esimd_test::type_name<T>() << std::endl;
  shared_allocator<T> Allocator(Queue);
  constexpr int32_t VL = 16;

  shared_vector<T> Shl(VL, 0, Allocator);
  for (T I = 0; I < VL; I++) {
    if (std::is_signed_v<T> && I < VL / 2)
      Shl[I] = std::numeric_limits<T>::min() + I;
    else
      Shl[I] = std::numeric_limits<T>::max() - I;
  }
  shared_vector<T> Shr = Shl;
  shared_vector<T> Asr = Shl;
  shared_vector<T> Lsr = Shl;

  std::vector<T> ExpectedShl(Shl.begin(), Shl.end());
  std::vector<T> ExpectedShr(Shr.begin(), Shr.end());
  std::vector<T> ExpectedLsr(Lsr.begin(), Lsr.end());
  std::vector<T> ExpectedAsr(Asr.begin(), Asr.end());

  int ScalarShiftFactor = 3;
  shared_vector<T> VectorShiftFactor(VL, 0, Allocator);
  for (T I = 0; I < VL; I++)
    VectorShiftFactor[I] = I % (sizeof(T) * 8);

  auto *ShlPtr = Shl.data();
  auto *ShrPtr = Shr.data();
  auto *LsrPtr = Lsr.data();
  auto *AsrPtr = Asr.data();
  auto *VectorShiftFactorPtr = VectorShiftFactor.data();
  Queue.submit([&](sycl::handler &cgh) {
    cgh.single_task([=]() [[intel::sycl_explicit_simd]] {
      simd<T, VL> ShiftFac(VectorShiftFactorPtr);
      simd<T, VL> shlV(ShlPtr);
      shlV = shl<T>(shlV, ShiftFac);
      shlV.copy_to(ShlPtr);

      simd<T, VL> shrV(ShrPtr);
      shrV = shr<T>(shrV, ShiftFac);
      shrV.copy_to(ShrPtr);

      simd<T, VL> lsrV(LsrPtr);
      lsrV = lsr<T>(lsrV, ShiftFac);
      lsrV.copy_to(LsrPtr);

      simd<T, VL> asrV(AsrPtr);
      asrV = asr<T>(asrV, ShiftFac);
      asrV.copy_to(AsrPtr);
    });
  });
  Queue.wait();
  for (int I = 0; I < VL; I++) {
    using UnsignedT = std::make_unsigned_t<T>;
    using SignedT = std::make_signed_t<T>;
    ExpectedShl[I] = ExpectedShl[I] << VectorShiftFactor[I];
    ExpectedLsr[I] =
        static_cast<
            typename __ESIMD_DNS::computation_type_t<UnsignedT, UnsignedT>>(
            ExpectedLsr[I]) >>
        VectorShiftFactor[I];
    ExpectedAsr[I] =
        static_cast<typename __ESIMD_DNS::computation_type_t<SignedT, SignedT>>(
            ExpectedAsr[I]) >>
        VectorShiftFactor[I];
    ExpectedShr[I] = std::is_signed_v<T> ? ExpectedAsr[I] : ExpectedLsr[I];
  }
  for (int I = 0; I < VL; I++) {
    if (ExpectedShl[I] != Shl[I]) {
      std::cout << "SHL: error at I = " << std::to_string(I) << ": "
                << std::to_string(ExpectedShl[I])
                << " != " << std::to_string(Shl[I]) << std::endl;
      return false;
    }
  }

  for (int I = 0; I < VL; I++) {
    if (ExpectedLsr[I] != Lsr[I]) {
      std::cout << "LSR: error at I = " << std::to_string(I) << ": "
                << std::to_string(ExpectedLsr[I])
                << " != " << std::to_string(Lsr[I]) << std::endl;
      return false;
    }
  }

  for (int I = 0; I < VL; I++) {
    if (ExpectedAsr[I] != Asr[I]) {
      std::cout << "ASR: error at I = " << std::to_string(I) << ": "
                << std::to_string(ExpectedAsr[I])
                << " != " << std::to_string(Asr[I]) << std::endl;
      return false;
    }
  }

  for (int I = 0; I < VL; I++) {
    if (ExpectedShr[I] != Shr[I]) {
      std::cout << "SHR: error at I = " << std::to_string(I) << ": "
                << std::to_string(ExpectedShr[I])
                << " != " << std::to_string(Shr[I]) << std::endl;
      return false;
    }
  }
  return true;
}

int main() {

  bool Pass = true;
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);
  Pass &= test_shift<uint8_t>(Q);
  Pass &= test_shift<int8_t>(Q);
  Pass &= test_shift<uint16_t>(Q);
  Pass &= test_shift<int16_t>(Q);
  Pass &= test_shift<uint32_t>(Q);
  Pass &= test_shift<int32_t>(Q);
#ifdef TEST_PVC
  Pass &= test_shift<uint64_t>(Q);
  Pass &= test_shift<int64_t>(Q);
#endif

  if (Pass) {
    std::cout << "Pass" << std::endl;
  }
  return !Pass;
}
