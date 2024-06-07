//==--local_accessor_gather_scatter_rgba.cpp  - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// REQUIRES-INTEL-DRIVER: lin: 26690, win: 101.4576
// The test checks functionality of the gather_rgba/scatter_rgba local
// accessor-based ESIMD intrinsics.

#include "esimd_test_utils.hpp"

using namespace sycl;

constexpr int MASKED_LANE_NUM_REV = 1;
constexpr int NUM_RGBA_CHANNELS =
    get_num_channels_enabled(sycl::ext::intel::esimd::rgba_channel_mask::ABGR);

std::string convertMaskToStr(sycl::ext::intel::esimd::rgba_channel_mask mask) {
  using namespace sycl::ext::intel::esimd;
  switch (mask) {
  case rgba_channel_mask::R:
    return "R";
  case rgba_channel_mask::GR:
    return "GR";
  case rgba_channel_mask::ABGR:
    return "ABGR";
  default:
    return "";
  }
  return "";
}

template <typename T, unsigned VL, auto CH_MASK> bool test(queue q) {
  using namespace sycl::ext::intel::esimd;
  constexpr int numChannels = get_num_channels_enabled(CH_MASK);
  constexpr size_t size = VL * numChannels;

  std::cout << "Testing T=" << typeid(T).name() << " VL=" << VL
            << " MASK=" << convertMaskToStr(CH_MASK) << "...\n";

  // The test is going to use size elements of T type.
  auto Dev = q.get_device();
  auto DeviceSLMSize = Dev.get_info<sycl::info::device::local_mem_size>();
  if (DeviceSLMSize < VL * NUM_RGBA_CHANNELS * sizeof(T)) {
    // Report an error - the test needs a fix.
    std::cerr << "Error: Test needs more SLM memory than device has!"
              << std::endl;
    return false;
  }

  T *A = new T[size];
  T *gold = new T[size];

  for (int i = 0; i < size; ++i) {
    A[i] = static_cast<T>(-i);
  }

  // Fill out the array with gold values.
  // R  R  R  R   ... G  G  G  G   ... B  B  B   B   ... A  A  A   A  ...
  // 0, 4, 8, 12, ... 1, 5, 9, 13, ... 2, 6, 10, 14, ... 3, 7, 11, 15 ...
  for (unsigned i = 0; i < numChannels; i++) {
    for (unsigned j = 0; j < VL; j++) {
      // masked lane is assigned/verified separately:
      if (j == VL - MASKED_LANE_NUM_REV) {
        gold[i * VL + j] = -1;
      } else {
        gold[i * VL + j] = j * numChannels + i;
      }
    }
  }

  try {
    // We need that many workitems
    sycl::range<1> GlobalRange{1};
    // Number of workitems in a workgroup
    sycl::range<1> LocalRange{1};
    sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

    buffer<T, 1> OutBuf(A, range<1>(size));
    q.submit([&](handler &cgh) {
       auto OutAcc = OutBuf.template get_access<access::mode::read_write>(cgh);
       auto LocalAcc = local_accessor<T, 1>(VL * NUM_RGBA_CHANNELS, cgh);

       cgh.parallel_for(Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
         using namespace sycl::ext::intel::esimd;
         constexpr int numChannels = get_num_channels_enabled(CH_MASK);

         // Prepare initial values in SLM:
         // 0, -1, -2, -3, -4 ...
         // slm_scatter only supports VL = 16 or 32, so conservatively write in
         // chunks of 16 elements.
         constexpr unsigned numStores = (VL * NUM_RGBA_CHANNELS) / 16;
         for (int i = 0; i < numStores; i++) {
           simd<T, 16> vals(-i * 16, -1);
           simd<uint32_t, 16> fourByteOffsets(i * 16 * sizeof(T), sizeof(T));
           scatter<T, 16>(LocalAcc, fourByteOffsets, vals, 0);
         }

         // Prepare values to store into SLM in a SOA manner, e.g.:
         // R  R  R  R  R ...G  G  G  G  G ...B  B  B  B  B ...A  A  A  A  A ...
         // 00,04,08,12,16...01,05,09,13,17...02,06,10,14,18...03,07,11,15,19...
         simd<T, VL * numChannels> valsIn;
         for (unsigned i = 0; i < numChannels; i++)
           for (unsigned j = 0; j < VL; j++)
             valsIn[i * VL + j] = j * numChannels + i;

         // Store values to SLM. In the SLM it will be transposed into AOS:
         // R  G  B  A  R  G  B  A ...
         // 0, 1, 2, 3, 4, 5, 6, 7 ...
         simd<uint32_t, VL> byteOffsets(0, sizeof(T) * NUM_RGBA_CHANNELS);
         scatter_rgba<CH_MASK>(LocalAcc, byteOffsets, valsIn, 0);

         // Load back values from SLM. They will be transposed back to SOA.
         // "_" = "undefined" (masked out lane/pixel in each channel)
         // 00,04,08,12,...,_,01,05,09,13,...,_,02,06,10,14,...,_,03,07,11,19,...,_
         simd_mask<VL> pred = 1;
         pred[VL - MASKED_LANE_NUM_REV] = 0; // mask out the last lane
         simd<T, VL * numChannels> valsOut =
             gather_rgba<CH_MASK>(LocalAcc, byteOffsets, 0, pred);
         // replace undefined values in the masked out lane with something
         // verifiable
         if constexpr (numChannels == 1) {
           valsOut.template select<numChannels, 1>(VL - MASKED_LANE_NUM_REV) =
               -1;
         } else {
           valsOut.template select<numChannels, VL>(VL - MASKED_LANE_NUM_REV) =
               -1;
         }

         uint32_t global_offset = ndi.get_global_id(0) * VL * NUM_RGBA_CHANNELS;
         valsOut.copy_to(OutAcc, global_offset);
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cerr << "SYCL exception caught: " << e.what() << '\n';
    delete[] A;
    delete[] gold;
    return false; // not success
  }
  int err_cnt = 0;
  for (unsigned i = 0; i < size; ++i) {
    if (A[i] != gold[i]) {
      if (++err_cnt < 35) {
        std::cout << "\nFAILED at index " << i << ": " << A[i]
                  << " != " << gold[i] << " (gold)";
      }
    }
  }

  if (err_cnt > 0) {
    std::cout << "\n  pass rate: "
              << ((float)(size - err_cnt) / (float)size) * 100.0f << "% ("
              << (size - err_cnt) << "/" << size << ")\n";
  }

  delete[] A;
  delete[] gold;

  if (err_cnt == 0)
    std::cout << "Passed\n";
  return err_cnt == 0;
}

template <typename T, unsigned VL> bool test(queue q) {
  using namespace sycl::ext::intel::esimd;
  bool passed = true;
  passed &= test<T, VL, rgba_channel_mask::R>(q);
  passed &= test<T, VL, rgba_channel_mask::GR>(q);
  passed &= test<T, VL, rgba_channel_mask::ABGR>(q);
  return passed;
}

int main(void) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  bool passed = true;
  passed &= test<int, 8>(q);
  passed &= test<int, 16>(q);
  passed &= test<int, 32>(q);
  passed &= test<float, 8>(q);
  passed &= test<float, 16>(q);
  passed &= test<float, 32>(q);

  std::cout << (passed ? "All tests passed.\n" : "Some tests failed!\n");
  return passed ? 0 : 1;
}
