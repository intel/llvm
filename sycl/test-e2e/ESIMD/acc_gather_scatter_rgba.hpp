//==-------- acc_gather_scatter_rgba.hpp  - DPC++ ESIMD on-device test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The test checks functionality of the gather_rgba/scatter_rgba accessor-based
// ESIMD intrinsics.

#include "esimd_test_utils.hpp"

using namespace sycl;

template <typename T>
using AccT = accessor<T, 1, access_mode::read_write, access::target::device>;

constexpr int MASKED_LANE_NUM_REV = 1;
constexpr int NUM_RGBA_CHANNELS =
    get_num_channels_enabled(sycl::ext::intel::esimd::rgba_channel_mask::ABGR);

template <typename T, unsigned VL, unsigned STRIDE, auto CH_MASK>
struct Kernel {
  AccT<T> InAcc;
  AccT<T> OutAcc;
  Kernel(AccT<T> InAcc, AccT<T> OutAcc) : InAcc(InAcc), OutAcc(OutAcc) {}

  void operator()(id<1> i) const SYCL_ESIMD_KERNEL {
    using namespace sycl::ext::intel::esimd;
    constexpr int numChannels = get_num_channels_enabled(CH_MASK);

    // Every workitem accesses contiguous block of VL * STRIDE elements,
    // where each element consists of RGBA channels.
    uint32_t global_offset = i * VL * STRIDE * NUM_RGBA_CHANNELS * sizeof(T);

    simd<uint32_t, VL> byteOffsets(0, STRIDE * sizeof(T) * NUM_RGBA_CHANNELS);
    simd<T, VL * numChannels> v;
    if constexpr (CH_MASK == rgba_channel_mask::ABGR)
      // Check that the default mask value is ABGR.
      v = gather_rgba(InAcc, byteOffsets, global_offset);
    else
      v = gather_rgba<CH_MASK>(InAcc, byteOffsets, global_offset);
    v += (int)i;

    simd_mask<VL> pred = 1;
    pred[VL - MASKED_LANE_NUM_REV] = 0; // mask out the last lane
    scatter_rgba<CH_MASK>(OutAcc, byteOffsets, v, global_offset, pred);
  }
};

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

template <typename T, unsigned VL, unsigned STRIDE, auto CH_MASK>
bool test(queue q) {
  size_t numWorkItems = 2;
  size_t size = VL * STRIDE * NUM_RGBA_CHANNELS * numWorkItems;
  using namespace sycl::ext::intel::esimd;
  constexpr int numChannels = get_num_channels_enabled(CH_MASK);

  std::cout << "Testing T=" << typeid(T).name() << " VL=" << VL
            << " STRIDE=" << STRIDE << " MASK=" << convertMaskToStr(CH_MASK)
            << "...\t";

  T *A = new T[size];
  T *B = new T[size];
  T *gold = new T[size];

  for (int i = 0; i < size; ++i) {
    A[i] = (T)i;
    B[i] = (T)-i;
    gold[i] = (T)-i;
  }

  // Fill out the array with gold values. The kernel only writes the elements
  // that are not masked. For example,
  // for STRIDE=1 and MASK=R, we have the following indices written:
  //   0, 4, 8, 12 ...
  // for STRIDE=2 and MASK=RG, we have the following indices written:
  //   0, 1, 8, 9, 16, 17 ...
  // All the other elements will be equal to '-A[i]'.
  auto blockSize = VL * STRIDE * NUM_RGBA_CHANNELS;
  for (unsigned i = 0; i < size; i += NUM_RGBA_CHANNELS * STRIDE)
    for (unsigned j = 0; j < numChannels; j++)
      gold[i + j] = A[i + j] + (i / (blockSize));

  // Account for masked out last lanes (with pred argument to scatter_rgba).
  auto maskedElementOffset = (VL - 1) * STRIDE * NUM_RGBA_CHANNELS;
  for (unsigned i = maskedElementOffset; i < size; i += blockSize)
    for (unsigned j = 0; j < numChannels; j++)
      gold[i + j] = -A[i + j];

  try {
    buffer<T, 1> InBuf(A, range<1>(size));
    buffer<T, 1> OutBuf(B, range<1>(size));
    range<1> glob_range{numWorkItems};
    auto e = q.submit([&](handler &cgh) {
      auto InAcc = InBuf.template get_access<access::mode::read_write>(cgh);
      auto OutAcc = OutBuf.template get_access<access::mode::read_write>(cgh);
      Kernel<T, VL, STRIDE, CH_MASK> kernel(InAcc, OutAcc);
      cgh.parallel_for(glob_range, kernel);
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cerr << "SYCL exception caught: " << e.what() << '\n';
    delete[] A;
    delete[] B;
    delete[] gold;
    return false; // not success
  }

  int err_cnt = 0;
  for (unsigned i = 0; i < size; ++i) {
    if (B[i] != gold[i]) {
      if (++err_cnt < 35) {
        std::cout << "\nFAILED at index " << i << ": " << B[i]
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
  delete[] B;
  delete[] gold;

  if (err_cnt == 0)
    std::cout << "Passed\n";
  return err_cnt == 0;
}

template <typename T, unsigned VL, unsigned STRIDE> bool test(queue q) {
  using namespace sycl::ext::intel::esimd;
  bool passed = true;
  passed &= test<T, VL, STRIDE, rgba_channel_mask::R>(q);
  passed &= test<T, VL, STRIDE, rgba_channel_mask::GR>(q);
  passed &= test<T, VL, STRIDE, rgba_channel_mask::ABGR>(q);
  return passed;
}

int main(void) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  bool passed = true;
  passed &= test<int, 16, 1>(q);
  passed &= test<int, 16, 2>(q);
  passed &= test<int, 16, 4>(q);
  passed &= test<int, 32, 1>(q);
  passed &= test<int, 32, 3>(q);
  passed &= test<int, 32, 8>(q);
  passed &= test<float, 16, 1>(q);
  passed &= test<float, 16, 2>(q);
  passed &= test<float, 16, 4>(q);
  passed &= test<float, 32, 1>(q);
  passed &= test<float, 32, 3>(q);
  passed &= test<float, 32, 8>(q);

  passed &= test<int, 8, 1>(q);
  passed &= test<int, 8, 3>(q);
  passed &= test<int, 8, 8>(q);
  passed &= test<float, 8, 1>(q);
  passed &= test<float, 8, 2>(q);
  passed &= test<float, 8, 4>(q);

  std::cout << (passed ? "All tests passed.\n" : "Some tests failed!\n");
  return passed ? 0 : 1;
}
