// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to unimplemented __esimd_scatter_scaled
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks functionality of the slm_gather_rgba/slm_scatter_rgba ESIMD
// API.

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace cl::sycl;

constexpr int MASKED_LANE_NUM_REV = 1;
constexpr int NUM_RGBA_CHANNELS = get_num_channels_enabled(
    sycl::ext::intel::experimental::esimd::rgba_channel_mask::ABGR);

template <typename T, unsigned VL, auto CH_MASK> struct Kernel {
  T *bufOut;
  Kernel(T *bufOut) : bufOut(bufOut) {}

  void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
    using namespace sycl::ext::intel::experimental::esimd;
    constexpr int numChannels = get_num_channels_enabled(CH_MASK);
    uint32_t i = ndi.get_global_id(0);

    // In this test, each group consist of one workitem. No barriers required.
    // Each workitem accesses contiguous block of VL elements, where
    // each element consists of RGBA channels.
    slm_init(VL * NUM_RGBA_CHANNELS * sizeof(T));

    // Prepare initial values in SLM:
    // 0, -1, -2, -3, -4 ...
    // slm_scatter only supports VL = 16 or 32, so conservatively write in
    // chunks of 16 elements.
    constexpr unsigned numStores = (VL * NUM_RGBA_CHANNELS) / 16;
    for (int i = 0; i < numStores; i++) {
      simd<T, 16> vals(-i * 16, -1);
      simd<uint32_t, 16> fourByteOffsets(i * 16 * sizeof(T), sizeof(T));
      slm_scatter<T, 16>(fourByteOffsets, vals);
    }

    // Prepare values to store into SLM in a SOA manner, e.g.:
    // R  R  R  R   ... G  G  G  G   ... B  B  B   B   ... A  A  A   A  ...
    // 0, 4, 8, 12, ... 1, 5, 9, 13, ... 2, 6, 10, 14, ... 3, 7, 11, 15 ...
    simd<T, VL * numChannels> valsIn;
    for (unsigned i = 0; i < numChannels; i++)
      for (unsigned j = 0; j < VL; j++)
        valsIn[i * VL + j] = j * numChannels + i;

    // Store values to SLM. In the SLM it will be transposed into AOS:
    // R  G  B  A  R  G  B  A ...
    // 0, 1, 2, 3, 4, 5, 6, 7 ...
    simd<uint32_t, VL> byteOffsets(0, sizeof(T) * NUM_RGBA_CHANNELS);
    slm_scatter_rgba<T, VL, CH_MASK>(byteOffsets, valsIn);

    // Load back values from SLM. They will be transposed back to SOA.
    simd_mask<VL> pred = 1;
    pred[VL - MASKED_LANE_NUM_REV] = 0; // mask out the last lane
    simd<T, VL *numChannels> valsOut =
        slm_gather_rgba<T, VL, CH_MASK>(byteOffsets, pred);

    // Copy results to the output USM buffer. Maximum write block size must be
    // at most 8 owords, so conservatively write in chunks of 8 elements.
    uint32_t global_offset = i * VL * NUM_RGBA_CHANNELS;
    for (unsigned i = 0; i < (VL * numChannels) / 8; i++) {
      simd<T, 8> valsToWrite = valsOut.template select<8, 1>(i * 8);
      valsToWrite.copy_to(bufOut + global_offset + i * 8);
    }
  }
};

std::string convertMaskToStr(
    sycl::ext::intel::experimental::esimd::rgba_channel_mask mask) {
  using namespace sycl::ext::intel::experimental::esimd;
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
  using namespace sycl::ext::intel::experimental::esimd;
  constexpr int numChannels = get_num_channels_enabled(CH_MASK);
  constexpr size_t size = VL * numChannels;

  std::cout << "Testing T=" << typeid(T).name() << " VL=" << VL
            << " MASK=" << convertMaskToStr(CH_MASK) << "...\n";

  auto dev = q.get_device();
  auto ctxt = q.get_context();
  T *A = static_cast<T *>(malloc_shared(size * sizeof(T), dev, ctxt));
  T *gold = new T[size];

  for (int i = 0; i < size; ++i) {
    A[i] = (T)-i;
  }

  // Fill out the array with gold values.
  // R  R  R  R   ... G  G  G  G   ... B  B  B   B   ... A  A  A   A  ...
  // 0, 4, 8, 12, ... 1, 5, 9, 13, ... 2, 6, 10, 14, ... 3, 7, 11, 15 ...
  for (unsigned i = 0; i < numChannels; i++)
    for (unsigned j = 0; j < VL; j++)
      gold[i * VL + j] = j * numChannels + i;

  // Account for masked out last lanes (with pred argument to slm_gather_rgba).
  unsigned maskedIndex = VL - 1;
  for (unsigned i = 0; i < numChannels; i++, maskedIndex += VL)
    gold[maskedIndex] = 0;

  try {
    // We need that many workitems
    sycl::range<1> GlobalRange{1};
    // Number of workitems in a workgroup
    sycl::range<1> LocalRange{1};
    sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

    auto e = q.submit([&](handler &cgh) {
      Kernel<T, VL, CH_MASK> kernel(A);
      cgh.parallel_for(Range, kernel);
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cerr << "SYCL exception caught: " << e.what() << '\n';
    free(A, ctxt);
    delete[] gold;
    return static_cast<bool>(e.code());
  }

  int err_cnt = 0;
  for (unsigned i = 0; i < size; ++i) {
    if (A[i] != gold[i]) {
      if (++err_cnt < 35) {
        std::cerr << "failed at index " << i << ": " << A[i]
                  << " != " << gold[i] << " (gold)\n";
      }
    }
  }

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(size - err_cnt) / (float)size) * 100.0f << "% ("
              << (size - err_cnt) << "/" << size << ")\n";
  }

  free(A, ctxt);
  delete[] gold;

  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  return err_cnt > 0 ? false : true;
}

template <typename T, unsigned VL> bool test(queue q) {
  using namespace sycl::ext::intel::experimental::esimd;
  bool passed = true;
  passed &= test<T, VL, rgba_channel_mask::R>(q);
  passed &= test<T, VL, rgba_channel_mask::GR>(q);
  passed &= test<T, VL, rgba_channel_mask::ABGR>(q);
  return passed;
}

int main(void) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
  passed &= test<int, 8>(q);
  passed &= test<int, 16>(q);
  passed &= test<int, 32>(q);
  passed &= test<float, 8>(q);
  passed &= test<float, 16>(q);
  passed &= test<float, 32>(q);

  return passed ? 0 : 1;
}
