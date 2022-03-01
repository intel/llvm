//==---------------- esimd_rgba_smoke.cpp  - DPC++ ESIMD on-device test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to unimplemented 'single_task()' method
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Smoke test for scatter/gather also illustrating correct use of these APIs

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <iostream>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

static constexpr unsigned NAllChs =
    get_num_channels_enabled(rgba_channel_mask::ABGR);

template <class T> void print_ch(T *ch) {
  unsigned int v = (unsigned int)(*ch);
  std::cout << (char)(v >> 16) << (v & 0xFF);
}

template <class T> void print_pixels(const char *title, T *p0, int N) {

  std::cout << title << ":  ";
  for (unsigned i = 0; i < N; ++i) {
    T *p = p0 + i * NAllChs;

    std::cout << "{";
    for (unsigned ch = 0; ch < NAllChs; ++ch) {
      print_ch(p + ch);

      if (ch < NAllChs - 1) {
        std::cout << ",";
      }
    }
    std::cout << "}";
    std::cout << " ";
  }
  std::cout << "\n";
}

void print_mask(rgba_channel_mask m) {
  const char ch_names[] = {'R', 'G', 'B', 'A'};
  const rgba_channel ch_vals[] = {rgba_channel::R, rgba_channel::G,
                                  rgba_channel::B, rgba_channel::A};

  for (int ch = 0; ch < sizeof(ch_names) / sizeof(ch_names[0]); ++ch) {
    if (is_channel_enabled(m, ch_vals[ch])) {
      std::cout << ch_names[ch];
    }
  }
}

template <class, int, int> class TestID;

template <rgba_channel_mask ChMask, unsigned NPixels, class T>
bool test_impl(queue q) {
  constexpr unsigned NOnChs = get_num_channels_enabled(ChMask);
  unsigned SizeIn = NPixels * NAllChs;
  unsigned SizeOut = NPixels * NOnChs;

  std::cout << "Testing mask=";
  print_mask(ChMask);
  std::cout << ", T=" << typeid(T).name() << ", NPixels=" << NPixels << "\n";

  T *A = malloc_shared<T>(SizeIn, q);
  T *B = malloc_shared<T>(SizeOut, q);
  T *C = malloc_shared<T>(SizeOut, q);

  for (unsigned p = 0; p < NPixels; ++p) {
    char ch_names[] = {'R', 'G', 'B', 'A'};

    for (int ch = 0; ch < sizeof(ch_names) / sizeof(ch_names[0]); ++ch) {
      A[p * NAllChs + ch] =
          (ch_names[ch] << 16) | p; // R0 G0 B0 A0 R1 G1 B1 ...
      B[p * NAllChs + ch] = 0;
      C[p * NAllChs + ch] = 0;
    }
  }
  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.single_task<TestID<T, NPixels, static_cast<int>(ChMask)>>(
          [=]() SYCL_ESIMD_KERNEL {
            constexpr unsigned NElems = NPixels * NOnChs;
            simd<T, NPixels> offsets(0, sizeof(T) * NAllChs);
            simd<T, NElems> p = gather_rgba<T, NPixels, ChMask>(A, offsets);
            // simply scatter back to B - should give same results as A in
            // enabled channels, the rest should remain zero:
            scatter_rgba<T, NPixels, ChMask>(B, offsets, p);
            // copy instead of scattering to C - thus getting AOS to SOA layout
            // layout conversion:
            //   R0 R1 ... G0 G1 ... B0 B1 ... A0 A1 ...
            // or, if say R and B are disables (rgba_channel_mask::AG is used):
            //   G0 G1 ... A0 A1 ... 0 0 0 ...
            p.copy_to(C);
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(A, q);
    free(B, q);
    free(C, q);
    return 1;
  }
  print_pixels("  A", A, NPixels);
  print_pixels("  B", B, NPixels);
  print_pixels("  C", C, NPixels);
  int err_cnt = 0;

  // Total count of A's enabled channels iterated through at given moment
  unsigned on_ch_cnt_all = 0;

  // clang-format off
  //Testing mask=RA, T=unsigned int, NPixels=8
  //  A:  {R0,G0,B0,A0} {R1,G1,B1,A1} {R2,G2,B2,A2} {R3,G3,B3,A3} {R4,G4,B4,A4} {R5,G5,B5,A5} {R6,G6,B6,A6} {R7,G7,B7,A7}
  //  B:  {R0, 0, 0,A0} {R1, 0, 0,A1} {R2, 0, 0,A2} {R3, 0, 0,A3} {R4, 0, 0,A4} {R5, 0, 0,A5} {R6, 0, 0,A6} {R7, 0, 0,A7}
  //  C:  {R0,R1,R2,R3} {R4,R5,R6,R7} {A0,A1,A2,A3} {A4,A5,A6,A7} { 0, 0, 0, 0} { 0, 0, 0, 0} { 0, 0, 0, 0} { 0, 0, 0, 0}
  //  clang-format on

  for (unsigned p = 0; p < NPixels; ++p) {
    const char ch_names[] = {'R', 'G', 'B', 'A'};
    const rgba_channel ch_vals[] = {rgba_channel::R, rgba_channel::G,
                                    rgba_channel::B, rgba_channel::A};
    // Counts enabled channels in current A's pixel
    unsigned ch_on_cnt = 0;

    for (int ch = 0; ch < sizeof(ch_names) / sizeof(ch_names[0]); ++ch) {
      unsigned ch_off = p * NAllChs + ch;

      // check C
      // Are we past the payload in C and at the trailing 0 area?
      bool c_done = on_ch_cnt_all >= NPixels * NOnChs;

      if (c_done) {
        if ((T)0 != C[ch_off]) {
          ++err_cnt;
          std::cout << "  error in C: non-zero at pixel=" << p
                    << " channel=" << ch_names[ch] << "\n";
        }
      }
      if (is_channel_enabled(ChMask, ch_vals[ch])) {
        // check B
        if (A[ch_off] != B[ch_off]) {
          ++err_cnt;
          std::cout << "  error in B at pixel=" << p
                    << " channel=" << ch_names[ch] << ": ";
          print_ch(B + ch_off);
          std::cout << " != ";
          print_ch(A + ch_off);
          std::cout << " (gold)\n";
        }
        // check C
        on_ch_cnt_all++;
        unsigned ch_off_c = NPixels * ch_on_cnt + p;
        ch_on_cnt++;
        if (A[ch_off] != C[ch_off_c]) {
          ++err_cnt;
          std::cout << "  error in C at pixel=" << p
                    << " channel=" << ch_names[ch] << ": ";
          print_ch(C + ch_off_c);
          std::cout << " != ";
          print_ch(A + ch_off);
          std::cout << " (gold)\n";
        }
      } else {
        // check B
        if ((T)0 != B[ch_off]) {
          ++err_cnt;
          std::cout << "  error in B: non-zero at pixel=" << p
                    << " channel=" << ch_names[ch] << "\n";
        }
      }
    }
  }

  free(A, q);
  free(B, q);
  free(C, q);
  std::cout << (err_cnt > 0 ? " FAILED\n" : " Passed\n");
  return err_cnt == 0;
}

template <rgba_channel_mask ChMask> bool test(queue q) {
  bool passed = true;
  passed &= test_impl<ChMask, 8, unsigned int>(q);
  passed &= test_impl<ChMask, 16, float>(q);
  passed &= test_impl<ChMask, 32, int>(q);
  return passed;
}

int main(void) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  bool passed = true;
  passed &= test<rgba_channel_mask::ABGR>(q);
  passed &= test<rgba_channel_mask::AR>(q);
  passed &= test<rgba_channel_mask::A>(q);
  passed &= test<rgba_channel_mask::R>(q);
  passed &= test<rgba_channel_mask::B>(q);
  // TODO disabled due to a compiler bug:
  //passed &= test<rgba_channel_mask::ABR>(q);

  std::cout << (passed ? "Test passed\n" : "Test FAILED\n");
  return passed ? 0 : 1;
}
