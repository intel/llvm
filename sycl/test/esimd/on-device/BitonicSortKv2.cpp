//==---------------- BitonicSortKv2.cpp  - DPC++ ESIMD on-device test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// TODO enable on Windows and Level Zero
// REQUIRES: linux && gpu && opencl
// RUN: %clangxx-esimd -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %ESIMD_RUN_PLACEHOLDER %t.out

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <algorithm>
#include <iostream>

using namespace cl::sycl;
using namespace sycl::INTEL::gpu;
using namespace std;

#define LOG2_ELEMENTS 16 // 24

/*
 * To avoid loading/storing excessive data from/to memory, the
 * implementation of the bitonic sort here tries to take advantage of GRF
 * space and do as much work as possible locally without going through memory.
 * The algorithm is implemented using 2 kernels, cmk_bitonic_sort_256 and
 * cmk_bitonic_merge. Given an input, the algorithm first divides the data
 * into 256-element chunks sorted by each HW threads. Since 256 elements are
 * loaded into GRFs, swapping elements within a chunk leverages
 * the expressiveness of Gen register regioning. Once esimd_bitonic_sort_256
 * is complete, two neighboring segments of 256-element form a bitonic order.
 * Cmk_bitonic_merge takes two 256-chunks and performs swapping elements
 * based on the sorting order directions, ascending or descending.
 */

template <typename ty, uint32_t size>
ESIMD_INLINE simd<ty, size> cmk_read(ty *buf, uint32_t offset) {
  simd<ty, size> v;
#pragma unroll
  for (uint32_t i = 0; i < size; i += 32) {
    v.template select<32, 1>(i) = block_load<ty, 32>(buf + offset + i);
  }
  return v;
}

template <typename ty, uint32_t size>
ESIMD_INLINE void cmk_write(ty *buf, uint32_t offset, simd<ty, size> v) {
#pragma unroll
  for (uint32_t i = 0; i < size; i += 32) {
    block_store<ty, 32>(buf + offset + i, v.template select<32, 1>(i));
  }
}

#define BASE_SZ 256
#define MAX_TS_WIDTH 512

// Function bitonic_exchange{1,2,4,8} compares and swaps elements with
// the particular strides
ESIMD_INLINE simd<uint32_t, BASE_SZ>
bitonic_exchange8(simd<uint32_t, BASE_SZ> A, simd<ushort, 32> flip) {
  simd<uint32_t, BASE_SZ> B;
#pragma unroll
  for (int i = 0; i < BASE_SZ; i += 32) {
    B.select<8, 1>(i) = A.select<8, 1>(i + 8);
    B.select<8, 1>(i + 8) = A.select<8, 1>(i);
    B.select<8, 1>(i + 16) = A.select<8, 1>(i + 24);
    B.select<8, 1>(i + 24) = A.select<8, 1>(i + 16);
    B.select<32, 1>(i).merge(A.select<32, 1>(i),
                             (A.select<32, 1>(i) < B.select<32, 1>(i)) ^ flip);
  }
  return B;
}

ESIMD_INLINE simd<uint32_t, BASE_SZ>
bitonic_exchange4(simd<uint32_t, BASE_SZ> A, simd<ushort, 32> flip) {
  simd<uint32_t, BASE_SZ> B;
#pragma unroll
  for (int i = 0; i < BASE_SZ; i += 32) {
    auto MA = A.select<32, 1>(i).format<uint32_t, 4, 8>();
    auto MB = B.select<32, 1>(i).format<uint32_t, 4, 8>();
    MB.select<4, 1, 4, 1>(0, 0) = MA.select<4, 1, 4, 1>(0, 4);
    MB.select<4, 1, 4, 1>(0, 4) = MA.select<4, 1, 4, 1>(0, 0);
    B.select<32, 1>(i).merge(A.select<32, 1>(i),
                             (A.select<32, 1>(i) < B.select<32, 1>(i)) ^ flip);
  }
  return B;
}

// The implementation of bitonic_exchange2 is similar to bitonic_exchange1.
// The only difference is stride distance and different flip vector.
// However the shuffling data patterns for stride 2 cannot be expressed
// concisely with Gen register regioning.
// we format vector A and B into matrix_ref<long long, 4, 4>.
// MB.select<4, 1, 2, 2>(0, 0) can be mapped to destination region well.
//   MB.select<4, 1, 2, 2>(0, 0) = MA.select<4, 1, 2, 2>(0, 1);
// is compiled to
//   mov(4) r34.0<2>:q r41.1<2; 1, 0>:q { Align1, Q1 }
//   mov(4) r36.0<2>:q r8.1<2; 1, 0>:q { Align1, Q1 }
//   mov(4) r34.1<2>:q r41.0<2; 1, 0>:q { Align1, Q1 }
//   mov(4) r36.1<2>:q r8.0<2; 1, 0>:q { Align1, Q1 }
// each mov copies four 64-bit data, which is 4X SIMD efficiency
// improvement over the straightforward implementation.
ESIMD_INLINE simd<uint32_t, BASE_SZ>
bitonic_exchange2(simd<uint32_t, BASE_SZ> A, simd<ushort, 32> flip) {
  simd<uint32_t, BASE_SZ> B;
#pragma unroll
  for (int i = 0; i < BASE_SZ; i += 32) {
    auto MB = B.select<32, 1>(i).format<long long, 4, 4>();
    auto MA = A.select<32, 1>(i).format<long long, 4, 4>();
    MB.select<4, 1, 2, 2>(0, 0) = MA.select<4, 1, 2, 2>(0, 1);
    MB.select<4, 1, 2, 2>(0, 1) = MA.select<4, 1, 2, 2>(0, 0);
    B.select<32, 1>(i).merge(A.select<32, 1>(i),
                             (A.select<32, 1>(i) < B.select<32, 1>(i)) ^ flip);
  }
  return B;
}

ESIMD_INLINE simd<uint32_t, BASE_SZ>
bitonic_exchange1(simd<uint32_t, BASE_SZ> A, simd<ushort, 32> flip) {
  simd<uint32_t, BASE_SZ> B;
#pragma unroll
  // each thread is handling 256-element chunk. Each iteration
  // compares and swaps two 32 elements
  for (int i = 0; i < BASE_SZ; i += 32) {
    // The first step is to select A's odd-position elements,
    // indicated by A.select<16,2>(i), which selects 16 elements
    // with stride 2 starting from location A[i], and copies
    // the selected elements to B[i] location with stride 2.
    auto T = B.select<32, 1>(i);
    T.select<16, 2>(0) = A.select<16, 2>(i + 1);
    // The next step selects 16 even-position elements starting
    // from A[i+1] and copies them over to B's odd positions
    // starting at B[i+1]. After the first two steps,
    // all even-odd pair elements are swapped.
    T.select<16, 2>(1) = A.select<16, 2>(i);
    // The final step determines if the swapped pairs in B are
    // the desired order and should be preserved. If not, their values
    // are overwritten by their corresponding original values
    // (before swapping). The comparisons determine which elements
    // in B already meet the sorting order requirement and which are not.
    // Consider the first two elements of A & B, B[0] and B[1] is
    // the swap of A[0] and A[1]. Element-wise < comparison tells
    // that A[0] < B[0], i.e., A[0] < A[1]. Since the desired sorting
    // order is A[0] < A[1], however, we already swap the two values
    // as we copy A to B. The XOR operation is to set the condition to
    // indicate which elements in original vector A have the right sorting
    // order. Those elements are then merged from A to B based on their
    // corresponding conditions. Consider B[2] and B[3] in this case.
    // The order already satisfies the sorting order. The flip vector
    // passed to this stage is [0,1,1,0,0,1,1,0]. The flip bit of B[2]
    // resets the condition so that the later merge operation preserves
    // B[2] and won't copy from A[2].
    auto mask = flip ^ (A.select<32, 1>(i) < T);
    T.merge(A.select<32, 1>(i), mask);
  }
  return B;
}

// bitonic_merge for stage m has recursive steps to compare and swap
// elements with stride 1 << m, 1 << (m - 1), ... , 8, 4, 2, 1.
// bitonic_merge is GRF based implementation that handles stride
// 1 to 128 compare - and - swap steps.For stride <= 128, 256 data
// items are kept in GRF. All compare-and-swap can all be completely
// done with GRF locally. Doing so avoids global synchronizations
// and repeating loads/stores. Parameter n indicates that bitonic_merge
// is handling stride 1 << n for bitonic stage m.
ESIMD_INLINE void bitonic_merge(uint32_t offset, simd<uint32_t, BASE_SZ> &A,
                                uint32_t n, uint32_t m) {
  // dist is the stride distance for compare-and-swap
  uint32_t dist = 1 << n;
  // number of exchange passes we need
  // this loop handles stride distance 128 down to 16. Each iteration
  // the distance is halved. Due to data access patterns of stride
  // 8, 4, 2 and 1 are within one GRF, those stride distance are handled
  // by custom tailored code to take advantage of register regioning.
  for (int k = 0; k < n - 3; k++, dist >>= 1) {
    // Each HW thread process 256 data elements. For a given stride
    // distance S, 256 elements are divided into 256/(2*S) groups.
    // within each group, two elements with distance S apart are
    // compared and swapped based on sorting direction.
    // This loop basically iterates through each group.
    for (int i = 0; i < BASE_SZ; i += dist * 2) {
      // Every bitonic stage, we need to maintain bitonic sorting order.
      // Namely, data are sorted into alternating ascending and descending
      // fashion. As show in Figure 9, the light blue background regions
      // are in ascending order, the light green background regions in
      // descending order. Whether data are in ascending or descending
      // regions depends on their position and the current bitonic stage
      // they are in. "offset+i" the position. For stage m, data of
      // chunks of 1<<(m+1) elements in all the stride steps have the
      // same order.
      bool dir_up = (((offset + i) >> (m + 1)) & 1) == 0;
      // each iteration swap 2 16-element chunks
      for (int j = 0; j < (dist >> 4); j++) {
        simd<uint32_t, 16> T = A.select<16, 1>(i + j * 16);
        auto T1 = A.select<16, 1>(i + j * 16);
        auto T2 = A.select<16, 1>(i + j * 16 + dist);
        if (dir_up) {
          T1.merge(T2, T2 < T1);
          T2.merge(T, T > T2);
        } else {
          T1.merge(T2, T2 > T1);
          T2.merge(T, T < T2);
        }
      }
    }
  }

  // Stride 1, 2, 4, and 8 in bitonic_merge are custom tailored to
  // take advantage of register regioning. The implementation is
  // similar to bitonic_exchange{1,2,4,8}.

  // exchange 8
  simd<ushort, 32> flip13 = esimd_unpack_mask<32>(0xff00ff00); //(init_mask13);
  simd<ushort, 32> flip14 = esimd_unpack_mask<32>(0x00ff00ff); //(init_mask14);
  simd<uint32_t, BASE_SZ> B;
  for (int i = 0; i < BASE_SZ; i += 32) {
    B.select<8, 1>(i) = A.select<8, 1>(i + 8);
    B.select<8, 1>(i + 8) = A.select<8, 1>(i);
    B.select<8, 1>(i + 16) = A.select<8, 1>(i + 24);
    B.select<8, 1>(i + 24) = A.select<8, 1>(i + 16);
    bool dir_up = (((offset + i) >> (m + 1)) & 1) == 0;
    if (dir_up)
      B.select<32, 1>(i).merge(A.select<32, 1>(i),
                               (A.select<32, 1>(i) < B.select<32, 1>(i)) ^
                                   flip13);
    else
      B.select<32, 1>(i).merge(A.select<32, 1>(i),
                               (A.select<32, 1>(i) < B.select<32, 1>(i)) ^
                                   flip14);
  }

  // exchange 4
  simd<ushort, 32> flip15 = esimd_unpack_mask<32>(0xf0f0f0f0); //(init_mask15);
  simd<ushort, 32> flip16 = esimd_unpack_mask<32>(0x0f0f0f0f); //(init_mask16);
#pragma unroll
  for (int i = 0; i < BASE_SZ; i += 32) {
    auto MA = A.select<32, 1>(i).format<uint32_t, 4, 8>();
    auto MB = B.select<32, 1>(i).format<uint32_t, 4, 8>();
    MA.select<4, 1, 4, 1>(0, 0) = MB.select<4, 1, 4, 1>(0, 4);
    MA.select<4, 1, 4, 1>(0, 4) = MB.select<4, 1, 4, 1>(0, 0);
    bool dir_up = (((offset + i) >> (m + 1)) & 1) == 0;
    if (dir_up)
      A.select<32, 1>(i).merge(B.select<32, 1>(i),
                               (B.select<32, 1>(i) < A.select<32, 1>(i)) ^
                                   flip15);
    else
      A.select<32, 1>(i).merge(B.select<32, 1>(i),
                               (B.select<32, 1>(i) < A.select<32, 1>(i)) ^
                                   flip16);
  }

  // exchange 2
  simd<ushort, 32> flip17 = esimd_unpack_mask<32>(0xcccccccc); //(init_mask17);
  simd<ushort, 32> flip18 = esimd_unpack_mask<32>(0x33333333); //(init_mask18);
#pragma unroll
  for (int i = 0; i < BASE_SZ; i += 32) {
    auto MB = B.select<32, 1>(i).format<long long, 4, 4>();
    auto MA = A.select<32, 1>(i).format<long long, 4, 4>();

    MB.select<4, 1, 2, 2>(0, 0) = MA.select<4, 1, 2, 2>(0, 1);
    MB.select<4, 1, 2, 2>(0, 1) = MA.select<4, 1, 2, 2>(0, 0);
    bool dir_up = (((offset + i) >> (m + 1)) & 1) == 0;
    if (dir_up)
      B.select<32, 1>(i).merge(A.select<32, 1>(i),
                               (A.select<32, 1>(i) < B.select<32, 1>(i)) ^
                                   flip17);
    else
      B.select<32, 1>(i).merge(A.select<32, 1>(i),
                               (A.select<32, 1>(i) < B.select<32, 1>(i)) ^
                                   flip18);
  }
  // exchange 1
  simd<ushort, 32> flip19 = esimd_unpack_mask<32>(0xaaaaaaaa); //(init_mask19);
  simd<ushort, 32> flip20 = esimd_unpack_mask<32>(0x55555555); //(init_mask20);
#pragma unroll
  // Each iteration compares and swaps 2 32-element chunks
  for (int i = 0; i < BASE_SZ; i += 32) {
    // As aforementioned in bitonic_exchange1.
    // switch even and odd elements of B and put them in A.
    auto T = A.select<32, 1>(i);
    T.select<16, 2>(0) = B.select<16, 2>(i + 1);
    T.select<16, 2>(1) = B.select<16, 2>(i);
    // determine whether data are in ascending or descending regions
    // depends on their position and the current bitonic stage
    // they are in. "offset+i" is the position. For stage m,
    // data of chunks of 1<<(m+1) elements in all the stride steps
    // have the same order. For instance, in stage 4, all first 32 elements
    // are in ascending order and the next 32 elements are in descending
    // order. "&1" determines the alternating ascending and descending order.
    bool dir_up = (((offset + i) >> (m + 1)) & 1) == 0;
    // choose flip vector based on the direction (ascending or descending).
    // Compare and swap
    if (dir_up)
      A.select<32, 1>(i).merge(B.select<32, 1>(i),
                               (B.select<32, 1>(i) < T) ^ flip19);
    else
      A.select<32, 1>(i).merge(B.select<32, 1>(i),
                               (B.select<32, 1>(i) < T) ^ flip20);
  }
}

// sorting 256 elements in ascending or descending order
ESIMD_INLINE void cmk_bitonic_sort_256(uint32_t *buf1, uint32_t *buf2,
                                       uint32_t idx) {
  uint h_pos = idx;
  uint32_t offset = (h_pos * BASE_SZ);

  // The first few stages are implemented with double buffers, A and B,
  // which reside in GRF.The output of a stride exchange step is fed
  // into the next exchange step as the input.cmk_read loads a 256-element
  // chunk starting at offset into vector A. The flip vectors basically
  // indicate what the desired sorting order for swapping.
  simd<uint32_t, BASE_SZ> A;
  simd<uint32_t, BASE_SZ> B;
  A = cmk_read<uint32_t, BASE_SZ>(buf1, offset);

  simd<ushort, 32> flip1 = esimd_unpack_mask<32>(0x66666666); //(init_mask1);

  simd<unsigned short, 32> mask;
  // stage 0
  B = bitonic_exchange1(A, flip1);
  // stage 1
  simd<ushort, 32> flip2 = esimd_unpack_mask<32>(0x3c3c3c3c); //(init_mask2);
  simd<ushort, 32> flip3 = esimd_unpack_mask<32>(0x5a5a5a5a); //(init_mask3);
  A = bitonic_exchange2(B, flip2);
  B = bitonic_exchange1(A, flip3);
  // stage 2
  simd<ushort, 32> flip4 = esimd_unpack_mask<32>(0x0ff00ff0); //(init_mask4);
  simd<ushort, 32> flip5 = esimd_unpack_mask<32>(0x33cc33cc); //(init_mask5);
  simd<ushort, 32> flip6 = esimd_unpack_mask<32>(0x55aa55aa); //(init_mask6);
  A = bitonic_exchange4(B, flip4);
  B = bitonic_exchange2(A, flip5);
  A = bitonic_exchange1(B, flip6);
  // stage 3
  simd<ushort, 32> flip7 = esimd_unpack_mask<32>(0x00ffff00);  //(init_mask7);
  simd<ushort, 32> flip8 = esimd_unpack_mask<32>(0x0f0ff0f0);  //(init_mask8);
  simd<ushort, 32> flip9 = esimd_unpack_mask<32>(0x3333cccc);  //(init_mask9);
  simd<ushort, 32> flip10 = esimd_unpack_mask<32>(0x5555aaaa); //(init_mask10);
  B = bitonic_exchange8(A, flip7);
  A = bitonic_exchange4(B, flip8);
  B = bitonic_exchange2(A, flip9);
  A = bitonic_exchange1(B, flip10);
  // stage 4,5,6,7 use generic bitonic_merge routine
  for (int i = 4; i < 8; i++)
    bitonic_merge(h_pos * BASE_SZ, A, i, i);

  // cmk_write writes out sorted data to the output buffer.
  cmk_write<uint32_t, BASE_SZ>(buf2, offset, A);
}

ESIMD_INLINE void cmk_bitonic_merge(uint32_t *buf, uint32_t n, uint32_t m,
                                    uint32_t idx) {
  // threads are mapped to a 2D space. take 2D origin (x,y) and unfold them
  // to get the thread position in 1D space. use tid read the data chunks
  // the thread needs to read from the index surface
  uint tid = idx;
  // which 2-to-(n+1) segment the thread needs to work on
  // each thread swap two 256-element blocks.
  uint32_t seg = tid / (1 << (n - 8));
  uint32_t seg_sz = 1 << (n + 1);
  // calculate the offset of the data this HW is reading. seg*seg_sz is
  // the starting address of the segment the thread is in. As aforementioned,
  // each segment needs 1<<(n-8) threads. tid%(1<<(n-8) which 256-element
  // chunk within the segment this HW thread is processing.
  uint32_t offset = (seg * seg_sz + (tid % (1 << (n - 8)) * BASE_SZ));
  // stride distance
  uint32_t dist = 1 << n;
  // determine whether data are in ascending or descending regions depends on
  // their position and the current bitonic stage they are in.
  // "offset" is the position. For stage m, data of chunks of 1<<(m+1)
  // elements in all the stride steps have the same order.
  // "&1" determines the alternating ascending and descending order.
  bool dir_up = ((offset >> (m + 1)) & 1) == 0;
  // read oword 32 elements each time
  simd<uint32_t, BASE_SZ> A;
  simd<uint32_t, BASE_SZ> B;

#pragma unroll
  for (int i = 0; i < BASE_SZ; i += 32) {
    // byte offset
    A.select<32, 1>(i) = cmk_read<uint32_t, 32>(buf, (offset + i));
    B.select<32, 1>(i) = cmk_read<uint32_t, 32>(buf, (offset + i + dist));
    // compare 32 elements at a time and merge the result based on
    // the sorting direction
    simd<uint32_t, 32> T = A.select<32, 1>(i);
    if (dir_up) {
      A.select<32, 1>(i).merge(B.select<32, 1>(i),
                               B.select<32, 1>(i) < A.select<32, 1>(i));
      B.select<32, 1>(i).merge(T, T > B.select<32, 1>(i));
    } else {
      A.select<32, 1>(i).merge(B.select<32, 1>(i),
                               B.select<32, 1>(i) > A.select<32, 1>(i));
      B.select<32, 1>(i).merge(T, T < B.select<32, 1>(i));
    }
  }
  // Once stride distance 256 is reached, all subsequent recursive steps
  // (n = 7, 6, ..., 1) can be resolved locally as all data reside
  // in vector A and B. Thus, reduce the overhead of returning back to
  // the host side and relaunch tasks. Also writing data back to
  // memory and reading it back is avoided. bitonic_merge is
  // the routine explained earlier.
  if (n == 8) {
    // Vector A has 256 elements. Call bitonic_merge to process
    // the remaining stride distance for A. A's sorted result is
    // immediately written out to memory. Doing so avoids spilling
    // because A's lifetime ends without interfering with
    // bitonic_merge(... B ...)
    bitonic_merge(offset, A, 7, m);
    cmk_write<uint32_t, BASE_SZ>(buf, offset, A);
    bitonic_merge(offset + dist, B, 7, m);
    cmk_write<uint32_t, BASE_SZ>(buf, (offset + dist), B);
  } else {
    cmk_write<uint32_t, BASE_SZ>(buf, offset, A);
    cmk_write<uint32_t, BASE_SZ>(buf, (offset + dist), B);
  }
}

static double report_time(const string &msg, event e0, event en) {
  cl_ulong time_start =
      e0.get_profiling_info<info::event_profiling::command_start>();
  cl_ulong time_end =
      en.get_profiling_info<info::event_profiling::command_end>();
  double elapsed = (time_end - time_start) / 1e6;
  cout << msg << elapsed << " milliseconds" << std::endl;
  return elapsed;
}

struct BitonicSort {
  enum {
    base_sort_size_ = 256,
    base_split_size_ = base_sort_size_ << 1,
    base_merge_size_ = 256
  };
  uint32_t size_;
  uint32_t *pInputs_;
  uint32_t *pActualOutputs_;
  uint32_t *pExpectOutputs_;
  uint32_t *pSortDirections_;
  queue *pQueue_;

public:
  bool Run(queue *pQ, uint32_t size) {
    bool result = true;
    Setup(pQ, size);
    RunExpect();
    RunActual();
    return Validate();
  }
  void Setup(queue *pQ, uint32_t size);
  void RunActual();
  void RunExpect();
  bool Validate();
  int Solve(uint32_t *pInputs, uint32_t *pOutputs, uint32_t size);
  void Teardown();
};

#define MAX_TS_WIDTH 512

void BitonicSort::Setup(queue *pQ, uint32_t size) {
  size_ = size;
  pQueue_ = pQ;
  auto dev = pQueue_->get_device();
  auto ctxt = pQueue_->get_context();
  pSortDirections_ = static_cast<uint32_t *>(
      malloc_shared(size_ * sizeof(int) / base_sort_size_, dev, ctxt));
  memset(pSortDirections_, 0, sizeof(uint32_t) * (size_ / base_sort_size_));

  pSortDirections_[0] = 0;
  for (uint32_t scale = 1; scale < size_ / 256; scale <<= 1) {
    for (uint32_t i = 0; i < scale; i++) {
      pSortDirections_[scale + i] = !pSortDirections_[i];
    }
  }

  pInputs_ =
      static_cast<uint32_t *>(malloc_shared(size_ * sizeof(int), dev, ctxt));
  for (uint32_t i = 0; i < size_; ++i) {
    pInputs_[i] = rand();
    // pInputs_[i] = rand() % (1 << 15);
  }

  pActualOutputs_ =
      static_cast<uint32_t *>(malloc_shared(size_ * sizeof(int), dev, ctxt));
  memset(pActualOutputs_, 0, sizeof(uint32_t) * size_);

  pExpectOutputs_ =
      static_cast<uint32_t *>(malloc_shared(size_ * sizeof(int), dev, ctxt));
  memcpy(pExpectOutputs_, pInputs_, sizeof(uint32_t) * size_);
}

void BitonicSort::RunActual() { Solve(pInputs_, pActualOutputs_, size_); }

void BitonicSort::RunExpect() {
  std::sort(pExpectOutputs_, pExpectOutputs_ + size_);
}

bool BitonicSort::Validate() {
  for (uint32_t i = 0; i < size_; ++i) {
    if (pExpectOutputs_[i] != pActualOutputs_[i]) {
      cout << "Difference is detected at i= " << i
           << " Expect = " << pExpectOutputs_[i]
           << " Actual = " << pActualOutputs_[i] << std::endl;
      return false;
    }
  }
  return true;
}

int BitonicSort::Solve(uint32_t *pInputs, uint32_t *pOutputs, uint32_t size) {
  uint32_t width, height; // thread space width and height
  uint32_t total_threads = size / base_sort_size_;
  // create ranges
  // We need that many workitems
  auto SortGlobalRange = cl::sycl::range<1>(total_threads);
  // Number of workitems in a workgroup
  cl::sycl::range<1> SortLocalRange{1};

  // enqueue sort265 kernel
  double total_time = 0;
  auto e = pQueue_->submit([&](handler &cgh) {
    cgh.parallel_for<class Sort256>(
        SortGlobalRange * SortLocalRange, [=](id<1> i) SYCL_ESIMD_KERNEL {
          using namespace sycl::INTEL::gpu;
          cmk_bitonic_sort_256(pInputs, pOutputs, i);
        });
  });
  e.wait();
  total_time += report_time("kernel time", e, e);

  // Each HW thread swap two 256-element chunks. Hence, we only need
  // to launch size/ (base_sort_size*2) HW threads
  total_threads = size / (base_sort_size_ * 2);
  // create ranges
  // We need that many workitems
  auto MergeGlobalRange = cl::sycl::range<1>(total_threads);
  // Number of workitems in a workgroup
  cl::sycl::range<1> MergeLocalRange{1};

  // enqueue merge kernel multiple times
  // this loop is for stage 8 to stage LOG2_ELEMENTS.
  event mergeEvent[(LOG2_ELEMENTS - 8) * (LOG2_ELEMENTS - 7) / 2];
  int k = 0;
  for (int i = 8; i < LOG2_ELEMENTS; i++) {
    // each step halves the stride distance of its prior step.
    // 1<<j is the stride distance that the invoked step will handle.
    // The recursive steps continue until stride distance 1 is complete.
    // For stride distance less than 1<<8, no global synchronization
    // is needed, i.e., all work can be done locally within HW threads.
    // Hence, the invocation of j==8 cmk_bitonic_merge finishes stride 256
    // compare-and-swap and then performs stride 128, 64, 32, 16, 8, 4, 2, 1
    // locally.
    for (int j = i; j >= 8; j--) {
      mergeEvent[k] = pQueue_->submit([&](handler &cgh) {
        cgh.parallel_for<class Merge>(MergeGlobalRange * MergeLocalRange,
                                      [=](id<1> tid) SYCL_ESIMD_KERNEL {
                                        using namespace sycl::INTEL::gpu;
                                        cmk_bitonic_merge(pOutputs, j, i, tid);
                                      });
      });
      // mergeEvent[k].wait();
      k++;
    }
  }
  mergeEvent[k - 1].wait();
  total_time += report_time("kernel time", mergeEvent[0], mergeEvent[k - 1]);

  cout << " Sorting Time = " << total_time << " msec " << std::endl;
  return 1;
}

void BitonicSort::Teardown() {
  auto ctxt = pQueue_->get_context();
  free(pExpectOutputs_, ctxt);
  free(pInputs_, ctxt);
  free(pActualOutputs_, ctxt);
  free(pSortDirections_, ctxt);
}

int main(int argc, char *argv[]) {
  int size = 1 << LOG2_ELEMENTS;
  cout << "BitonicSort (" << size << ") Start..." << std::endl;

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler(),
          property::queue::enable_profiling{});

  BitonicSort bitonicSort;

  bool passed = bitonicSort.Run(&q, size);

  bitonicSort.Teardown();

  cout << (passed ? "=> PASSED" : "=> FAILED") << std::endl << std::endl;

  return !passed;
}
