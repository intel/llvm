//==------------ radix_sort.cpp - DPC++ ESIMD on-device test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test implements parallel radix sort on GPU

#include "esimd_test_utils.hpp"

#include <algorithm>
#include <iostream>
#include <sycl/CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#define LOG2_ELEMENTS 20

// the number of bits necessary for representing the radix R
#define N_RADIX_BITS 4
#define RADIX (1 << N_RADIX_BITS)
#define N_WI 16 // number of work items
#define N_ELEM 512
#define N_ELEM_WI (N_ELEM / N_WI)
#define LOCAL_SIZE 64

#define MAX_TS_WIDTH 1024
#define TUPLE_SZ 1 // kernel can only handle TUPLE_SZ can be 1, 2, 4

#if TUPLE_SZ == 1
#define GATHER_SCATTER_MASK rgba_channel_mask::R
#elif TUPLE_SZ == 2
#define GATHER_SCATTER_MASK rgba_channel_mask::GR
#elif TUPLE_SZ == 4
#define GATHER_SCATTER_MASK rgba_channel_mask::ABGR
#endif

#define LOG_ENTRIES 8
#define PREFIX_ENTRIES (1 << LOG_ENTRIES)
#define PREFIX_ENTRIES_LOW 32
#define ENTRIES_THRESHOLD 2048
#define MIN_NUM_THREADS 1
#define REMAINING_ENTRIES 64

using namespace sycl;
using namespace sycl::ext::intel::esimd;

void compute_local_prefixsum(unsigned int prefixSum[], unsigned int size,
                             unsigned elem_stride, unsigned thread_stride) {

  unsigned entry_per_thread = thread_stride / elem_stride;
  unsigned local_sum[TUPLE_SZ];
  for (unsigned k = 0; k < size; k += thread_stride) {
    memset(local_sum, 0, TUPLE_SZ * sizeof(unsigned)); // init 0

    for (int i = 0; i < entry_per_thread; i++) {
      for (int j = 0; j < TUPLE_SZ; j++) {
        local_sum[j] +=
            prefixSum[(k + (i + 1) * elem_stride - 1) * TUPLE_SZ + j];
      }
    }
    // store local_sum in the last entry
    memcpy(&prefixSum[(k + entry_per_thread * elem_stride - 1) * TUPLE_SZ],
           local_sum, TUPLE_SZ * sizeof(unsigned));
  }
}

void compute_local_prefixsum_remaining(unsigned int prefixSum[],
                                       unsigned int size,
                                       unsigned elem_stride) {

  unsigned local_sum[TUPLE_SZ];
  memset(local_sum, 0, TUPLE_SZ * sizeof(unsigned)); // init 0

  for (int i = 0; i < size / elem_stride; i++) {
    for (int j = 0; j < TUPLE_SZ; j++) {
      local_sum[j] += prefixSum[((i + 1) * elem_stride - 1) * TUPLE_SZ + j];
    }
    // update every elem_stride entry
    memcpy(&prefixSum[((i + 1) * elem_stride - 1) * TUPLE_SZ], local_sum,
           TUPLE_SZ * sizeof(unsigned));
  }
}

void compute_prefixsum_up(unsigned int prefixSum[], unsigned int size,
                          unsigned elem_stride, unsigned thread_stride) {
  unsigned entry_per_thread = thread_stride / elem_stride;
  unsigned local_sum[TUPLE_SZ];
  for (unsigned k = 0; k < size; k += thread_stride) {
    if (k == 0)
      memset(local_sum, 0, TUPLE_SZ * sizeof(unsigned)); // init 0
    else // get the last entry from previous chunk
      memcpy(local_sum, &prefixSum[(k - 1) * TUPLE_SZ],
             TUPLE_SZ * sizeof(unsigned));

    // no need to update the last entry
    // the last entry has the correct value computed by compute_local_prefixsum
    for (int i = 0; i < entry_per_thread - 1; i++) {
      for (int j = 0; j < TUPLE_SZ; j++) {
        local_sum[j] +=
            prefixSum[(k + (i + 1) * elem_stride - 1) * TUPLE_SZ + j];
      }
      memcpy(&prefixSum[(k + (i + 1) * elem_stride - 1) * TUPLE_SZ], local_sum,
             TUPLE_SZ * sizeof(unsigned));
    }
  }
}

void compute_prefixsum(unsigned int input[], unsigned int prefixSum[],
                       unsigned int size) {

  for (int j = 0; j < TUPLE_SZ; j++) // init first entry
    prefixSum[j] = input[j];

  for (int i = 1; i < size; i++) {
    for (int j = 0; j < TUPLE_SZ; j++) {
      prefixSum[i * TUPLE_SZ + j] =
          input[i * TUPLE_SZ + j] + prefixSum[(i - 1) * TUPLE_SZ + j];
    }
  }
}

void cmk_acum_iterative(unsigned *buf, unsigned h_pos,
                        unsigned int stride_elems, unsigned int stride_threads,
                        unsigned n_entries) {

  simd<unsigned int, 32> element_offset(0, 1); // 0, 1, 2, ..., 31

  // global offset for a thread
  unsigned int global_offset = (h_pos * stride_threads * TUPLE_SZ);
  // element offsets for scattered read: [e0,e1,e2,...,e31] where e_i =
  // global_offset + # prefix_entries + prefix_entries - 1;
  element_offset =
      (((element_offset + 1) * stride_elems - 1) * TUPLE_SZ + global_offset) *
      sizeof(unsigned);

  simd<unsigned int, 32 * TUPLE_SZ> S, T;

#pragma unroll
  for (unsigned int i = 0; i < TUPLE_SZ; i++) {
    S.select<32, TUPLE_SZ>(i) =
        gather<unsigned, 32>(buf, element_offset + i * sizeof(unsigned));
  }

#pragma unroll
  for (int j = 1; j < n_entries / 32; j++) {
    element_offset += (stride_elems * 32 * TUPLE_SZ) * sizeof(unsigned);
#pragma unroll
    for (unsigned int i = 0; i < TUPLE_SZ; i++) {
      T.select<32, TUPLE_SZ>(i) =
          gather<unsigned, 32>(buf, element_offset + i * sizeof(unsigned));
    }
    S += T;
  }

  auto cnt_table = S.bit_cast_view<unsigned int, 32, TUPLE_SZ>();
  // sum reduction for each bin
  cnt_table.select<16, 1, TUPLE_SZ, 1>(0, 0) +=
      cnt_table.select<16, 1, TUPLE_SZ, 1>(16, 0);
  cnt_table.select<8, 1, TUPLE_SZ, 1>(0, 0) +=
      cnt_table.select<8, 1, TUPLE_SZ, 1>(8, 0);
  cnt_table.select<4, 1, TUPLE_SZ, 1>(0, 0) +=
      cnt_table.select<4, 1, TUPLE_SZ, 1>(4, 0);
  cnt_table.select<2, 1, TUPLE_SZ, 1>(0, 0) +=
      cnt_table.select<2, 1, TUPLE_SZ, 1>(2, 0);
  cnt_table.select<1, 1, TUPLE_SZ, 1>(0, 0) +=
      cnt_table.select<1, 1, TUPLE_SZ, 1>(1, 0);

  simd<unsigned, 8> voff(0, 1);        // 0, 1, 2, 3
  simd<ushort, 8> p = voff < TUPLE_SZ; // predicate
  voff = (voff + (global_offset + stride_threads * TUPLE_SZ - TUPLE_SZ)) *
         sizeof(unsigned);
  scatter<unsigned, 8>(buf, voff, S.select<8, 1>(0), p);
}

// final reduction. One thread to compute prefix all remaining entries
void cmk_acum_final(unsigned *buf, unsigned h_pos, unsigned int stride_elems,
                    unsigned remaining) {
  simd<unsigned, 32> elm32(0, 1);

  // element offsets for scattered read: [e0,e1,e2,...,e31] where e_i =
  // global_offset + # prefix_entries + prefix_entries - 1;
  simd<unsigned, 32> element_offset =
      (((elm32 + 1) * stride_elems - 1) * TUPLE_SZ) * sizeof(unsigned);

  simd<unsigned, 32 * TUPLE_SZ> S;
  simd<unsigned, TUPLE_SZ> prev = 0;
  for (unsigned i = 0; i < remaining; i += 32) {

    simd<ushort, 32> p = elm32 < remaining;

    S = gather_rgba<GATHER_SCATTER_MASK, unsigned int, 32>(buf, element_offset,
                                                           p);

    auto cnt_table = S.bit_cast_view<unsigned int, TUPLE_SZ, 32>();
    cnt_table.column(0) += prev;
#pragma unroll
    for (unsigned j = 0; j < TUPLE_SZ; j++) {
      // step 1
      cnt_table.select<1, 1, 16, 2>(j, 1) +=
          cnt_table.select<1, 1, 16, 2>(j, 0);
      // step 2
      cnt_table.select<1, 1, 8, 4>(j, 2) += cnt_table.select<1, 1, 8, 4>(j, 1);
      cnt_table.select<1, 1, 8, 4>(j, 3) += cnt_table.select<1, 1, 8, 4>(j, 1);
      // step 3
      cnt_table.select<1, 1, 4, 1>(j, 4) +=
          cnt_table.replicate_vs_w_hs<1, 0, 4, 0>(j, 3);
      cnt_table.select<1, 1, 4, 1>(j, 12) +=
          cnt_table.replicate_vs_w_hs<1, 0, 4, 0>(j, 11);
      cnt_table.select<1, 1, 4, 1>(j, 20) +=
          cnt_table.replicate_vs_w_hs<1, 0, 4, 0>(j, 19);
      cnt_table.select<1, 1, 4, 1>(j, 28) +=
          cnt_table.replicate_vs_w_hs<1, 0, 4, 0>(j, 27);
      // step 4
      cnt_table.select<1, 1, 8, 1>(j, 8) +=
          cnt_table.replicate_vs_w_hs<1, 0, 8, 0>(j, 7);
      cnt_table.select<1, 1, 8, 1>(j, 24) +=
          cnt_table.replicate_vs_w_hs<1, 0, 8, 0>(j, 23);
      // step 5
      cnt_table.select<1, 1, 16, 1>(j, 16) +=
          cnt_table.replicate_vs_w_hs<1, 0, 16, 0>(j, 15);
    }
    scatter_rgba<GATHER_SCATTER_MASK, unsigned int, 32>(buf, element_offset, S,
                                                        p);
    elm32 += 32;
    element_offset += stride_elems * TUPLE_SZ * sizeof(unsigned) * 32;
    prev = cnt_table.column(31);
  }
}

void cmk_prefix_iterative(unsigned *buf, unsigned h_pos,
                          unsigned int stride_elems, unsigned stride_thread,
                          unsigned n_entries) {
  simd<unsigned, 32> elm32(0, 1);

  unsigned global_offset = h_pos * stride_thread * TUPLE_SZ;

  // element offsets for scattered read: [e0,e1,e2,...,e31] where e_i =
  // global_offset + # prefix_entries + prefix_entries - 1;
  simd<unsigned, 32> element_offset =
      (((elm32 + 1) * stride_elems - 1) * TUPLE_SZ + global_offset) *
      sizeof(unsigned);

  // read the accumulated sum from its previous chunk
  simd<unsigned, TUPLE_SZ> prev = 0;
  if (h_pos == 0)
    prev = 0;
  else {
    // WA gather does not take less than 8
    // simd<unsigned, TUPLE_SZ> rd_off(0,1);  // 0, 1, 2, 3
    simd<unsigned, 8> rd_off(0, 1);
    rd_off += (global_offset - TUPLE_SZ);
    simd<unsigned, 8> temp;
    temp = gather<unsigned, 8>(buf, rd_off * sizeof(unsigned));
    prev = temp.select<TUPLE_SZ, 1>(0);
  }

  simd<unsigned, 32 * TUPLE_SZ> S;
  unsigned n_iter = n_entries / 32;
  for (unsigned i = 0; i < n_iter; i++) {

    S = gather_rgba<GATHER_SCATTER_MASK, unsigned int, 32>(buf, element_offset);

    auto cnt_table = S.bit_cast_view<unsigned int, TUPLE_SZ, 32>();
    cnt_table.column(0) += prev;
#pragma unroll
    for (unsigned j = 0; j < TUPLE_SZ; j++) {
      // step 1
      cnt_table.select<1, 1, 16, 2>(j, 1) +=
          cnt_table.select<1, 1, 16, 2>(j, 0);
      // step 2
      cnt_table.select<1, 1, 8, 4>(j, 2) += cnt_table.select<1, 1, 8, 4>(j, 1);
      cnt_table.select<1, 1, 8, 4>(j, 3) += cnt_table.select<1, 1, 8, 4>(j, 1);
      // step 3
      cnt_table.select<1, 1, 4, 1>(j, 4) +=
          cnt_table.replicate_vs_w_hs<1, 0, 4, 0>(j, 3);
      cnt_table.select<1, 1, 4, 1>(j, 12) +=
          cnt_table.replicate_vs_w_hs<1, 0, 4, 0>(j, 11);
      cnt_table.select<1, 1, 4, 1>(j, 20) +=
          cnt_table.replicate_vs_w_hs<1, 0, 4, 0>(j, 19);
      cnt_table.select<1, 1, 4, 1>(j, 28) +=
          cnt_table.replicate_vs_w_hs<1, 0, 4, 0>(j, 27);
      // step 4
      cnt_table.select<1, 1, 8, 1>(j, 8) +=
          cnt_table.replicate_vs_w_hs<1, 0, 8, 0>(j, 7);
      cnt_table.select<1, 1, 8, 1>(j, 24) +=
          cnt_table.replicate_vs_w_hs<1, 0, 8, 0>(j, 23);
      // step 5
      cnt_table.select<1, 1, 16, 1>(j, 16) +=
          cnt_table.replicate_vs_w_hs<1, 0, 16, 0>(j, 15);
    }

    // during reduction phase, we've already computed prefix sum and saved in
    // the last entry. Here we avoid double counting the last entry
    if (i == n_iter - 1)
      cnt_table.column(31) -= cnt_table.column(30);

    scatter_rgba<GATHER_SCATTER_MASK>(buf, element_offset, S);

    element_offset += stride_elems * TUPLE_SZ * sizeof(unsigned) * 32;
    prev = cnt_table.column(31);
  }
}

void hierarchical_prefix(queue &q, unsigned *buf, unsigned elem_stride,
                         unsigned thrd_stride, unsigned n_entries,
                         unsigned entry_per_th) {
  if (n_entries <= REMAINING_ENTRIES) {
    q.submit([&](handler &cgh) {
      cgh.parallel_for<class Accum_final>(
          range<2>{1, 1} * range<2>{1, 1}, [=](item<2> it) SYCL_ESIMD_KERNEL {
            cmk_acum_final(buf, it.get_id(0), elem_stride, n_entries);
          });
    });
    return;
  }

  if (entry_per_th == PREFIX_ENTRIES) {
    q.submit([&](handler &cgh) {
      cgh.parallel_for<class Accum_iterative1>(
          range<2>{n_entries / entry_per_th, 1} * range<2>{1, 1},
          [=](item<2> it) SYCL_ESIMD_KERNEL {
            cmk_acum_iterative(buf, it.get_id(0), elem_stride, thrd_stride,
                               PREFIX_ENTRIES);
          });
    });
  } else {
    q.submit([&](handler &cgh) {
      cgh.parallel_for<class Accum_iterative2>(
          range<2>{n_entries / entry_per_th, 1} * range<2>{1, 1},
          [=](item<2> it) SYCL_ESIMD_KERNEL {
            cmk_acum_iterative(buf, it.get_id(0), elem_stride, thrd_stride,
                               PREFIX_ENTRIES_LOW);
          });
    });
  }

  // if number of remaining entries <= 4K , each thread  accumulates smaller
  // number of entries to keep EUs saturated
  if (n_entries / entry_per_th > 4096)
    hierarchical_prefix(q, buf, thrd_stride, thrd_stride * PREFIX_ENTRIES,
                        n_entries / entry_per_th, PREFIX_ENTRIES);
  else
    hierarchical_prefix(q, buf, thrd_stride, thrd_stride * PREFIX_ENTRIES_LOW,
                        n_entries / entry_per_th, PREFIX_ENTRIES_LOW);

  if (entry_per_th == PREFIX_ENTRIES) {
    q.submit([&](handler &cgh) {
      cgh.parallel_for<class Prefix_iterative1>(
          range<2>{n_entries / entry_per_th, 1} * range<2>{1, 1},
          [=](item<2> it) SYCL_ESIMD_KERNEL {
            cmk_prefix_iterative(buf, it.get_id(0), elem_stride, thrd_stride,
                                 PREFIX_ENTRIES);
          });
    });
  } else {
    q.submit([&](handler &cgh) {
      cgh.parallel_for<class Prefix_iterative2>(
          range<2>{n_entries / entry_per_th, 1} * range<2>{1, 1},
          [=](item<2> it) SYCL_ESIMD_KERNEL {
            cmk_prefix_iterative(buf, it.get_id(0), elem_stride, thrd_stride,
                                 PREFIX_ENTRIES_LOW);
          });
    });
  }
}

//
// This kerenel compute histogram value. Each workitem of N_WI computes its own
// histograms each iteration N_RADIX_BITS bits are extracted. v is the value of
// the extracted bits. Inc corresponding histogram entry, histogram[v]++ The
// output hist, H(c, t, i), is laid out as SOA T: total HW threads t: thread id
// i: work item
// N_WI: number of WI per HW thread
// H(c, t, i) = hist[c*T*N_WI + t*N_WI + i]
// That is, histogram of Radix 0 are packed to the front
//
void cmk_radix_count(
    unsigned *in,         // input keys
    unsigned *hist,       // output of histogram
    unsigned h_pos,       // thread id
    unsigned w_radix_val, // number of elements of a radix value, i.e., T * N_WI
    unsigned bit_pos)     // process bits <bit_pos, bit_pos + N_RADIX_BITS -1>
{
  unsigned int mask = (RADIX - 1) << bit_pos;

  // each thread processes BASE_SZ elements
  // the offset of the data chunk processed by this thread
  unsigned int offset = (h_pos * N_ELEM);
  simd<unsigned, N_WI> init(0, 1); // 0, 1, 2, ..., 31

  // each WI process contiguous N_ELEM_WI elements
  simd<unsigned, N_WI> elem_offset =
      (init * N_ELEM_WI + offset) * sizeof(unsigned); // byte offset

  simd<unsigned, RADIX * N_WI> V = 0;
  auto counters = V.bit_cast_view<unsigned, RADIX, N_WI>();

  // each WI process N_ELEM_WI. each iteration reads in 4 elements (gather_rgba)
  for (int i = 0; i < N_ELEM_WI / 4; i++) {

    // use gather_rgba. Each address reads 4 channels
    simd<unsigned, 4 * N_WI> S;
    S = gather_rgba<rgba_channel_mask::ABGR, unsigned int, N_WI>(in,
                                                                 elem_offset);
    auto g4 = S.bit_cast_view<unsigned, 4, N_WI>();

#pragma unroll
    for (int j = 0; j < 4; j++) {
      simd<unsigned, N_WI> m = g4.row(j) & mask; // WA
      simd<unsigned, N_WI> val = m >> bit_pos;
      // simd<unsigned, N_WI> val = (g4.row(j) & mask) >> bit_pos;
#pragma unroll
      for (int k = 0; k < RADIX; k++) {
        counters.row(k).merge(counters.row(k) + 1, val == k);
      }
    }
    elem_offset += 4 * sizeof(unsigned); // byte offset
  }

  // write out counters
#pragma unroll
  for (int i = 0; i < RADIX; i++) {
    block_store<unsigned, N_WI>(hist + i * w_radix_val + h_pos * N_WI,
                                counters.row(i));
  }
}

void cmk_radix_reorder(
    unsigned *in,         // input that need to be reordered
    unsigned *out,        // reordered output
    unsigned *histogram,  // scanned histogram
    unsigned h_pos,       // thread id
    unsigned w_radix_val, // number of elements of a radix value, i.e., T * N_WI
    unsigned bit_pos)     // process bits <bit_pos, bit_pos + N_RADIX_BITS -1>
{

  simd<unsigned, RADIX * N_WI> T;
  auto H = T.bit_cast_view<unsigned, RADIX, N_WI>();
  //
  // read scanned hisogram data
  //
  unsigned offset = h_pos * N_WI;
#pragma unroll
  for (int i = 0; i < RADIX; i++) {
    H.row(i) = block_load<unsigned, N_WI>(histogram + i * w_radix_val + offset);
  }

  simd<unsigned, N_WI> init(0, 1); // 0, 1, 2, ...,

  unsigned int mask = (RADIX - 1) << bit_pos;

  // each thread processes BASE_SZ elements
  // the offset of the data chunk processed by this thread
  offset = (h_pos * N_ELEM);

  // each WI process contiguous N_ELEM_WI elements
  simd<unsigned, N_WI> elem_offset =
      (init * N_ELEM_WI + offset) * sizeof(unsigned); // byte offset

  // each WI process N_ELEM_WI. each iteration reads in 4 elements (gather_rgba)
  for (int i = 0; i < N_ELEM_WI / 4; i++) {
    // use gather_rgba. Each address reads 4 channels
    simd<unsigned, 4 * N_WI> S;
    S = gather_rgba<rgba_channel_mask::ABGR, unsigned int, N_WI>(in,
                                                                 elem_offset);
    auto g4 = S.bit_cast_view<unsigned, 4, N_WI>();

#pragma unroll
    for (int j = 0; j < 4; j++) {
      // each WI computes its radix value
      simd<unsigned, N_WI> r = g4.row(j);
      simd<unsigned, N_WI> m = (r & mask); // temp WA
      simd<unsigned, N_WI> val = m >> bit_pos;
      // simd<unsigned, N_WI> val = (g4.row(j) & mask) >> bit_pos;
      // add init {0, 1, 2, ...} so each WI access its corresponding histogram
      simd<unsigned, N_WI> idx = val * N_WI + init;
      simd<unsigned, N_WI> pos;
#pragma unroll
      for (unsigned k = 0; k < N_WI; k++) {
        // pos[k] = T[idx[k]]++;
        unsigned t = idx[k];
        unsigned p = T[t];
        T.select<1, 1>(t) = p + 1;
        pos.select<1, 1>(k) = p;
      }
      scatter<unsigned, N_WI>(out, pos * sizeof(unsigned), r);
    }
    elem_offset += 4 * sizeof(unsigned); // byte offset
  }
}

void dump_local_thread(unsigned *buf, unsigned local_id, unsigned bit_pos) {
  unsigned int mask = (RADIX - 1) << bit_pos;

  std::cout << "Local ID " << local_id << std::endl;
  for (int i = 0; i < N_WI; i++) {
    std::cout << "  WI " << i << ": ";
    for (int j = 0; j < N_ELEM / N_WI; j++) {
      std::cout << buf[local_id * N_ELEM + i * (N_ELEM / N_WI) + j] << "("
                << ((buf[local_id * N_ELEM + i * (N_ELEM / N_WI) + j] & mask) >>
                    bit_pos)
                << ") ";
      if (j > 0 && (j + 1) % 10 == 0)
        std::cout << " | ";
    }
    std::cout << std::endl;
  }
}

void radix_sort(queue &q, unsigned *in, unsigned *out, unsigned size) {
  auto dev = q.get_device();
  auto ctxt = q.get_context();
  //
  // each HW thread process N_ELEM elements
  // N_WI workitems generate histograms indepenently. Each WI generate RADIX
  // histograms
  //
  unsigned int hist_sz = size / N_ELEM * N_WI * RADIX;
  // we add 4 dwords (16 bytes) here
  // the prefix sum algorithm is inclusive. e.g, [1, 2, 2, 3] --> [1, 3, 5, 8]
  // Radix sort needs exclusive scan [1, 2, 2, 3] --> [0, 1, 3, 5]
  // we add padding in the beginning and initialize it to zero [ 0, 1, 2, 2, 3]
  // and then apply prefix scan to [1, 2, 2, 3] the end result is [0, 1, 3, 5,
  // 8]. Reordering pass take [0, 1, 3, 5] by addjusting the starting histogram
  // address We add 16-byte padding because block load requres oword (16-byte)
  // alignment
  unsigned int *histogram = static_cast<unsigned int *>(
      malloc_shared((hist_sz + 4) * sizeof(unsigned int), dev, ctxt));
  histogram[0] = histogram[1] = histogram[2] = histogram[3] =
      0; // actually only need to init histogram[3]

  // each iteration process N_RADIX_BITS; sorting 32 bits integer takes
  // 32/N_RADIX_BITS iterations
  for (int i = 0; i < 32 / N_RADIX_BITS; i++) {
    std::cout << "iteration " << i << std::endl;

    // init to zero for debugging. For final version, remove this
    // memset(out, -1, sizeof(unsigned) * size);  // set to zero for debugging
    // purpose

    // histogram + 4 to skip padding
    q.submit([&](handler &cgh) {
      cgh.parallel_for<class Radix_Count>(
          range<2>{size / N_ELEM, 1} * range<2>{1, 1},
          [=](item<2> it) SYCL_ESIMD_KERNEL {
            cmk_radix_count(in, histogram + 4, it.get_id(0),
                            size / N_ELEM * N_WI, i * N_RADIX_BITS);
          });
    });
    //
    // scan histogram and compute prefix sum
    //
    hierarchical_prefix(q, histogram + 4, 1, PREFIX_ENTRIES, hist_sz,
                        PREFIX_ENTRIES);
    //
    // reorder elements based on scanned histogram
    //
    // hitogram + 3 to get the exclusive prefix scan
    q.submit([&](handler &cgh) {
      cgh.parallel_for<class Radix_Reorder>(
          range<2>{size / N_ELEM, 1} * range<2>{1, 1},
          [=](item<2> it) SYCL_ESIMD_KERNEL {
            cmk_radix_reorder(in, out, histogram + 3, it.get_id(0),
                              size / N_ELEM * N_WI, i * N_RADIX_BITS);
          });
    });

    //
    // next iteration takes out buffer as input
    //
    std::swap(in, out);
  }
  q.wait();
  free(histogram, ctxt);
}

//************************************
// Demonstrate summation of arrays both in scalar on CPU and parallel on device
//************************************
int main(int argc, char *argv[]) {

  unsigned int *pInputs;
  unsigned log2_element = LOG2_ELEMENTS;
  unsigned int size = 1 << log2_element;

  cl::sycl::range<2> LocalRange{1, 1};

  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler(),
          property::queue::in_order());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto ctxt = q.get_context();

  // allocate and initialized input
  pInputs = static_cast<unsigned int *>(
      malloc_shared(size * sizeof(unsigned int), dev, ctxt));
  for (unsigned int i = 0; i < size; ++i) {
    unsigned x;
    x = rand();
    if (i % 3 == 0)
      x |= (rand() & 0xFF) << 16;
    if (i % 4 == 0)
      x |= (rand() & 0xFF) << 24;

    pInputs[i] = x;
  }
  // sorting takes multiple iteration. Each iteration takes output of its
  // previous iteation as input allocate a temp buf for alternating input and
  // output
  unsigned int *tmp_buf = static_cast<unsigned int *>(
      malloc_shared(size * sizeof(unsigned int), dev, ctxt));

  // allocate & compute expected result
  unsigned int *pExpectOutputs =
      static_cast<unsigned int *>(malloc(size * sizeof(unsigned int)));
  memcpy(pExpectOutputs, pInputs, sizeof(unsigned int) * size);
  std::sort(pExpectOutputs, pExpectOutputs + size);

  radix_sort(q, pInputs, tmp_buf, size);

  bool pass = memcmp(pInputs, pExpectOutputs, size * sizeof(unsigned int)) == 0;

  for (int i = 0; i < size; i++) {
    if (pInputs[i] != pExpectOutputs[i]) {
      std::cout << " pInputs[" << i << "] is " << pInputs[i] << " execpted "
                << pExpectOutputs[i] << std::endl;
      break;
    }
  }
  std::cout << "Radix Sort " << (pass ? "=> PASSED" : "=> FAILED") << std::endl
            << std::endl;

  free(pInputs, ctxt);
  free(tmp_buf, ctxt);
  free(pExpectOutputs);
  return 0;
}
