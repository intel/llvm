//==------- sort_helper.hpp - helper functions to do group sorting----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//

#ifndef __LIBDEVICE_SORT_H__
#define __LIBDEVICE_SORT_H__
#include "group_helper.hpp"
#include <cstdint>

#if defined(__SPIR__)
// A simple compare function for fp16 type emulated with uint16_t
// 1:   great than
// 0:   equal to
// -1:  less than
// -2:  can't compare(NAN)
int fp16_comp(uint16_t a, uint16_t b) {
  uint16_t a_sig = a >> 15;
  uint16_t a_exp = (a & 0x7fff) >> 10;
  uint16_t a_fra = a & 0x3ff;
  uint16_t b_sig = b >> 15;
  uint16_t b_exp = (b & 0x7fff) >> 10;
  uint16_t b_fra = b & 0x3ff;
  if (((a_exp == 0x1f) && (a_fra != 0x0)) || ((b_exp == 0x1f) && (b_fra != 0x0)))
    return -2;

  if ((a_sig == 0) && (b_sig == 1))
    return 1;

  if ((a_sig == 1) && (b_sig == 0))
    return -1;

  if (a_exp > b_exp)
    return (a_sig == 0) ? 1 : -1;

  if (a_exp < b_exp)
    return (a_sig == 0) ? -1 : 1;

  if (a_fra == b_fra)
    return 0;

  if (a_sig == 0)
    return (a_fra > b_fra) ? 1 : -1;
  else
    return (a_fra > b_fra) ? -1 : 1;
}

template <typename Tp, typename Compare>
void bubble_sort(Tp *first, const size_t beg, const size_t end, Compare comp) {
  if (beg < end) {
    Tp temp;
    for (size_t i = beg; i < end; ++i)
      for (size_t j = i + 1; j < end; ++j) {
        if (!comp(first[i], first[j])) {
          temp = first[i];
          first[i] = first[j];
          first[j] = temp;
        }
      }
  }
}

// widx: work-item id with a work-group
// chunks: number of sorted chunks waiting to be merged
// n: total number of elements waiting to be sorted
// msize: number of elements in a chunk ready to be merged
template <typename Tp, typename Compare>
void merge(Tp *din, Tp *dout, size_t widx, size_t msize, size_t chunks,
           size_t n, Compare comp) {
  if (2 * widx >= chunks)
    return;

  size_t beg1 = 2 * widx * msize;
  size_t end1 = beg1 + msize;
  size_t beg2, end2;
  if (end1 >= n)
    end1 = beg2 = end2 = n;
  else {
    beg2 = end1;
    end2 = beg2 + msize;
    if (end2 >= n)
      end2 = n;
  }

  size_t output_idx = 2 * widx * msize;
  while ((beg1 != end1) && (beg2 != end2)) {
    if (comp(din[beg1], din[beg2]))
      dout[output_idx++] = din[beg1++];
    else
      dout[output_idx++] = din[beg2++];
  }

  while (beg1 != end1)
    dout[output_idx++] = din[beg1++];
  while (beg2 != end2)
    dout[output_idx++] = din[beg2++];
}

template <typename Tp, typename Compare>
void merge_sort(Tp *first, uint32_t n, uint8_t *scratch, Compare comp) {
  const size_t idx = __get_wg_local_linear_id();
  const size_t wg_size = __get_wg_local_range();
  const size_t chunk_size = (n - 1) / wg_size + 1;

  const size_t bubble_beg = (idx * chunk_size) >= n ? n : idx * chunk_size;
  const size_t bubble_end =
      ((idx + 1) * chunk_size) > n ? n : (idx + 1) * chunk_size;
  bubble_sort(first, bubble_beg, bubble_end, comp);
  group_barrier();
  Tp *scratch1 = reinterpret_cast<Tp *>(scratch);
  bool data_in_scratch = false;
  // We have wg_size chunks here, each chunk has chunk_size elements which
  // are sorted. The last chunck's element number may be smaller.
  size_t chunks_to_merge = (n - 1) / chunk_size + 1;
  size_t merge_size = chunk_size;
  while (chunks_to_merge > 1) {
    // workitem 0 will merge chunk 0, 1.
    // workitem 1 will merge chunk 2, 3.
    // workitem idx will merge chunk 2 * idx and 2 * idx + 1
    Tp *data_in = data_in_scratch ? scratch1 : first;
    Tp *data_out = data_in_scratch ? first : scratch1;
    merge<Tp, Compare>(data_in, data_out, idx, merge_size, chunks_to_merge, n,
                       comp);
    group_barrier();
    chunks_to_merge = (chunks_to_merge - 1) / 2 + 1;
    merge_size <<= 1;
    data_in_scratch = !data_in_scratch;
  }
  if (data_in_scratch) {
    for (size_t i = idx * chunk_size; i < bubble_end; ++i)
      first[i] = scratch1[i];
    group_barrier();
  }
}

// Each work-item holds some input elements located in private memory and apply
// group sorting to all work-items' input. The sorted data will be copied back
// to each work-item's private memory.
// Assumption about scratch memory size:
// scratch_size >= n * wg_size * sizeof(Tp) * 2
template <typename Tp, typename Compare>
void private_merge_sort_close(Tp *first, uint32_t n, uint8_t *scratch,
                              Compare comp) {
  const size_t idx = __get_wg_local_linear_id();
  const size_t wg_size = __get_wg_local_range();
  Tp *temp_buffer = reinterpret_cast<Tp *>(scratch);
  for (size_t i = 0; i < n; ++i)
    temp_buffer[idx * n + i] = first[i];

  group_barrier();
  // do group sorting for whole input data
  merge_sort(temp_buffer, n * wg_size,
             reinterpret_cast<uint8_t *>(temp_buffer + n * wg_size), comp);

  for (size_t i = 0; i < n; ++i)
    first[i] = temp_buffer[idx * n + i];
}

template <typename Tp, typename Compare>
void private_merge_sort_spread(Tp *first, uint32_t n, uint8_t *scratch,
                               Compare comp) {
  const size_t idx = __get_wg_local_linear_id();
  const size_t wg_size = __get_wg_local_range();
  Tp *temp_buffer = reinterpret_cast<Tp *>(scratch);
  for (size_t i = 0; i < n; ++i)
    temp_buffer[idx * n + i] = first[i];

  group_barrier();
  // do group sorting for whole input data
  merge_sort(temp_buffer, n * wg_size,
             reinterpret_cast<uint8_t *>(temp_buffer + n * wg_size), comp);

  for (size_t i = 0; i < n; ++i)
    first[i] = temp_buffer[i * wg_size + idx];
}

// sub group sort implementation, each work-item holds an element, the total
// number of input elements is work group size.
// Assumption about scratch memory size:
// scratch_size >= wg_size * sizeof(Tp) * 2
template <typename Tp, typename Compare>
Tp sub_group_merge_sort(Tp value, uint8_t *scratch, Compare comp) {
  const size_t idx = __get_wg_local_linear_id();
  const size_t wg_size = __get_wg_local_range();
  Tp *temp_buffer = reinterpret_cast<Tp *>(scratch);
  temp_buffer[idx] = value;

  group_barrier();
  merge_sort(temp_buffer, wg_size,
             reinterpret_cast<uint8_t *>(temp_buffer + wg_size), comp);
  return temp_buffer[idx];
}

#endif

#endif
