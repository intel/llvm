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
void bubble_sort(int32_t *first, const size_t beg, const size_t end) {
  if (beg < end) {
    for (size_t i = beg; i < end; ++i)
      for (size_t j = i + 1; j < end; ++j) {
        if (first[i] > first[j]) {
          first[i] = first[i] ^ first[j];
          first[j] = first[i] ^ first[j];
          first[i] = first[i] ^ first[j];
        }
      }
  }
}

void merge(int32_t *din, int32_t *dout, size_t widx, size_t msize,
           size_t chunks, size_t n) {
  if (2 * widx >= chunks)
    return;
  size_t beg1 = 2 * widx * msize;
  size_t end1 = beg1 + msize;
  size_t beg2, end2;
  if (end1 >= n) {
    end1 = beg2 = end2 = n;
  } else {
    beg2 = end1;
    end2 = beg2 + msize;
    if (end2 >= n)
      end2 = n;
  }
  size_t output_idx = 2 * widx * msize;
  while ((beg1 != end1) && (beg2 != end2)) {
    if (din[beg1] < din[beg2])
      dout[output_idx++] = din[beg1++];
    else
      dout[output_idx++] = din[beg2++];
  }

  while (beg1 != end1)
    dout[output_idx++] = din[beg1++];
  while (beg2 != end2)
    dout[output_idx++] = din[beg2++];
}

void merge_sort(int32_t *first, uint32_t n, uint8_t *scratch) {
  const size_t idx = __get_wg_local_linear_id();
  const size_t wg_size = __get_wg_local_range();
  const size_t chunk_size = (n - 1) / wg_size + 1;

  const size_t bubble_beg = (idx * chunk_size) >= n ? n : idx * chunk_size;
  const size_t bubble_end =
      ((idx + 1) * chunk_size) > n ? n : (idx + 1) * chunk_size;
  bubble_sort(first, bubble_beg, bubble_end);
  group_barrier();
  int32_t *scratch1 = reinterpret_cast<int32_t *>(scratch);
  bool data_in_scratch = false;
  // We have wg_size chunks here, each chunk has chunk_size elements which
  // are sorted. The last chunck's element number may be smaller.
  size_t chunks_to_merge = (n - 1) / chunk_size + 1;
  size_t merge_size = chunk_size;
  while (chunks_to_merge > 1) {
    // workitem 0 will merge chunk 0, 1.
    // workitem 1 will merge chunk 2, 3.
    // workitem idx will merge chunk 2 * idx and 2 * idx + 1
    int32_t *data_in = data_in_scratch ? scratch1 : first;
    int32_t *data_out = data_in_scratch ? first : scratch1;
    merge(data_in, data_out, idx, merge_size, chunks_to_merge, n);
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
#endif

#endif
