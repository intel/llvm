//==------- sort_helper.hpp - helper functions to do group sorting----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//

#pragma once
#include "group_helper.hpp"
#include <cstdint>

#if defined(__SPIR__) || defined(__SPIRV__)
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

static void __get_chunk_size(size_t group_id, size_t group_size, size_t n,
                             size_t *beg, size_t *end) {
  size_t tmp = n % group_size;
  size_t chunk_size = n / group_size;
  if (tmp) {
    if (group_id < tmp) {
      *beg = group_id * (chunk_size + 1);
      *end = *beg + chunk_size + 1;
    } else {
      *beg = tmp * (chunk_size + 1) + (group_id - tmp) * chunk_size;
      *end = *beg + chunk_size;
    }
  } else {
    *beg = group_id * chunk_size;
    *end = *beg + chunk_size;
  }
}

template <typename KeyT, typename ValT, typename Compare>
void merge_key_value(KeyT *keys_in, KeyT *keys_out, ValT *vals_in,
                     ValT *vals_out, size_t widx, size_t iter_num,
                     size_t chunks_to_merge, Compare comp) {
  if (2 * widx >= chunks_to_merge)
    return;

  //
}

template <typename KeyT, typename ValT, typename Compare>
void bubble_sort_key_value(KeyT *keys, ValT *vals, const size_t beg,
                           const size_t end, Compare comp) {
  if (beg < end) {
    KeyT temp_key;
    ValT temp_val;
    for (size_t i = beg; i < end; ++i)
      for (size_t j = i + 1; j < end; ++j) {
        if (!comp(keys[i], keys[j])) {
          temp_key = keys[i];
          keys[i] = keys[j];
          keys[j] = temp_key;
          temp_val = vals[i];
          vals[i] = vals[j];
          vals[j] = temp_val;
        }
      }
  }
}

// We have following assumption for scratch memory size for key-value
// group sort: size of scratch > (sizeof(KeyT) + sizeof(ValT)) +
// max(alignof(KeyT), alignof(ValT)).
template <typename KeyT, typename ValT, typename Compare>
void merge_sort_key_value(KeyT *keys, ValT *vals, size_t n, uint8_t *scratch,
                          Compare comp) {
  const size_t idx = __get_wg_local_linear_id();
  const size_t wg_size = __get_wg_local_range();
  const size_t bubble_beg, bubble_end;
  __get_chunk_size(idx, wg_size, n, &bubble_beg, &bubble_end);
  bubble_sort(keys, vals, bubble_beg, bubble_end, comp);
  group_barrier();
  bool data_in_scratch = false;
  KeyT *scratch_keys = reinterpret_cast<KeyT *>(scratch);
  uint8_t *val_offset = scratch + sizeof(KeyT) * (n + 1);
  val_offset += alignof(ValT) - val_offset % alignof(ValT);
  ValT *scratch_vals = reinterpret_cast<ValT *>(val_offset);
  // If n > work_group_size, each work item holds sorted elements to be merged.
  // Otherwise, only n work items hold 1 element. Chunk size <= work group size.
  size_t chunks_to_merge = (n > wg_size) ? wg_size : n;
  size_t iter_num = 0;
  while (chunks_to_merge > 1) {
    // workitem 0 will merge chunk 0, 1.
    // workitem 1 will merge chunk 2, 3.
    // workitem idx will merge chunk 2 * idx and 2 * idx + 1
    KeyT *keys_in = data_in_scratch ? scratch_keys : keys;
    KeyT *keys_out = data_in_scratch ? keys : scratch_keys;
    ValT *vals_in = data_in_scratch ? scratch_vals : vals;
    ValT *vals_out = data_in_scratch ? vals : scratch_vals;
    merge_key_value<KeyT, ValT, Compare>(keys_in, keys_out, vals_in, vals_out,
                                         idx, iter_num, chunks_to_merge, comp);
    // merge<Tp, Compare>(data_in, data_out, idx, merge_size, chunks_to_merge,
    // n,
    //                   comp);
    group_barrier();
    chunks_to_merge = (chunks_to_merge - 1) / 2 + 1;
    data_in_scratch = !data_in_scratch;
  }
}

#endif // __SPIR__ || __SPIRV__
