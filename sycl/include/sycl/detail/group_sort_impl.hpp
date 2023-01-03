//==------------ group_sort_impl.hpp ---------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file includes some functions for group sorting algorithm implementations
//

#pragma once

#include <sycl/detail/helpers.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/multi_ptr.hpp>

#ifdef __SYCL_DEVICE_ONLY__

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

// ---- merge sort implementation

// following two functions could be useless if std::[lower|upper]_bound worked
// well
template <typename Acc, typename Value, typename Compare>
size_t lower_bound(Acc acc, size_t first, size_t last, const Value &value,
                   Compare comp) {
  size_t n = last - first;
  size_t cur = n;
  size_t it;
  while (n > 0) {
    it = first;
    cur = n / 2;
    it += cur;
    if (comp(acc[it], value)) {
      n -= cur + 1, first = ++it;
    } else
      n = cur;
  }
  return first;
}

template <typename Acc, typename Value, typename Compare>
size_t upper_bound(Acc acc, const size_t first, const size_t last,
                   const Value &value, Compare comp) {
  return detail::lower_bound(acc, first, last, value,
                             [comp](auto x, auto y) { return !comp(y, x); });
}

// swap for all data types including tuple-like types
template <typename T> void swap_tuples(T &a, T &b) { std::swap(a, b); }

template <template <typename...> class TupleLike, typename T1, typename T2>
void swap_tuples(TupleLike<T1, T2> &&a, TupleLike<T1, T2> &&b) {
  std::swap(std::get<0>(a), std::get<0>(b));
  std::swap(std::get<1>(a), std::get<1>(b));
}

template <typename Iter> struct GetValueType {
  using type = typename std::iterator_traits<Iter>::value_type;
};

template <typename ElementType, access::address_space Space,
          access::decorated IsDecorated>
struct GetValueType<sycl::multi_ptr<ElementType, Space, IsDecorated>> {
  using type = ElementType;
};

// since we couldn't assign data to raw memory, it's better to use placement
// for first assignment
template <typename Acc, typename T>
void set_value(Acc ptr, const size_t idx, const T &val, bool is_first) {
  if (is_first) {
    ::new (ptr + idx) T(val);
  } else {
    ptr[idx] = val;
  }
}

template <typename InAcc, typename OutAcc, typename Compare>
void merge(const size_t offset, InAcc &in_acc1, OutAcc &out_acc1,
           const size_t start_1, const size_t end_1, const size_t end_2,
           const size_t start_out, Compare comp, const size_t chunk,
           bool is_first) {
  const size_t start_2 = end_1;
  // Borders of the sequences to merge within this call
  const size_t local_start_1 =
      sycl::min(static_cast<size_t>(offset + start_1), end_1);
  const size_t local_end_1 =
      sycl::min(static_cast<size_t>(local_start_1 + chunk), end_1);
  const size_t local_start_2 =
      sycl::min(static_cast<size_t>(offset + start_2), end_2);
  const size_t local_end_2 =
      sycl::min(static_cast<size_t>(local_start_2 + chunk), end_2);

  const size_t local_size_1 = local_end_1 - local_start_1;
  const size_t local_size_2 = local_end_2 - local_start_2;

  // TODO: process cases where all elements of 1st sequence > 2nd, 2nd > 1st
  // to improve performance

  // Process 1st sequence
  if (local_start_1 < local_end_1) {
    // Reduce the range for searching within the 2nd sequence and handle bound
    // items find left border in 2nd sequence
    const auto local_l_item_1 = in_acc1[local_start_1];
    size_t l_search_bound_2 =
        detail::lower_bound(in_acc1, start_2, end_2, local_l_item_1, comp);
    const size_t l_shift_1 = local_start_1 - start_1;
    const size_t l_shift_2 = l_search_bound_2 - start_2;

    set_value(out_acc1, start_out + l_shift_1 + l_shift_2, local_l_item_1,
              is_first);

    size_t r_search_bound_2{};
    // find right border in 2nd sequence
    if (local_size_1 > 1) {
      const auto local_r_item_1 = in_acc1[local_end_1 - 1];
      r_search_bound_2 = detail::lower_bound(in_acc1, l_search_bound_2, end_2,
                                             local_r_item_1, comp);
      const auto r_shift_1 = local_end_1 - 1 - start_1;
      const auto r_shift_2 = r_search_bound_2 - start_2;

      set_value(out_acc1, start_out + r_shift_1 + r_shift_2, local_r_item_1,
                is_first);
    }

    // Handle intermediate items
    for (size_t idx = local_start_1 + 1; idx < local_end_1 - 1; ++idx) {
      const auto intermediate_item_1 = in_acc1[idx];
      // we shouldn't seek in whole 2nd sequence. Just for the part where the
      // 1st sequence should be
      l_search_bound_2 =
          detail::lower_bound(in_acc1, l_search_bound_2, r_search_bound_2,
                              intermediate_item_1, comp);
      const size_t shift_1 = idx - start_1;
      const size_t shift_2 = l_search_bound_2 - start_2;

      set_value(out_acc1, start_out + shift_1 + shift_2, intermediate_item_1,
                is_first);
    }
  }
  // Process 2nd sequence
  if (local_start_2 < local_end_2) {
    // Reduce the range for searching within the 1st sequence and handle bound
    // items find left border in 1st sequence
    const auto local_l_item_2 = in_acc1[local_start_2];
    size_t l_search_bound_1 =
        detail::upper_bound(in_acc1, start_1, end_1, local_l_item_2, comp);
    const size_t l_shift_1 = l_search_bound_1 - start_1;
    const size_t l_shift_2 = local_start_2 - start_2;

    set_value(out_acc1, start_out + l_shift_1 + l_shift_2, local_l_item_2,
              is_first);

    size_t r_search_bound_1{};
    // find right border in 1st sequence
    if (local_size_2 > 1) {
      const auto local_r_item_2 = in_acc1[local_end_2 - 1];
      r_search_bound_1 = detail::upper_bound(in_acc1, l_search_bound_1, end_1,
                                             local_r_item_2, comp);
      const size_t r_shift_1 = r_search_bound_1 - start_1;
      const size_t r_shift_2 = local_end_2 - 1 - start_2;

      set_value(out_acc1, start_out + r_shift_1 + r_shift_2, local_r_item_2,
                is_first);
    }

    // Handle intermediate items
    for (auto idx = local_start_2 + 1; idx < local_end_2 - 1; ++idx) {
      const auto intermediate_item_2 = in_acc1[idx];
      // we shouldn't seek in whole 1st sequence. Just for the part where the
      // 2nd sequence should be
      l_search_bound_1 =
          detail::upper_bound(in_acc1, l_search_bound_1, r_search_bound_1,
                              intermediate_item_2, comp);
      const size_t shift_1 = l_search_bound_1 - start_1;
      const size_t shift_2 = idx - start_2;

      set_value(out_acc1, start_out + shift_1 + shift_2, intermediate_item_2,
                is_first);
    }
  }
}

template <typename Iter, typename Compare>
void bubble_sort(Iter first, const size_t begin, const size_t end,
                 Compare comp) {
  if (begin < end) {
    for (size_t i = begin; i < end; ++i) {
      // Handle intermediate items
      for (size_t idx = i + 1; idx < end; ++idx) {
        if (comp(first[idx], first[i])) {
          detail::swap_tuples(first[i], first[idx]);
        }
      }
    }
  }
}

template <typename Group, typename Iter, typename Compare>
void merge_sort(Group group, Iter first, const size_t n, Compare comp,
                std::byte *scratch) {
  using T = typename GetValueType<Iter>::type;
  const size_t idx = group.get_local_linear_id();
  const size_t local = group.get_local_range().size();
  const size_t chunk = (n - 1) / local + 1;

  // we need to sort within work item first
  bubble_sort(first, idx * chunk, sycl::min((idx + 1) * chunk, n), comp);
  sycl::group_barrier(group);

  T *temp = reinterpret_cast<T *>(scratch);
  bool data_in_temp = false;
  bool is_first = true;
  size_t sorted_size = 1;
  while (sorted_size * chunk < n) {
    const size_t start_1 =
        sycl::min(2 * sorted_size * chunk * (idx / sorted_size), n);
    const size_t end_1 = sycl::min(start_1 + sorted_size * chunk, n);
    const size_t end_2 = sycl::min(end_1 + sorted_size * chunk, n);
    const size_t offset = chunk * (idx % sorted_size);

    if (!data_in_temp) {
      merge(offset, first, temp, start_1, end_1, end_2, start_1, comp, chunk,
            is_first);
    } else {
      merge(offset, temp, first, start_1, end_1, end_2, start_1, comp, chunk,
            /*is_first*/ false);
    }
    sycl::group_barrier(group);

    data_in_temp = !data_in_temp;
    sorted_size *= 2;
    if (is_first)
      is_first = false;
  }

  // copy back if data is in a temporary storage
  if (data_in_temp) {
    for (size_t i = 0; i < chunk; ++i) {
      if (idx * chunk + i < n) {
        first[idx * chunk + i] = temp[idx * chunk + i];
      }
    }
    sycl::group_barrier(group);
  }
}

// traits for ascending functors
template <typename CompT> struct IsCompAscending {
  static constexpr bool value = false;
};
template <typename Type> struct IsCompAscending<std::less<Type>> {
  static constexpr bool value = true;
};

// get number of states radix bits can represent
constexpr uint32_t getStatesInBits(uint32_t radix_bits) {
  return (1 << radix_bits);
}

//------------------------------------------------------------------------
// Ordered traits for a given size and integral/float flag
//------------------------------------------------------------------------

template <size_t type_size, bool is_integral_type> struct GetOrdered {};

template <> struct GetOrdered<1, true> {
  using Type = uint8_t;
  constexpr static int8_t mask = 0x80;
};

template <> struct GetOrdered<2, true> {
  using Type = uint16_t;
  constexpr static int16_t mask = 0x8000;
};

template <> struct GetOrdered<4, true> {
  using Type = uint32_t;
  constexpr static int32_t mask = 0x80000000;
};

template <> struct GetOrdered<8, true> {
  using Type = uint64_t;
  constexpr static int64_t mask = 0x8000000000000000;
};

template <> struct GetOrdered<2, false> {
  using Type = uint16_t;
  constexpr static uint32_t nmask = 0xFFFF; // for negative numbers
  constexpr static uint32_t pmask = 0x8000; // for positive numbers
};

template <> struct GetOrdered<4, false> {
  using Type = uint32_t;
  constexpr static uint32_t nmask = 0xFFFFFFFF; // for negative numbers
  constexpr static uint32_t pmask = 0x80000000; // for positive numbers
};

template <> struct GetOrdered<8, false> {
  using Type = uint64_t;
  constexpr static uint64_t nmask = 0xFFFFFFFFFFFFFFFF; // for negative numbers
  constexpr static uint64_t pmask = 0x8000000000000000; // for positive numbers
};

//------------------------------------------------------------------------
// Ordered type for a given type
//------------------------------------------------------------------------

// for unknown/unsupported type we do not have any trait
template <typename ValT, typename Enabled = void> struct Ordered {};

// for unsigned integrals we use the same type
template <typename ValT>
struct Ordered<ValT, std::enable_if_t<std::is_integral<ValT>::value &&
                                      std::is_unsigned<ValT>::value>> {
  using Type = ValT;
};

// for signed integrals or floatings we map: size -> corresponding unsigned
// integral
template <typename ValT>
struct Ordered<
    ValT, std::enable_if_t<
              (std::is_integral<ValT>::value && std::is_signed<ValT>::value) ||
              std::is_floating_point<ValT>::value ||
              std::is_same<ValT, sycl::half>::value ||
              std::is_same<ValT, sycl::ext::oneapi::bfloat16>::value>> {
  using Type =
      typename GetOrdered<sizeof(ValT), std::is_integral<ValT>::value>::Type;
};

// shorthand
template <typename ValT> using OrderedT = typename Ordered<ValT>::Type;

//------------------------------------------------------------------------
// functions for conversion to Ordered type
//------------------------------------------------------------------------

// for already Ordered types (any uints) we use the same type
template <typename ValT>
std::enable_if_t<std::is_same_v<ValT, OrderedT<ValT>>, OrderedT<ValT>>
convertToOrdered(ValT value) {
  return value;
}

// converts integral type to Ordered (in terms of bitness) type
template <typename ValT>
std::enable_if_t<!std::is_same<ValT, OrderedT<ValT>>::value &&
                     std::is_integral<ValT>::value,
                 OrderedT<ValT>>
convertToOrdered(ValT value) {
  ValT result = value ^ GetOrdered<sizeof(ValT), true>::mask;
  return *reinterpret_cast<OrderedT<ValT> *>(&result);
}

// converts floating type to Ordered (in terms of bitness) type
template <typename ValT>
std::enable_if_t<!std::is_same<ValT, OrderedT<ValT>>::value &&
                     (std::is_floating_point<ValT>::value ||
                      std::is_same<ValT, sycl::half>::value ||
                      std::is_same<ValT, sycl::ext::oneapi::bfloat16>::value),
                 OrderedT<ValT>>
convertToOrdered(ValT value) {
  OrderedT<ValT> uvalue = *reinterpret_cast<OrderedT<ValT> *>(&value);
  // check if value negative
  OrderedT<ValT> is_negative = uvalue >> (sizeof(ValT) * CHAR_BIT - 1);
  // for positive: 00..00 -> 00..00 -> 10..00
  // for negative: 00..01 -> 11..11 -> 11..11
  OrderedT<ValT> ordered_mask =
      (is_negative * GetOrdered<sizeof(ValT), false>::nmask) |
      GetOrdered<sizeof(ValT), false>::pmask;
  return uvalue ^ ordered_mask;
}

//------------------------------------------------------------------------
// bit pattern functions
//------------------------------------------------------------------------

// required for descending comparator support
template <bool flag> struct InvertIf {
  template <typename ValT> ValT operator()(ValT value) { return value; }
};

// invert value if descending comparator is passed
template <> struct InvertIf<true> {
  template <typename ValT> ValT operator()(ValT value) { return ~value; }

  // invertation for bool type have to be logical, rather than bit
  bool operator()(bool value) { return !value; }
};

// get bit values in a certain bucket of a value
template <uint32_t radix_bits, bool is_comp_asc, typename ValT>
uint32_t getBucketValue(ValT value, uint32_t radix_iter) {
  // invert value if we need to sort in descending order
  value = InvertIf<!is_comp_asc>{}(value);

  // get bucket offset idx from the end of bit type (least significant bits)
  uint32_t bucket_offset = radix_iter * radix_bits;

  // get offset mask for one bucket, e.g.
  // radix_bits=2: 0000 0001 -> 0000 0100 -> 0000 0011
  OrderedT<ValT> bucket_mask = (1u << radix_bits) - 1u;

  // get bits under bucket mask
  return (value >> bucket_offset) & bucket_mask;
}
template <typename ValT> ValT getDefaultValue(bool is_comp_asc) {
  if (is_comp_asc)
    return (std::numeric_limits<ValT>::max)();
  else
    return std::numeric_limits<ValT>::lowest();
}

template <bool is_key_value_sort> struct ValuesAssigner {
  template <typename IterInT, typename IterOutT>
  void operator()(IterOutT output, size_t idx_out, IterInT input,
                  size_t idx_in) {
    output[idx_out] = input[idx_in];
  }

  template <typename IterOutT, typename ValT>
  void operator()(IterOutT output, size_t idx_out, ValT value) {
    output[idx_out] = value;
  }
};

template <> struct ValuesAssigner<false> {
  template <typename IterInT, typename IterOutT>
  void operator()(IterOutT, size_t, IterInT, size_t) {}

  template <typename IterOutT, typename ValT>
  void operator()(IterOutT, size_t, ValT) {}
};

// The iteration of radix sort for unknown number of elements per work item
template <uint32_t radix_bits, bool is_key_value_sort, bool is_comp_asc,
          typename KeysT, typename ValueT, typename GroupT>
void performRadixIterDynamicSize(GroupT group,
                                 const uint32_t items_per_work_item,
                                 const uint32_t radix_iter, const size_t n,
                                 KeysT *keys_input, ValueT *vals_input,
                                 KeysT *keys_output, ValueT *vals_output,
                                 uint32_t *memory) {
  const uint32_t radix_states = getStatesInBits(radix_bits);
  const size_t wgsize = group.get_local_linear_range();
  const size_t idx = group.get_local_linear_id();

  // 1.1. Zeroinitialize local memory
  uint32_t *scan_memory = reinterpret_cast<uint32_t *>(memory);
  for (uint32_t state = 0; state < radix_states; ++state)
    scan_memory[state * wgsize + idx] = 0;

  sycl::group_barrier(group);

  // 1.2. count values and write result to private count array and count memory
  for (uint32_t i = 0; i < items_per_work_item; ++i) {
    const uint32_t val_idx = items_per_work_item * idx + i;
    // get value, convert it to Ordered (in terms of bitness)
    const auto val =
        convertToOrdered((val_idx < n) ? keys_input[val_idx]
                                       : getDefaultValue<ValueT>(is_comp_asc));
    // get bit values in a certain bucket of a value
    const uint32_t bucket_val =
        getBucketValue<radix_bits, is_comp_asc>(val, radix_iter);

    // increment counter for this bit bucket
    if (val_idx < n)
      scan_memory[bucket_val * wgsize + idx]++;
  }

  sycl::group_barrier(group);

  // 2.1 Scan. Upsweep: reduce over radix states
  uint32_t reduced = 0;
  for (uint32_t i = 0; i < radix_states; ++i)
    reduced += scan_memory[idx * radix_states + i];

  // 2.2. Exclusive scan: over work items
  uint32_t scanned =
      sycl::exclusive_scan_over_group(group, reduced, std::plus<uint32_t>());

  // 2.3. Exclusive downsweep: exclusive scan over radix states
  for (uint32_t i = 0; i < radix_states; ++i) {
    uint32_t value = scan_memory[idx * radix_states + i];
    scan_memory[idx * radix_states + i] = scanned;
    scanned += value;
  }

  sycl::group_barrier(group);

  uint32_t private_scan_memory[radix_states] = {0};

  // 3. Reorder
  for (uint32_t i = 0; i < items_per_work_item; ++i) {
    const uint32_t val_idx = items_per_work_item * idx + i;
    // get value, convert it to Ordered (in terms of bitness)
    auto val =
        convertToOrdered((val_idx < n) ? keys_input[val_idx]
                                       : getDefaultValue<ValueT>(is_comp_asc));
    // get bit values in a certain bucket of a value
    uint32_t bucket_val =
        getBucketValue<radix_bits, is_comp_asc>(val, radix_iter);

    uint32_t new_offset_idx = private_scan_memory[bucket_val]++ +
                              scan_memory[bucket_val * wgsize + idx];
    if (val_idx < n) {
      keys_output[new_offset_idx] = keys_input[val_idx];
      ValuesAssigner<is_key_value_sort>()(vals_output, new_offset_idx,
                                          vals_input, val_idx);
    }
  }
}

// The iteration of radix sort for known number of elements per work item
template <size_t items_per_work_item, uint32_t radix_bits, bool is_comp_asc,
          bool is_key_value_sort, bool is_blocked, typename KeysT,
          typename ValsT, typename GroupT>
void performRadixIterStaticSize(GroupT group, const uint32_t radix_iter,
                                const uint32_t last_iter, KeysT *keys,
                                ValsT vals, std::byte *memory) {
  const uint32_t radix_states = getStatesInBits(radix_bits);
  const size_t wgsize = group.get_local_linear_range();
  const size_t idx = group.get_local_linear_id();

  // 1.1. count per witem: create a private array for storing count values
  uint32_t count_arr[items_per_work_item] = {0};
  uint32_t ranks[items_per_work_item] = {0};

  // 1.1. Zeroinitialize local memory
  uint32_t *scan_memory = reinterpret_cast<uint32_t *>(memory);
  for (uint32_t i = 0; i < radix_states; ++i)
    scan_memory[i * wgsize + idx] = 0;

  sycl::group_barrier(group);

  uint32_t *pointers[items_per_work_item] = {nullptr};
  // 1.2. count values and write result to private count array
  for (uint32_t i = 0; i < items_per_work_item; ++i) {
    // get value, convert it to Ordered (in terms of bitness)
    OrderedT<KeysT> val = convertToOrdered(keys[i]);
    // get bit values in a certain bucket of a value
    uint32_t bucket_val =
        getBucketValue<radix_bits, is_comp_asc>(val, radix_iter);
    pointers[i] = scan_memory + (bucket_val * wgsize + idx);
    count_arr[i] = (*pointers[i])++;
  }
  sycl::group_barrier(group);

  // 2.1 Scan. Upsweep: reduce over radix states
  uint32_t reduced = 0;
  for (uint32_t i = 0; i < radix_states; ++i)
    reduced += scan_memory[idx * radix_states + i];

  // 2.2. Exclusive scan: over work items
  uint32_t scanned =
      sycl::exclusive_scan_over_group(group, reduced, std::plus<uint32_t>());

  // 2.3. Exclusive downsweep: exclusive scan over radix states
  for (uint32_t i = 0; i < radix_states; ++i) {
    uint32_t value = scan_memory[idx * radix_states + i];
    scan_memory[idx * radix_states + i] = scanned;
    scanned += value;
  }

  sycl::group_barrier(group);

  // 2.4. Fill ranks with offsets
  for (uint32_t i = 0; i < items_per_work_item; ++i)
    ranks[i] = count_arr[i] + *pointers[i];

  sycl::group_barrier(group);

  // 3. Reorder
  KeysT *keys_temp = reinterpret_cast<KeysT *>(memory);
  ValsT *vals_temp = reinterpret_cast<ValsT *>(
      memory + wgsize * items_per_work_item * sizeof(KeysT));
  for (uint32_t i = 0; i < items_per_work_item; ++i) {
    keys_temp[ranks[i]] = keys[i];
    ValuesAssigner<is_key_value_sort>()(vals_temp, ranks[i], vals, i);
  }

  sycl::group_barrier(group);

  // 4. Copy back to input
  for (uint32_t i = 0; i < items_per_work_item; ++i) {
    size_t shift = idx * items_per_work_item + i;
    if constexpr (!is_blocked) {
      if (radix_iter == last_iter - 1)
        shift = i * wgsize + idx;
    }
    keys[i] = keys_temp[shift];
    ValuesAssigner<is_key_value_sort>()(vals, i, vals_temp, shift);
  }
}

template <bool is_key_value_sort, bool is_comp_asc,
          uint32_t items_per_work_item = 1, uint32_t radix_bits = 4,
          typename GroupT, typename KeysT, typename ValsT>
void privateDynamicSort(GroupT group, KeysT *keys, ValsT *values,
                        const size_t n, std::byte *scratch,
                        const uint32_t first_bit, const uint32_t last_bit) {
  const size_t wgsize = group.get_local_linear_range();
  constexpr uint32_t radix_states = getStatesInBits(radix_bits);
  const uint32_t first_iter = first_bit / radix_bits;
  const uint32_t last_iter = last_bit / radix_bits;

  KeysT *keys_input = keys;
  ValsT *vals_input = values;
  const uint32_t runtime_items_per_work_item = (n - 1) / wgsize + 1;

  // set pointers to unaligned memory
  uint32_t *scan_memory = reinterpret_cast<uint32_t *>(scratch);
  KeysT *keys_output = reinterpret_cast<KeysT *>(
      scratch + radix_states * wgsize * sizeof(uint32_t));
  // Adding 4 bytes extra space for keys due to specifics of some hardware
  // architectures.
  ValsT *vals_output = reinterpret_cast<ValsT *>(
      keys_output + is_key_value_sort * n * sizeof(KeysT) + alignof(uint32_t));

  for (uint32_t radix_iter = first_iter; radix_iter < last_iter; ++radix_iter) {
    performRadixIterDynamicSize<radix_bits, is_key_value_sort, is_comp_asc>(
        group, runtime_items_per_work_item, radix_iter, n, keys_input,
        vals_input, keys_output, vals_output, scan_memory);

    sycl::group_barrier(group);

    std::swap(keys_input, keys_output);
    std::swap(vals_input, vals_output);
  }
}

template <bool is_key_value_sort, bool is_blocked, bool is_comp_asc,
          size_t items_per_work_item = 1, uint32_t radix_bits = 4,
          typename GroupT, typename T, typename U>
void privateStaticSort(GroupT group, T *keys, U *values, std::byte *scratch,
                       const uint32_t first_bit, const uint32_t last_bit) {

  const uint32_t first_iter = first_bit / radix_bits;
  const uint32_t last_iter = last_bit / radix_bits;

  for (uint32_t radix_iter = first_iter; radix_iter < last_iter; ++radix_iter) {
    performRadixIterStaticSize<items_per_work_item, radix_bits, is_comp_asc,
                               is_key_value_sort, is_blocked>(
        group, radix_iter, last_iter, keys, values, scratch);
    sycl::group_barrier(group);
  }
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
#endif
