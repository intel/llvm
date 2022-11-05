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

#if __cplusplus >= 201703L
#include <sycl/detail/helpers.hpp>

#ifdef __SYCL_DEVICE_ONLY__

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

// ---- merge sort implementation

// following two functions could be useless if std::[lower|upper]_bound worked
// well
template <typename Acc, typename Value, typename Compare>
std::size_t lower_bound(Acc acc, std::size_t first, std::size_t last,
                        const Value &value, Compare comp) {
  std::size_t n = last - first;
  std::size_t cur = n;
  std::size_t it;
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
std::size_t upper_bound(Acc acc, const std::size_t first,
                        const std::size_t last, const Value &value,
                        Compare comp) {
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
void set_value(Acc ptr, const std::size_t idx, const T &val, bool is_first) {
  if (is_first) {
    ::new (ptr + idx) T(val);
  } else {
    ptr[idx] = val;
  }
}

template <typename InAcc, typename OutAcc, typename Compare>
void merge(const std::size_t offset, InAcc &in_acc1, OutAcc &out_acc1,
           const std::size_t start_1, const std::size_t end_1,
           const std::size_t end_2, const std::size_t start_out, Compare comp,
           const std::size_t chunk, bool is_first) {
  const std::size_t start_2 = end_1;
  // Borders of the sequences to merge within this call
  const std::size_t local_start_1 =
      sycl::min(static_cast<std::size_t>(offset + start_1), end_1);
  const std::size_t local_end_1 =
      sycl::min(static_cast<std::size_t>(local_start_1 + chunk), end_1);
  const std::size_t local_start_2 =
      sycl::min(static_cast<std::size_t>(offset + start_2), end_2);
  const std::size_t local_end_2 =
      sycl::min(static_cast<std::size_t>(local_start_2 + chunk), end_2);

  const std::size_t local_size_1 = local_end_1 - local_start_1;
  const std::size_t local_size_2 = local_end_2 - local_start_2;

  // TODO: process cases where all elements of 1st sequence > 2nd, 2nd > 1st
  // to improve performance

  // Process 1st sequence
  if (local_start_1 < local_end_1) {
    // Reduce the range for searching within the 2nd sequence and handle bound
    // items find left border in 2nd sequence
    const auto local_l_item_1 = in_acc1[local_start_1];
    std::size_t l_search_bound_2 =
        detail::lower_bound(in_acc1, start_2, end_2, local_l_item_1, comp);
    const std::size_t l_shift_1 = local_start_1 - start_1;
    const std::size_t l_shift_2 = l_search_bound_2 - start_2;

    set_value(out_acc1, start_out + l_shift_1 + l_shift_2, local_l_item_1,
              is_first);

    std::size_t r_search_bound_2{};
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
    for (std::size_t idx = local_start_1 + 1; idx < local_end_1 - 1; ++idx) {
      const auto intermediate_item_1 = in_acc1[idx];
      // we shouldn't seek in whole 2nd sequence. Just for the part where the
      // 1st sequence should be
      l_search_bound_2 =
          detail::lower_bound(in_acc1, l_search_bound_2, r_search_bound_2,
                              intermediate_item_1, comp);
      const std::size_t shift_1 = idx - start_1;
      const std::size_t shift_2 = l_search_bound_2 - start_2;

      set_value(out_acc1, start_out + shift_1 + shift_2, intermediate_item_1,
                is_first);
    }
  }
  // Process 2nd sequence
  if (local_start_2 < local_end_2) {
    // Reduce the range for searching within the 1st sequence and handle bound
    // items find left border in 1st sequence
    const auto local_l_item_2 = in_acc1[local_start_2];
    std::size_t l_search_bound_1 =
        detail::upper_bound(in_acc1, start_1, end_1, local_l_item_2, comp);
    const std::size_t l_shift_1 = l_search_bound_1 - start_1;
    const std::size_t l_shift_2 = local_start_2 - start_2;

    set_value(out_acc1, start_out + l_shift_1 + l_shift_2, local_l_item_2,
              is_first);

    std::size_t r_search_bound_1{};
    // find right border in 1st sequence
    if (local_size_2 > 1) {
      const auto local_r_item_2 = in_acc1[local_end_2 - 1];
      r_search_bound_1 = detail::upper_bound(in_acc1, l_search_bound_1, end_1,
                                             local_r_item_2, comp);
      const std::size_t r_shift_1 = r_search_bound_1 - start_1;
      const std::size_t r_shift_2 = local_end_2 - 1 - start_2;

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
      const std::size_t shift_1 = l_search_bound_1 - start_1;
      const std::size_t shift_2 = idx - start_2;

      set_value(out_acc1, start_out + shift_1 + shift_2, intermediate_item_2,
                is_first);
    }
  }
}

template <typename Iter, typename Compare>
void bubble_sort(Iter first, const std::size_t begin, const std::size_t end,
                 Compare comp) {
  if (begin < end) {
    for (std::size_t i = begin; i < end; ++i) {
      // Handle intermediate items
      for (std::size_t idx = i + 1; idx < end; ++idx) {
        if (comp(first[idx], first[i])) {
          detail::swap_tuples(first[i], first[idx]);
        }
      }
    }
  }
}

template <typename Group, typename Iter, typename Compare>
void merge_sort(Group group, Iter first, const std::size_t n, Compare comp,
                std::byte *scratch) {
  using T = typename GetValueType<Iter>::type;
  auto id = sycl::detail::Builder::getNDItem<Group::dimensions>();
  const std::size_t idx = id.get_local_linear_id();
  const std::size_t local = group.get_local_range().size();
  const std::size_t chunk = (n - 1) / local + 1;

  // we need to sort within work item first
  bubble_sort(first, idx * chunk, sycl::min((idx + 1) * chunk, n), comp);
  id.barrier();

  T *temp = reinterpret_cast<T *>(scratch);
  bool data_in_temp = false;
  bool is_first = true;
  std::size_t sorted_size = 1;
  while (sorted_size * chunk < n) {
    const std::size_t start_1 =
        sycl::min(2 * sorted_size * chunk * (idx / sorted_size), n);
    const std::size_t end_1 = sycl::min(start_1 + sorted_size * chunk, n);
    const std::size_t end_2 = sycl::min(end_1 + sorted_size * chunk, n);
    const std::size_t offset = chunk * (idx % sorted_size);

    if (!data_in_temp) {
      merge(offset, first, temp, start_1, end_1, end_2, start_1, comp, chunk,
            is_first);
    } else {
      merge(offset, temp, first, start_1, end_1, end_2, start_1, comp, chunk,
            /*is_first*/ false);
    }
    id.barrier();

    data_in_temp = !data_in_temp;
    sorted_size *= 2;
    if (is_first)
      is_first = false;
  }

  // copy back if data is in a temporary storage
  if (data_in_temp) {
    for (std::size_t i = 0; i < chunk; ++i) {
      if (idx * chunk + i < n) {
        first[idx * chunk + i] = temp[idx * chunk + i];
      }
    }
    id.barrier();
  }
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
#endif
#endif // __cplusplus >=201703L
