//==------- group_helpers_sorters.hpp - SYCL sorters and group helpers -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
#include <sycl/detail/group_sort_impl.hpp>
#include <sycl/ext/oneapi/experimental/builtins.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::experimental {

namespace detail {

// TODO: Convert to SYCL properties
struct is_blocked {};
struct is_striped {};

} // namespace detail

// ---- group helpers
template <typename Group, size_t Extent> class group_with_scratchpad {
  Group g;
  sycl::span<std::byte, Extent> scratch;

public:
  group_with_scratchpad(Group g_, sycl::span<std::byte, Extent> scratch_)
      : g(g_), scratch(scratch_) {}
  Group get_group() const { return g; }
  sycl::span<std::byte, Extent> get_memory() const { return scratch; }
};

// ---- sorters
namespace default_sorters {

template <typename CompareT = std::less<>> class joint_sorter {
  CompareT comp;
  std::byte *scratch;
  size_t scratch_size;

public:
  template <size_t Extent>
  joint_sorter(sycl::span<std::byte, Extent> scratch_,
               CompareT comp_ = CompareT())
      : comp(comp_), scratch(scratch_.data()), scratch_size(scratch_.size()) {}

  template <typename Group, typename Ptr>
  void operator()(Group g, Ptr first, Ptr last) {
    (void)g;
    (void)first;
    (void)last;
#ifdef __SYCL_DEVICE_ONLY__
    using T = typename sycl::detail::GetValueType<Ptr>::type;
    if (scratch_size >= memory_required<T>(Group::fence_scope, last - first))
      sycl::detail::merge_sort(g, first, last - first, comp, scratch);
#else
    throw sycl::exception(
        std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
        "default_sorters::joint_sorter constructor is not supported on host "
        "device.");
#endif
  }

  template <typename Group, typename T> T operator()(Group g, T val) {
#ifdef __SYCL_DEVICE_ONLY__
    auto range_size = g.get_local_range().size();
    if (scratch_size >= memory_required<T>(Group::fence_scope, range_size)) {
      size_t local_id = g.get_local_linear_id();
      T *temp = reinterpret_cast<T *>(scratch);
      ::new (temp + local_id) T(val);
      sycl::detail::merge_sort(g, temp, range_size, comp,
                               scratch + range_size * sizeof(T));
      val = temp[local_id];
    }
#else
    (void)g;
    throw sycl::exception(
        std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
        "default_sorter operator() is not supported on host device.");
#endif
    return val;
  }

  template <typename T>
  static constexpr size_t memory_required(sycl::memory_scope,
                                          size_t range_size) {
    return range_size * sizeof(T) + alignof(T);
  }
};

template <typename T, std::size_t ElementsPerWorkItem = 1,
          typename CompareT = std::less<>>
class group_sorter {
  CompareT comp;
  std::byte *scratch;
  std::size_t scratch_size;

public:
  template <std::size_t Extent>
  group_sorter(sycl::span<std::byte, Extent> scratch_,
               CompareT comp_ = CompareT{})
      : comp(comp_), scratch(scratch_.data()), scratch_size(scratch_.size()) {}

  template <typename Group> T operator()(Group g, T val) {
    (void)g;
    (void)val;
#ifdef __SYCL_DEVICE_ONLY__
    auto range_size = g.get_local_range().size();
    if (scratch_size >= memory_required(Group::fence_scope, range_size)) {
      std::size_t local_id = g.get_local_linear_id();
      T *temp = reinterpret_cast<T *>(scratch);
      ::new (temp + local_id) T(val);
      sycl::detail::merge_sort(g, temp, range_size, comp,
                               scratch + range_size * sizeof(T));
      val = temp[local_id];
    }
    return val;
#else
    throw sycl::exception(
        std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
        "default_sorter operator() is not supported on host device.");
#endif
  }

  // TODO: Add a check for the property type
  template <typename Group, typename Properties>
  void operator()(Group g, sycl::span<T, ElementsPerWorkItem> values,
                  Properties properties) {
#ifdef __SYCL_DEVICE_ONLY__
    auto range_size = g.get_local_linear_range();
    if (scratch_size >=
        memory_required(Group::fence_scope, g.get_local_range().size())) {
      size_t local_id = g.get_local_linear_id();
      auto temp = reinterpret_cast<T *>(scratch);

      // to temp
      for (std::uint32_t i = 0; i < ElementsPerWorkItem; ++i) {
        ::new (temp + local_id * ElementsPerWorkItem + i) T(values[i]);
      }

      // sort
      sycl::detail::merge_sort(
          g, std::make_tuple(temp), range_size * ElementsPerWorkItem,
          [this](auto x, auto y) {
            return comp(std::get<0>(x), std::get<0>(y));
          },
          scratch + range_size * ElementsPerWorkItem * sizeof(T) + alignof(T));

      // from temp
      std::size_t shift{};
      for (std::uint32_t i = 0; i < ElementsPerWorkItem; ++i) {
        if constexpr (std::is_same_v<Properties, detail::is_blocked>) {
          shift = local_id * ElementsPerWorkItem + i;
        } else if constexpr (std::is_same_v<Properties, detail::is_striped>) {
          shift = i * range_size + local_id;
        }
        values[i] = temp[shift];
      }
    }
#endif
    (void)values;
    (void)g;
    (void)properties;
  }

  static constexpr std::size_t memory_required(sycl::memory_scope scope,
                                               size_t range_size) {
    return 2 * joint_sorter<>::template memory_required<T>(scope, range_size);
  }
};

template <typename T, typename U, typename CompareT = std::less<>,
          std::size_t ElementsPerWorkItem = 1>
class group_key_value_sorter {

  CompareT comp;
  std::byte *scratch;
  std::size_t scratch_size;

public:
  template <std::size_t Extent>
  group_key_value_sorter(sycl::span<std::byte, Extent> scratch_,
                         CompareT comp_ = {})
      : comp(comp_), scratch(scratch_.data()), scratch_size(scratch_.size()) {}

  template <typename Group>
  std::tuple<T, U> operator()(Group g, T key, U value) {

    static_assert(ElementsPerWorkItem == 1,
                  "ElementsPerWorkItem must be equal 1");

    using KeyValue = std::tuple<T, U>;
    auto this_comp = this->comp;
    auto comp_key_value = [this_comp](const KeyValue &lhs,
                                      const KeyValue &rhs) {
      return this_comp(std::get<0>(lhs).front(), std::get<0>(rhs).front());
    };
    return group_sorter<KeyValue, ElementsPerWorkItem,
                        decltype(comp_key_value)>(
        sycl::span{scratch, scratch_size},
        comp_key_value)(g, KeyValue(key, value));
  }

  template <typename Group, typename Properties>
  void operator()(Group g, sycl::span<T, ElementsPerWorkItem> keys,
                  sycl::span<U, ElementsPerWorkItem> values,
                  Properties property) {
#ifdef __SYCL_DEVICE_ONLY__
    auto range_size = g.get_local_linear_range();
    if (scratch_size >=
        memory_required(Group::fence_scope, g.get_local_range().size())) {
      size_t local_id = g.get_local_linear_id();
      auto temp = sycl::detail::ptrToTuple<T, U>(
          scratch, range_size * ElementsPerWorkItem);

      // to temp
      for (std::uint32_t i = 0; i < ElementsPerWorkItem; ++i) {
        ::new (std::get<0>(temp) + local_id * ElementsPerWorkItem + i)
            T(keys[i]);
        ::new (std::get<1>(temp) + local_id * ElementsPerWorkItem + i)
            T(values[i]);
      }

      // sort
      sycl::detail::merge_sort(
          g, temp, range_size * ElementsPerWorkItem,
          [this](auto x, auto y) {
            return comp(std::get<0>(x), std::get<0>(y));
          },
          scratch +
              range_size * ElementsPerWorkItem * sizeof(std::tuple<T, U>) +
              alignof(std::tuple<T, U>));

      // from temp
      std::size_t shift{};
      for (std::uint32_t i = 0; i < ElementsPerWorkItem; ++i) {
        if constexpr (std::is_same_v<Properties, detail::is_blocked>) {
          shift = local_id * ElementsPerWorkItem + i;
        } else if constexpr (std::is_same_v<Properties, detail::is_striped>) {
          shift = i * g.get_local_linear_range() + local_id;
        }

        keys[i] = std::get<0>(temp)[shift];
        values[i] = std::get<1>(temp)[shift];
      }
    }
    // TODO: add else branch
#endif
  }

  static constexpr std::size_t memory_required(sycl::memory_scope scope,
                                               std::size_t range_size) {
    return group_sorter<std::tuple<T, U>, ElementsPerWorkItem,
                        CompareT>::memory_required(scope, range_size);
  }
};

} // namespace default_sorters

enum class sorting_order { ascending, descending };

namespace detail {

template <typename T, sorting_order = sorting_order::ascending>
struct ConvertToComp {
  using Type = std::less<T>;
};

template <typename T> struct ConvertToComp<T, sorting_order::descending> {
  using Type = std::greater<T>;
};
} // namespace detail

namespace radix_sorters {

template <typename ValT, sorting_order OrderT = sorting_order::ascending,
          unsigned int BitsPerPass = 4>
class joint_sorter {

  std::byte *scratch = nullptr;
  uint32_t first_bit = 0;
  uint32_t last_bit = 0;
  std::size_t scratch_size = 0;

  static constexpr uint32_t bits = BitsPerPass;

public:
  template <std::size_t Extent>
  joint_sorter(sycl::span<std::byte, Extent> scratch_,
               const std::bitset<sizeof(ValT) *CHAR_BIT> mask =
                   std::bitset<sizeof(ValT) * CHAR_BIT>(
                       std::numeric_limits<unsigned long long>::max()))
      : scratch(scratch_.data()), scratch_size(scratch_.size()) {
    static_assert((std::is_arithmetic<ValT>::value ||
                   std::is_same<ValT, sycl::half>::value ||
                   std::is_same<ValT, sycl::ext::oneapi::bfloat16>::value),
                  "radix sort is not supported for the given type");

    first_bit = 0;
    while (first_bit < mask.size() && !mask[first_bit])
      ++first_bit;

    last_bit = first_bit;
    while (last_bit < mask.size() && mask[last_bit])
      ++last_bit;
  }

  template <typename GroupT, typename PtrT>
  void operator()(GroupT g, PtrT first, PtrT last) {
    (void)g;
    (void)first;
    (void)last;
#ifdef __SYCL_DEVICE_ONLY__
    sycl::detail::privateDynamicSort</*is_key_value=*/false,
                                     OrderT == sorting_order::ascending,
                                     /*empty*/ 1, BitsPerPass>(
        g, first, /*empty*/ first, (last - first) > 0 ? (last - first) : 0,
        scratch, first_bit, last_bit);
#else
    throw sycl::exception(
        std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
        "radix_sorters::joint_sorter is not supported on host device.");
#endif
  }

  static constexpr std::size_t memory_required(sycl::memory_scope scope,
                                               std::size_t range_size) {
    // Scope is not important so far
    (void)scope;
    return range_size * sizeof(ValT) +
           (1 << bits) * range_size * sizeof(uint32_t) + alignof(uint32_t);
  }
};

template <typename ValT, sorting_order OrderT = sorting_order::ascending,
          size_t ElementsPerWorkItem = 1, unsigned int BitsPerPass = 4>
class group_sorter {

  std::byte *scratch = nullptr;
  uint32_t first_bit = 0;
  uint32_t last_bit = 0;
  std::size_t scratch_size = 0;

  static constexpr uint32_t bits = BitsPerPass;

public:
  template <std::size_t Extent>
  group_sorter(sycl::span<std::byte, Extent> scratch_,
               const std::bitset<sizeof(ValT) *CHAR_BIT> mask =
                   std::bitset<sizeof(ValT) * CHAR_BIT>(
                       std::numeric_limits<unsigned long long>::max()))
      : scratch(scratch_.data()), scratch_size(scratch_.size()) {
    static_assert((std::is_arithmetic<ValT>::value ||
                   std::is_same<ValT, sycl::half>::value ||
                   std::is_same<ValT, sycl::ext::oneapi::bfloat16>::value),
                  "radix sort is not usable");

    first_bit = 0;
    while (first_bit < mask.size() && !mask[first_bit])
      ++first_bit;

    last_bit = first_bit;
    while (last_bit < mask.size() && mask[last_bit])
      ++last_bit;
  }

  template <typename GroupT> ValT operator()(GroupT g, ValT val) {
    (void)g;
    (void)val;
#ifdef __SYCL_DEVICE_ONLY__
    ValT result[]{val};
    sycl::detail::privateStaticSort</*is_key_value=*/false,
                                    /*is_blocked=*/true,
                                    OrderT == sorting_order::ascending,
                                    /*items_per_work_item=*/1, bits>(
        g, result, /*empty*/ result, scratch, first_bit, last_bit);
    return result[0];
#else
    throw sycl::exception(
        std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
        "radix_sorter is not supported on host device.");
#endif
  }

  template <typename Group, typename Properties>
  void operator()(Group g, sycl::span<ValT, ElementsPerWorkItem> values,
                  Properties properties) {
    (void)values;
    (void)g;
    (void)properties;
    static_assert(std::is_same_v<decltype(g), sycl::sub_group>);
  }

  static constexpr size_t memory_required(sycl::memory_scope scope,
                                          size_t range_size) {
    (void)scope;
    return std::max(range_size * sizeof(ValT),
                    range_size * (1 << bits) * sizeof(uint32_t));
  }
};

template <typename T, typename U,
          sorting_order Order = sorting_order::ascending,
          size_t ElementsPerWorkItem = 1, unsigned int BitsPerPass = 4>
class group_key_value_sorter {

  std::byte *scratch;
  uint32_t first_bit;
  uint32_t last_bit;
  size_t scratch_size;

  static constexpr uint32_t bits = BitsPerPass;

public:
  template <std::size_t Extent>
  group_key_value_sorter(
      sycl::span<std::byte, Extent> scratch_,
      const std::bitset<sizeof(T) *CHAR_BIT> mask =
          std::bitset<sizeof(T) * CHAR_BIT>(
              std::numeric_limits<unsigned long long>::max()))
      : scratch(scratch_.data()), scratch_size(scratch_.size()) {
    static_assert(
        (std::is_arithmetic<T>::value || std::is_same<T, sycl::half>::value),
        "radix sort is not usable");
    first_bit = 0;
    while (first_bit < mask.size() && !mask[first_bit])
      ++first_bit;

    last_bit = first_bit;
    while (last_bit < mask.size() && mask[last_bit])
      ++last_bit;
  }

  template <typename Group> std::tuple<T, U> operator()(Group g, T key, U val) {
    static_assert(ElementsPerWorkItem == 1, "ElementsPerWorkItem must be 1");
    T key_result[]{key};
    U val_result[]{val};
#ifdef __SYCL_DEVICE_ONLY__

#if 0
    sycl::detail::privateStaticSort<
        /*is_key_value=*/true,
        /*is_blocked=*/true, 1, bits>(
        g, key_result, val_result, Order == sorting_order::ascending,
        scratch, sycl::detail::Builder::getNDItem<Group::dimensions>(),
        first_bit, last_bit);
#endif
#endif
    key = key_result[0];
    val = val_result[0];
    return {key, val};
  }

  template <typename Group, typename Properties>
  void operator()(Group g, sycl::span<T, ElementsPerWorkItem> keys,
                  sycl::span<U, ElementsPerWorkItem> vals,
                  Properties property) {
#ifdef __SYCL_DEVICE_ONLY__
    sycl::detail::privateStaticSort<
        /*is_key_value=*/true, std::is_same_v<Properties, detail::is_blocked>,
        Order == sorting_order::ascending, ElementsPerWorkItem, bits>(
        g, keys.data(), vals.data(), scratch, first_bit, last_bit);
#endif
  }

  static constexpr std::size_t memory_required(sycl::memory_scope scope,
                                               std::size_t range_size) {
    return std::max(range_size * ElementsPerWorkItem * (sizeof(T) + sizeof(U)) +
                        2 * alignof(uint32_t),
                    range_size * (1 << bits) * sizeof(uint32_t) + alignof(T));
  }
};

} // namespace radix_sorters

enum class group_algorithm_data_placement { blocked, striped };

template <typename ValT, sorting_order OrderT = sorting_order::ascending,
          unsigned int BitsPerPass = 4>
class radix_sorter {

  std::byte *scratch = nullptr;
  uint32_t first_bit = 0;
  uint32_t last_bit = 0;
  size_t scratch_size = 0;

  static constexpr uint32_t bits = BitsPerPass;

public:
  template <size_t Extent>
  radix_sorter(sycl::span<std::byte, Extent> scratch_,
               const std::bitset<sizeof(ValT) *CHAR_BIT> mask =
                   std::bitset<sizeof(ValT) * CHAR_BIT>(
                       (std::numeric_limits<unsigned long long>::max)()))
      : scratch(scratch_.data()), scratch_size(scratch_.size()) {
    static_assert((std::is_arithmetic<ValT>::value ||
                   std::is_same<ValT, sycl::half>::value ||
                   std::is_same<ValT, sycl::ext::oneapi::bfloat16>::value),
                  "radix sort is not usable");

    first_bit = 0;
    while (first_bit < mask.size() && !mask[first_bit])
      ++first_bit;

    last_bit = first_bit;
    while (last_bit < mask.size() && mask[last_bit])
      ++last_bit;
  }

  template <typename GroupT, typename PtrT>
  void operator()(GroupT g, PtrT first, PtrT last) {
    (void)g;
    (void)first;
    (void)last;
#ifdef __SYCL_DEVICE_ONLY__
    sycl::detail::privateDynamicSort</*is_key_value=*/false,
                                     OrderT == sorting_order::ascending,
                                     /*empty*/ 1, BitsPerPass>(
        g, first, /*empty*/ first, (last - first) > 0 ? (last - first) : 0,
        scratch, first_bit, last_bit);
#else
    throw sycl::exception(
        std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
        "radix_sorter is not supported on host device.");
#endif
  }

  template <typename GroupT> ValT operator()(GroupT g, ValT val) {
    (void)g;
    (void)val;
#ifdef __SYCL_DEVICE_ONLY__
    ValT result[]{val};
    sycl::detail::privateStaticSort</*is_key_value=*/false,
                                    /*is_blocked=*/true,
                                    OrderT == sorting_order::ascending,
                                    /*items_per_work_item=*/1, bits>(
        g, result, /*empty*/ result, scratch, first_bit, last_bit);
    return result[0];
#else
    throw sycl::exception(
        std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
        "radix_sorter is not supported on host device.");
#endif
  }

  static constexpr size_t memory_required(sycl::memory_scope scope,
                                          size_t range_size) {
    // Scope is not important so far
    (void)scope;
    return range_size * sizeof(ValT) +
           (1 << bits) * range_size * sizeof(uint32_t) + alignof(uint32_t);
  }

  // memory_helpers
  template <int dimensions = 1>
  static constexpr size_t memory_required(sycl::memory_scope scope,
                                          sycl::range<dimensions> local_range) {
    // Scope is not important so far
    (void)scope;
    return (std::max)(local_range.size() * sizeof(ValT),
                      local_range.size() * (1 << bits) * sizeof(uint32_t));
  }
};

} // namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
#endif
