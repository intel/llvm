//==------- group_helpers_sorters.hpp - SYCL sorters and group helpers -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)

#include <sycl/aliases.hpp>             // for half
#include <sycl/builtins.hpp>            // for min
#include <sycl/detail/pi.h>             // for PI_ERROR_INVALID_DEVICE
#include <sycl/exception.hpp>           // for sycl_category, exception
#include <sycl/ext/oneapi/bfloat16.hpp> // for bfloat16
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/memory_enums.hpp> // for memory_scope
#include <sycl/range.hpp>        // for range
#include <sycl/sycl_span.hpp>    // for span

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/detail/group_sort_impl.hpp>
#endif

#include <bitset>       // for bitset
#include <cstddef>      // for size_t, byte
#include <functional>   // for less, greater
#include <limits.h>     // for CHAR_BIT
#include <limits>       // for numeric_limits
#include <stdint.h>     // for uint32_t
#include <system_error> // for error_code
#include <type_traits>  // for is_same, is_arithmetic

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

enum class group_algorithm_data_placement { blocked, striped };

struct input_data_placement_key
    : detail::compile_time_property_key<detail::PropKind::InputDataPlacement> {
  template <group_algorithm_data_placement Placement>
  using value_t =
      property_value<input_data_placement_key,
                     std::integral_constant<int, static_cast<int>(Placement)>>;
};

struct output_data_placement_key
    : detail::compile_time_property_key<detail::PropKind::OutputDataPlacement> {
  template <group_algorithm_data_placement Placement>
  using value_t =
      property_value<output_data_placement_key,
                     std::integral_constant<int, static_cast<int>(Placement)>>;
};

template <group_algorithm_data_placement Placement>
inline constexpr input_data_placement_key::value_t<Placement>
    input_data_placement;

template <group_algorithm_data_placement Placement>
inline constexpr output_data_placement_key::value_t<Placement>
    output_data_placement;

namespace detail {

template <typename Properties>
constexpr bool isInputBlocked(Properties properties) {
  if constexpr (properties.template has_property<input_data_placement_key>())
    return properties.template get_property<input_data_placement_key>() ==
           input_data_placement<group_algorithm_data_placement::blocked>;
  else
    return true;
}

template <typename Properties>
constexpr bool isOutputBlocked(Properties properties) {
  if constexpr (properties.template has_property<output_data_placement_key>())
    return properties.template get_property<output_data_placement_key>() ==
           output_data_placement<group_algorithm_data_placement::blocked>;
  else
    return true;
}

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

// Default sorter provided by the first version of the extension specification.
template <typename Compare = std::less<>> class default_sorter {
  Compare comp;
  sycl::span<std::byte> scratch;

public:
  template <size_t Extent>
  default_sorter(sycl::span<std::byte, Extent> scratch_,
                 Compare comp_ = Compare())
      : comp(comp_), scratch(scratch_) {}

  template <typename Group, typename Ptr>
  void operator()([[maybe_unused]] Group g, [[maybe_unused]] Ptr first,
                  [[maybe_unused]] Ptr last) {
#ifdef __SYCL_DEVICE_ONLY__
    using T = typename sycl::detail::GetValueType<Ptr>::type;
    size_t n = last - first;
    T *scratch_begin = sycl::detail::align_scratch<T>(scratch, g, n);
    sycl::detail::merge_sort(g, first, n, comp, scratch_begin);
#else
    throw sycl::exception(
        std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
        "default_sorter constructor is not supported on host device.");
#endif
  }

  template <typename Group, typename T>
  T operator()([[maybe_unused]] Group g, T val) {
#ifdef __SYCL_DEVICE_ONLY__
    std::size_t local_id = g.get_local_linear_id();
    auto range_size = g.get_local_range().size();
    T *scratch_begin = sycl::detail::align_scratch<T>(
        scratch, g, /* output storage and temporary storage */ 2 * range_size);
    scratch_begin[local_id] = val;
    sycl::detail::merge_sort(g, scratch_begin, range_size, comp,
                             scratch_begin + range_size);
    val = scratch_begin[local_id];
#else
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

  template <typename T, int dim = 1>
  static constexpr size_t memory_required(sycl::memory_scope scope,
                                          sycl::range<dim> r) {
    return 2 * memory_required<T>(scope, r.size());
  }
};

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

// Radix sorter provided by the first version of the extension specification.
template <typename ValT, sorting_order OrderT = sorting_order::ascending,
          unsigned int BitsPerPass = 4>
class radix_sorter {

  sycl::span<std::byte> scratch;
  uint32_t first_bit = 0;
  uint32_t last_bit = 0;

  static constexpr uint32_t bits = BitsPerPass;
  using bitset_t = std::bitset<sizeof(ValT) * CHAR_BIT>;

public:
  template <size_t Extent>
  radix_sorter(sycl::span<std::byte, Extent> scratch_,
               const bitset_t mask = bitset_t{}.set())
      : scratch(scratch_) {
    static_assert((std::is_arithmetic<ValT>::value ||
                   std::is_same<ValT, sycl::half>::value ||
                   std::is_same<ValT, sycl::ext::oneapi::bfloat16>::value),
                  "radix sort is not usable");

    for (first_bit = 0; first_bit < mask.size() && !mask[first_bit];
         ++first_bit)
      ;
    for (last_bit = first_bit; last_bit < mask.size() && mask[last_bit];
         ++last_bit)
      ;
  }

  template <typename GroupT, typename PtrT>
  void operator()([[maybe_unused]] GroupT g, [[maybe_unused]] PtrT first,
                  [[maybe_unused]] PtrT last) {
#ifdef __SYCL_DEVICE_ONLY__
    sycl::detail::privateDynamicSort</*is_key_value=*/false,
                                     OrderT == sorting_order::ascending,
                                     /*empty*/ 1, BitsPerPass>(
        g, first, /*empty*/ first, last - first, scratch.data(), first_bit,
        last_bit);
#else
    throw sycl::exception(
        std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
        "radix_sorter is not supported on host device.");
#endif
  }

  template <typename GroupT>
  ValT operator()([[maybe_unused]] GroupT g, [[maybe_unused]] ValT val) {
#ifdef __SYCL_DEVICE_ONLY__
    ValT result[]{val};
    sycl::detail::privateStaticSort</*is_key_value=*/false,
                                    /*is_blocked=*/true,
                                    OrderT == sorting_order::ascending,
                                    /*items_per_work_item=*/1, bits>(
        g, result, /*empty*/ result, scratch.data(), first_bit, last_bit);
    return result[0];
#else
    throw sycl::exception(
        std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
        "radix_sorter is not supported on host device.");
#endif
  }

  static constexpr size_t memory_required(sycl::memory_scope,
                                          size_t range_size) {
    return range_size * sizeof(ValT) +
           (1 << bits) * range_size * sizeof(uint32_t) + alignof(uint32_t);
  }

  // memory_helpers
  template <int dimensions = 1>
  static constexpr size_t memory_required(sycl::memory_scope,
                                          sycl::range<dimensions> local_range) {
    return (std::max)(local_range.size() * sizeof(ValT),
                      local_range.size() * (1 << bits) * sizeof(uint32_t));
  }
};

// Default sorters provided by the second version of the extension
// specification.
namespace default_sorters {

template <typename CompareT = std::less<>> class joint_sorter {
  CompareT comp;
  sycl::span<std::byte> scratch;

public:
  template <size_t Extent>
  joint_sorter(sycl::span<std::byte, Extent> scratch_,
               CompareT comp_ = CompareT())
      : comp(comp_), scratch(scratch_) {}

  template <typename Group, typename Ptr>
  void operator()([[maybe_unused]] Group g, [[maybe_unused]] Ptr first,
                  [[maybe_unused]] Ptr last) {
#ifdef __SYCL_DEVICE_ONLY__
    using T = typename sycl::detail::GetValueType<Ptr>::type;
    size_t n = last - first;
    T *scratch_begin = sycl::detail::align_scratch<T>(scratch, g, n);
    sycl::detail::merge_sort(g, first, n, comp, scratch_begin);
#else
    throw sycl::exception(
        std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
        "default_sorter constructor is not supported on host device.");
#endif
  }

  template <typename T>
  static size_t memory_required(sycl::memory_scope, size_t range_size) {
    return range_size * sizeof(T) + alignof(T);
  }
};

template <typename T, typename CompareT = std::less<>,
          std::size_t ElementsPerWorkItem = 1>
class group_sorter {
  CompareT comp;
  sycl::span<std::byte> scratch;

public:
  template <std::size_t Extent>
  group_sorter(sycl::span<std::byte, Extent> scratch_,
               CompareT comp_ = CompareT{})
      : comp(comp_), scratch(scratch_) {}

  template <typename Group> T operator()([[maybe_unused]] Group g, T val) {
#ifdef __SYCL_DEVICE_ONLY__
    std::size_t local_id = g.get_local_linear_id();
    auto range_size = g.get_local_range().size();
    T *scratch_begin = sycl::detail::align_scratch<T>(
        scratch, g, /* output storage and temporary storage */ 2 * range_size);
    scratch_begin[local_id] = val;
    sycl::detail::merge_sort(g, scratch_begin, range_size, comp,
                             scratch_begin + range_size);
    val = scratch_begin[local_id];
#else
    throw sycl::exception(
        std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
        "default_sorter operator() is not supported on host device.");
#endif
    return val;
  }

  template <typename Group, typename Properties>
  void operator()([[maybe_unused]] Group g,
                  [[maybe_unused]] sycl::span<T, ElementsPerWorkItem> values,
                  [[maybe_unused]] Properties properties) {
#ifdef __SYCL_DEVICE_ONLY__
    std::size_t local_id = g.get_local_linear_id();
    auto wg_size = g.get_local_range().size();
    auto number_of_elements = wg_size * ElementsPerWorkItem;
    T *scratch_begin = sycl::detail::align_scratch<T>(
        scratch, g,
        /* output storage and temporary storage */ 2 * number_of_elements);
    for (std::uint32_t i = 0; i < ElementsPerWorkItem; ++i)
      scratch_begin[local_id * ElementsPerWorkItem + i] = values[i];
    sycl::detail::merge_sort(g, scratch_begin, number_of_elements, comp,
                             scratch_begin + number_of_elements);

    std::size_t shift{};
    for (std::uint32_t i = 0; i < ElementsPerWorkItem; ++i) {
      if constexpr (detail::isOutputBlocked(properties)) {
        shift = local_id * ElementsPerWorkItem + i;
      } else {
        shift = i * wg_size + local_id;
      }
      values[i] = scratch_begin[shift];
    }
#endif
  }

  static std::size_t memory_required(sycl::memory_scope scope,
                                     size_t range_size) {
    return 2 * joint_sorter<>::template memory_required<T>(
                   scope, range_size * ElementsPerWorkItem);
  }
};

template <typename KeyTy, typename ValueTy, typename CompareT = std::less<>,
          std::size_t ElementsPerWorkItem = 1>
class group_key_value_sorter {
  CompareT comp;
  sycl::span<std::byte> scratch;

public:
  template <std::size_t Extent>
  group_key_value_sorter(sycl::span<std::byte, Extent> scratch_,
                         CompareT comp_ = {})
      : comp(comp_), scratch(scratch_) {}

  template <typename Group>
  std::tuple<KeyTy, ValueTy> operator()(Group g, KeyTy key, ValueTy value) {
    static_assert(ElementsPerWorkItem == 1,
                  "ElementsPerWorkItem must be equal 1");

    using KeyValue = std::tuple<KeyTy, ValueTy>;
    auto comp_key_value = [this_comp = this->comp](const KeyValue &lhs,
                                                   const KeyValue &rhs) {
      return this_comp(std::get<0>(lhs), std::get<0>(rhs));
    };
    return group_sorter<KeyValue, decltype(comp_key_value),
                        ElementsPerWorkItem>(scratch, comp_key_value)(
        g, KeyValue(key, value));
  }

  static std::size_t memory_required(sycl::memory_scope scope,
                                     std::size_t range_size) {
    return group_sorter<std::tuple<KeyTy, ValueTy>, CompareT,
                        ElementsPerWorkItem>::memory_required(scope,
                                                              range_size);
  }
};
} // namespace default_sorters

// Radix sorters provided by the second version of the extension specification.
namespace radix_sorters {

template <typename ValT, sorting_order OrderT = sorting_order::ascending,
          unsigned int BitsPerPass = 4>
class joint_sorter {

  sycl::span<std::byte> scratch;
  uint32_t first_bit = 0;
  uint32_t last_bit = 0;

  static constexpr uint32_t bits = BitsPerPass;
  using bitset_t = std::bitset<sizeof(ValT) * CHAR_BIT>;

public:
  template <std::size_t Extent>
  joint_sorter(sycl::span<std::byte, Extent> scratch_,
               const bitset_t mask = bitset_t{}.set())
      : scratch(scratch_) {
    static_assert((std::is_arithmetic<ValT>::value ||
                   std::is_same<ValT, sycl::half>::value ||
                   std::is_same<ValT, sycl::ext::oneapi::bfloat16>::value),
                  "radix sort is not supported for the given type");

    for (first_bit = 0; first_bit < mask.size() && !mask[first_bit];
         ++first_bit)
      ;
    for (last_bit = first_bit; last_bit < mask.size() && mask[last_bit];
         ++last_bit)
      ;
  }

  template <typename GroupT, typename PtrT>
  void operator()([[maybe_unused]] GroupT g, [[maybe_unused]] PtrT first,
                  [[maybe_unused]] PtrT last) {
#ifdef __SYCL_DEVICE_ONLY__
    sycl::detail::privateDynamicSort</*is_key_value=*/false,
                                     OrderT == sorting_order::ascending,
                                     /*empty*/ 1, BitsPerPass>(
        g, first, /*empty*/ first, last - first, scratch.data(), first_bit,
        last_bit);
#else
    throw sycl::exception(
        std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
        "radix_sorter is not supported on host device.");
#endif
  }

  static constexpr std::size_t
  memory_required([[maybe_unused]] sycl::memory_scope scope,
                  std::size_t range_size) {
    return range_size * sizeof(ValT) +
           (1 << bits) * range_size * sizeof(uint32_t) + alignof(uint32_t);
  }
};

template <typename ValT, sorting_order OrderT = sorting_order::ascending,
          size_t ElementsPerWorkItem = 1, unsigned int BitsPerPass = 4>
class group_sorter {

  sycl::span<std::byte> scratch;
  uint32_t first_bit = 0;
  uint32_t last_bit = 0;

  static constexpr uint32_t bits = BitsPerPass;
  using bitset_t = std::bitset<sizeof(ValT) * CHAR_BIT>;

public:
  template <std::size_t Extent>
  group_sorter(sycl::span<std::byte, Extent> scratch_,
               const bitset_t mask = bitset_t{}.set())
      : scratch(scratch_) {
    static_assert((std::is_arithmetic<ValT>::value ||
                   std::is_same<ValT, sycl::half>::value ||
                   std::is_same<ValT, sycl::ext::oneapi::bfloat16>::value),
                  "radix sort is not usable");

    for (first_bit = 0; first_bit < mask.size() && !mask[first_bit];
         ++first_bit)
      ;
    for (last_bit = first_bit; last_bit < mask.size() && mask[last_bit];
         ++last_bit)
      ;
  }

  template <typename GroupT>
  ValT operator()([[maybe_unused]] GroupT g, [[maybe_unused]] ValT val) {
#ifdef __SYCL_DEVICE_ONLY__
    ValT result[]{val};
    sycl::detail::privateStaticSort</*is_key_value=*/false,
                                    /*is_blocked=*/true,
                                    OrderT == sorting_order::ascending,
                                    /*items_per_work_item=*/1, bits>(
        g, result, /*empty*/ result, scratch.data(), first_bit, last_bit);
    return result[0];
#else
    throw sycl::exception(
        std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
        "radix_sorter is not supported on host device.");
#endif
  }

  template <typename Group, typename Properties>
  void operator()([[maybe_unused]] Group g,
                  [[maybe_unused]] sycl::span<ValT, ElementsPerWorkItem> values,
                  [[maybe_unused]] Properties properties) {
#ifdef __SYCL_DEVICE_ONLY__
    sycl::detail::privateStaticSort<
        /*is_key_value=*/false, detail::isOutputBlocked(properties),
        OrderT == sorting_order::ascending, ElementsPerWorkItem, bits>(
        g, values.data(), /*empty*/ values.data(), scratch.data(), first_bit,
        last_bit);
#endif
  }

  static constexpr size_t
  memory_required([[maybe_unused]] sycl::memory_scope scope,
                  size_t range_size) {
    return (std::max)(range_size * sizeof(ValT),
                      range_size * (1 << bits) * sizeof(uint32_t));
  }
};

template <typename KeyTy, typename ValueTy,
          sorting_order Order = sorting_order::ascending,
          size_t ElementsPerWorkItem = 1, unsigned int BitsPerPass = 4>
class group_key_value_sorter {
  sycl::span<std::byte> scratch;
  uint32_t first_bit;
  uint32_t last_bit;

  static constexpr uint32_t bits = BitsPerPass;
  using bitset_t = std::bitset<sizeof(KeyTy) * CHAR_BIT>;

public:
  template <std::size_t Extent>
  group_key_value_sorter(sycl::span<std::byte, Extent> scratch_,
                         const bitset_t mask = bitset_t{}.set())
      : scratch(scratch_) {
    static_assert((std::is_arithmetic<KeyTy>::value ||
                   std::is_same<KeyTy, sycl::half>::value),
                  "radix sort is not usable");
    for (first_bit = 0; first_bit < mask.size() && !mask[first_bit];
         ++first_bit)
      ;
    for (last_bit = first_bit; last_bit < mask.size() && mask[last_bit];
         ++last_bit)
      ;
  }

  template <typename Group>
  std::tuple<KeyTy, ValueTy> operator()([[maybe_unused]] Group g, KeyTy key,
                                        ValueTy val) {
    static_assert(ElementsPerWorkItem == 1, "ElementsPerWorkItem must be 1");
    KeyTy key_result[]{key};
    ValueTy val_result[]{val};
#ifdef __SYCL_DEVICE_ONLY__
    sycl::detail::privateStaticSort<
        /*is_key_value=*/true,
        /*is_blocked=*/true, Order == sorting_order::ascending, 1, bits>(
        g, key_result, val_result, scratch.data(), first_bit, last_bit);
#endif
    key = key_result[0];
    val = val_result[0];
    return {key, val};
  }

  static constexpr std::size_t memory_required(sycl::memory_scope,
                                               std::size_t range_size) {
    return (std::max)(range_size * ElementsPerWorkItem *
                          (sizeof(KeyTy) + sizeof(ValueTy)),
                      range_size * (1 << bits) * sizeof(uint32_t));
  }
};
} // namespace radix_sorters

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
#endif
