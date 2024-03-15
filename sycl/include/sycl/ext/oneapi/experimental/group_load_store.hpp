//==---- group_load_store.hpp --- SYCL extension for group loads/stores ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements https://github.com/intel/llvm/pull/7593

#pragma once

#include <sycl/ext/oneapi/properties/properties.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

// TODO: Should that go into other place (shared between different extensions)?
enum class data_placement_enum { blocked, striped };

struct data_placement_key
    : detail::compile_time_property_key<detail::PropKind::DataPlacement> {
  template <data_placement_enum Placement>
  using value_t =
      property_value<data_placement_key,
                     // TODO: Extension uses data_placement_enum directly here.
                     std::integral_constant<int, static_cast<int>(Placement)>>;
};

template <data_placement_enum Placement>
inline constexpr data_placement_key::value_t<Placement> data_placement;

inline constexpr data_placement_key::value_t<data_placement_enum::blocked>
    data_placement_blocked;
inline constexpr data_placement_key::value_t<data_placement_enum::striped>
    data_placement_striped;

namespace detail {
using namespace sycl::detail;

template <typename InputIteratorT, typename OutputElemT>
inline constexpr bool verify_load_types =
    std::is_convertible_v<remove_decoration_t<typename std::iterator_traits<
                              InputIteratorT>::value_type>,
                          OutputElemT>;

template <typename InputElemT, typename OutputIteratorT>
inline constexpr bool verify_store_types =
    std::is_convertible_v<InputElemT,
                          remove_decoration_t<typename std::iterator_traits<
                              OutputIteratorT>::value_type>>;

template <typename Properties> constexpr bool isBlocked(Properties properties) {
  if constexpr (properties.template has_property<data_placement_key>())
    return properties.template get_property<data_placement_key>() ==
           data_placement_blocked;
  else
    return true;
}

template <bool IsBlocked, int VEC_OR_ARRAY_SIZE, typename GroupTy>
int get_mem_idx(GroupTy g, int vec_or_array_idx) {
  if constexpr (IsBlocked)
    return g.get_local_linear_id() * VEC_OR_ARRAY_SIZE + vec_or_array_idx;
  else
    return g.get_local_linear_id() +
           g.get_local_linear_range() * vec_or_array_idx;
}
} // namespace detail

#ifdef __SYCL_DEVICE_ONLY__
// Load API span overload.
template <typename Group, typename InputIteratorT, typename OutputT,
          std::size_t ElementsPerWorkItem,
          typename Properties = decltype(properties())>
std::enable_if_t<detail::verify_load_types<InputIteratorT, OutputT> &&
                 detail::is_generic_group_v<Group>>
group_load(Group g, InputIteratorT in_ptr,
           span<OutputT, ElementsPerWorkItem> out, Properties properties = {}) {
  constexpr bool blocked = detail::isBlocked(properties);

  group_barrier(g);
  for (int i = 0; i < out.size(); ++i)
    out[i] = in_ptr[detail::get_mem_idx<blocked, ElementsPerWorkItem>(g, i)];
  group_barrier(g);
}

// Store API span overload.
template <typename Group, typename InputT, std::size_t ElementsPerWorkItem,
          typename OutputIteratorT,
          typename Properties = decltype(properties())>
std::enable_if_t<detail::verify_store_types<InputT, OutputIteratorT> &&
                 detail::is_generic_group_v<Group>>
group_store(Group g, const span<InputT, ElementsPerWorkItem> in,
            OutputIteratorT out_ptr, Properties properties = {}) {
  constexpr bool blocked = detail::isBlocked(properties);

  group_barrier(g);
  for (int i = 0; i < in.size(); ++i)
    out_ptr[detail::get_mem_idx<blocked, ElementsPerWorkItem>(g, i)] = in[i];
  group_barrier(g);
}

// Load API scalar.
template <typename Group, typename InputIteratorT, typename OutputT,
          typename Properties = decltype(properties())>
std::enable_if_t<detail::verify_load_types<InputIteratorT, OutputT> &&
                 detail::is_generic_group_v<Group>>
group_load(Group g, InputIteratorT in_ptr, OutputT &out,
           Properties properties = {}) {
  group_load(g, in_ptr, span<OutputT, 1>(&out, 1), properties);
}

// Store API scalar.
template <typename Group, typename InputT, typename OutputIteratorT,
          typename Properties = decltype(properties())>
std::enable_if_t<detail::verify_store_types<InputT, OutputIteratorT> &&
                 detail::is_generic_group_v<Group>>
group_store(Group g, const InputT &in, OutputIteratorT out_ptr,
            Properties properties = {}) {
  group_store(g, span<const InputT, 1>(&in, 1), out_ptr, properties);
}

// Load API sycl::vec overload.
template <typename Group, typename InputIteratorT, typename OutputT, int N,
          typename Properties = decltype(properties())>
std::enable_if_t<detail::verify_load_types<InputIteratorT, OutputT> &&
                 detail::is_generic_group_v<Group>>
group_load(Group g, InputIteratorT in_ptr, sycl::vec<OutputT, N> &out,
           Properties properties = {}) {
  group_load(g, in_ptr, span<OutputT, N>(&out[0], N), properties);
}

// Store API sycl::vec overload.
template <typename Group, typename InputT, int N, typename OutputIteratorT,
          typename Properties = decltype(properties())>
std::enable_if_t<detail::verify_store_types<InputT, OutputIteratorT> &&
                 detail::is_generic_group_v<Group>>
group_store(Group g, const sycl::vec<InputT, N> &in, OutputIteratorT out_ptr,
            Properties properties = {}) {
  group_store(g, span<const InputT, N>(&in[0], N), out_ptr, properties);
}

#else
template <typename... Args> void group_load(Args...) {
  throw sycl::exception(
      std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
      "Group loads/stores are not supported on host.");
}
template <typename... Args> void group_store(Args...) {
  throw sycl::exception(
      std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
      "Group loads/stores are not supported on host.");
}
#endif
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
