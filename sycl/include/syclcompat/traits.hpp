/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL compatibility extension
 *
 *  traits.hpp
 *
 *  Description:
 *    Type traits for the SYCL compatibility extension
 **************************************************************************/

#pragma once

#include <cstddef>
#include <sycl/detail/core.hpp>
#include <type_traits>

namespace syclcompat {

// Equivalent to C++20's std::type_identity (used to create non-deduced
// contexts)
template <class T> struct type_identity {
  using type = T;
};
template <class T> using type_identity_t = typename type_identity<T>::type;

// Defines the operand type for arithemtic operations on T. This is identity
// for all types except pointers, for which it is std::ptrdiff_t
template <typename T> struct arith {
  using type = std::conditional_t<std::is_pointer_v<T>, std::ptrdiff_t, T>;
};
template <typename T> using arith_t = typename arith<T>::type;

// Traits to check device function signature matches args (with or without local
// mem)
template <auto F, typename... Args>
struct device_fn_invocable : std::is_invocable<decltype(F), Args...> {};

template <auto F, typename... Args>
struct device_fn_lmem_invocable : std::is_invocable<decltype(F), Args..., char *> {};

template <typename LaunchPolicy, auto F, typename... Args>
constexpr inline bool args_compatible =
    std::conditional_t<LaunchPolicy::HasLocalMem, device_fn_lmem_invocable<F, Args...>,
                     device_fn_invocable<F, Args...>>::value;

namespace detail {
// Trait for identifying sycl::range and sycl::nd_range.

template <typename T> struct is_range : std::false_type{};
template <int Dim> struct is_range<sycl::range<Dim>> : std::true_type {};

template <typename T>
constexpr bool is_range_v = is_range<T>::value;

template <typename T> struct is_nd_range : std::false_type{};
template <int Dim> struct is_nd_range<sycl::nd_range<Dim>> : std::true_type {};

template <typename T>
constexpr bool is_nd_range_v = is_nd_range<T>::value;

template <typename T>
constexpr bool is_range_or_nd_range_v = std::disjunction_v<is_range<T>, is_nd_range<T>>;

// Trait to extract dimension from range & nd_range
template <typename T> struct range_dimension;

template <template <int Dim> typename RangeT, int RangeDim>
struct range_dimension<RangeT<RangeDim>> {
  static_assert(is_range_or_nd_range_v<RangeT<RangeDim>>);
  static constexpr int Dim = RangeDim;
};

template <typename RangeT>
constexpr int range_dimension_v = range_dimension<RangeT>::Dim;

// Trait range_to_item_t to convert nd_range -> nd_item, range -> item
template <typename T> struct range_to_item_map;
template <int Dim> struct range_to_item_map<sycl::nd_range<Dim>> {
  using ItemT = sycl::nd_item<Dim>;
};
template <int Dim> struct range_to_item_map<sycl::range<Dim>> {
  using ItemT = sycl::item<Dim>;
};

template <typename T>
using range_to_item_t = typename range_to_item_map<T>::ItemT;

} // namespace detail

} // namespace syclcompat
