//==----------- fusion_properties.hpp --- SYCL fusion properties -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>              // for mode, placeholder
#include <sycl/detail/property_helper.hpp>     // for DataLessPropKind, Dat...
#include <sycl/properties/property_traits.hpp> // for is_property, is_prope...
#include <sycl/queue.hpp>                      // for queue

#include <type_traits> // for true_type

namespace sycl {
inline namespace _V1 {
namespace ext::codeplay::experimental::property {

class promote_private
    : public detail::DataLessProperty<detail::FusionPromotePrivate> {};

class promote_local
    : public detail::DataLessProperty<detail::FusionPromoteLocal> {};

class no_barriers : public detail::DataLessProperty<detail::FusionNoBarrier> {};

class force_fusion : public detail::DataLessProperty<detail::FusionForce> {};

namespace queue {
class enable_fusion : public detail::DataLessProperty<detail::FusionEnable> {};
} // namespace queue

} // namespace ext::codeplay::experimental::property

// Forward declarations
template <typename T, int Dimensions, typename AllocatorT, typename Enable>
class buffer;

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT>
class accessor;

class queue;

// Property trait specializations.
template <>
struct is_property<ext::codeplay::experimental::property::promote_private>
    : std::true_type {};

template <>
struct is_property<ext::codeplay::experimental::property::promote_local>
    : std::true_type {};

template <>
struct is_property<ext::codeplay::experimental::property::no_barriers>
    : std::true_type {};

template <>
struct is_property<ext::codeplay::experimental::property::force_fusion>
    : std::true_type {};

template <>
struct is_property<ext::codeplay::experimental::property::queue::enable_fusion>
    : std::true_type {};

// Buffer property trait specializations
template <typename T, int Dimensions, typename AllocatorT>
struct is_property_of<ext::codeplay::experimental::property::promote_private,
                      buffer<T, Dimensions, AllocatorT, void>>
    : std::true_type {};

template <typename T, int Dimensions, typename AllocatorT>
struct is_property_of<ext::codeplay::experimental::property::promote_local,
                      buffer<T, Dimensions, AllocatorT, void>>
    : std::true_type {};

// Accessor property trait specializations
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT>
struct is_property_of<ext::codeplay::experimental::property::promote_private,
                      accessor<DataT, Dimensions, AccessMode, AccessTarget,
                               IsPlaceholder, PropertyListT>> : std::true_type {
};

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT>
struct is_property_of<ext::codeplay::experimental::property::promote_local,
                      accessor<DataT, Dimensions, AccessMode, AccessTarget,
                               IsPlaceholder, PropertyListT>> : std::true_type {
};

// Queue property trait specializations
template <>
struct is_property_of<
    ext::codeplay::experimental::property::queue::enable_fusion, queue>
    : std::true_type {};

} // namespace _V1
} // namespace sycl
