//==----------- fusion_properties.hpp --- SYCL fusion properties -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>
#include <sycl/detail/property_helper.hpp>
#include <sycl/properties/property_traits.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace codeplay {
namespace property {

class promote_private
    : public detail::DataLessProperty<detail::FusionPromotePrivate> {};

class promote_local
    : public detail::DataLessProperty<detail::FusionPromoteLocal> {};

} // namespace property
} // namespace codeplay
} // namespace ext

// Forward declarations
template <typename T, int Dimensions, typename AllocatorT, typename Enable>
class buffer;

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT>
class accessor;

// Property trait specializations.
template <>
struct is_property<ext::codeplay::property::promote_private> : std::true_type {
};

template <>
struct is_property<ext::codeplay::property::promote_local> : std::true_type {};

// Buffer property trait specializations
template <typename T, int Dimensions, typename AllocatorT>
struct is_property_of<ext::codeplay::property::promote_private,
                      buffer<T, Dimensions, AllocatorT, void>>
    : std::true_type {};

template <typename T, int Dimensions, typename AllocatorT>
struct is_property_of<ext::codeplay::property::promote_local,
                      buffer<T, Dimensions, AllocatorT, void>>
    : std::true_type {};

// Accessor property trait specializations
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT>
struct is_property_of<ext::codeplay::property::promote_private,
                      accessor<DataT, Dimensions, AccessMode, AccessTarget,
                               IsPlaceholder, PropertyListT>> : std::true_type {
};

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT>
struct is_property_of<ext::codeplay::property::promote_local,
                      accessor<DataT, Dimensions, AccessMode, AccessTarget,
                               IsPlaceholder, PropertyListT>> : std::true_type {
};

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
