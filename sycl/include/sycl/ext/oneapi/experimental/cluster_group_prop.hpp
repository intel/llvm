//==--- cluster_group_prop.hpp --- SYCL extension for cuda cluster group ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/range.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

namespace cuda {
template <int Dim>
struct cluster_size
    : ::sycl::ext::oneapi::experimental::detail::run_time_property_key<
          ::sycl::ext::oneapi::experimental::detail::ClusterLaunch> {
  cluster_size(const range<Dim> &size) : size(size) {}
  sycl::range<Dim> get_cluster_size() { return size; }

private:
  range<Dim> size;
};

template <int Dim> using cluster_size_key = cluster_size<Dim>;

} // namespace cuda

template <>
struct is_property_key<cuda::cluster_size_key<1>> : std::true_type {};
template <>
struct is_property_key<cuda::cluster_size_key<2>> : std::true_type {};
template <>
struct is_property_key<cuda::cluster_size_key<3>> : std::true_type {};

template <typename T>
struct is_property_key_of<cuda::cluster_size_key<1>, T> : std::true_type {};

template <typename T>
struct is_property_key_of<cuda::cluster_size_key<2>, T> : std::true_type {};

template <typename T>
struct is_property_key_of<cuda::cluster_size_key<3>, T> : std::true_type {};

template <>
struct is_property_value<cuda::cluster_size_key<1>>
    : is_property_key<cuda::cluster_size_key<1>> {};
template <>
struct is_property_value<cuda::cluster_size_key<2>>
    : is_property_key<cuda::cluster_size_key<2>> {};
template <>
struct is_property_value<cuda::cluster_size_key<3>>
    : is_property_key<cuda::cluster_size_key<3>> {};

template <typename O>
struct is_property_value_of<cuda::cluster_size_key<1>, O>
    : is_property_key_of<cuda::cluster_size_key<1>, O> {};

template <typename O>
struct is_property_value_of<cuda::cluster_size_key<2>, O>
    : is_property_key_of<cuda::cluster_size_key<2>, O> {};

template <typename O>
struct is_property_value_of<cuda::cluster_size_key<3>, O>
    : is_property_key_of<cuda::cluster_size_key<3>, O> {};

namespace detail {
template <typename PropertiesT> constexpr std::size_t getClusterDim() {
  if constexpr (PropertiesT::template has_property<
                    sycl::ext::oneapi::experimental::cuda::cluster_size_key<
                        1>>()) {
    return 1;
  }
  if constexpr (PropertiesT::template has_property<
                    sycl::ext::oneapi::experimental::cuda::cluster_size_key<
                        2>>()) {
    return 2;
  }
  if constexpr (PropertiesT::template has_property<
                    sycl::ext::oneapi::experimental::cuda::cluster_size_key<
                        3>>()) {
    return 3;
  }
  return 0;
}
} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
