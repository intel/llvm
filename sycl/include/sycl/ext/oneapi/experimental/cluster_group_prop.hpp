//==--- cluster_group_prop.hpp --- SYCL extension for cluster group
//------------------==//
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
namespace _V1 {
namespace ext::oneapi::experimental {

template <int Dim>
struct cluster_size : detail::run_time_property_key<detail::ClusterLaunch> {
  cluster_size(const range<Dim> &size) : size(size) {}
  sycl::range<Dim> get_cluster_size() { return size; }
  range<Dim> size;
};

template <int Dim> using cluster_size_key = cluster_size<Dim>;

template <> struct is_property_key<cluster_size_key<1>> : std::true_type {};
template <> struct is_property_key<cluster_size_key<2>> : std::true_type {};
template <> struct is_property_key<cluster_size_key<3>> : std::true_type {};

template <typename T>
struct is_property_key_of<cluster_size_key<1>, T> : std::true_type {};

template <typename T>
struct is_property_key_of<cluster_size_key<2>, T> : std::true_type {};

template <typename T>
struct is_property_key_of<cluster_size_key<3>, T> : std::true_type {};

template<> struct is_property_value<cluster_size_key<1>> : is_property_key<cluster_size_key<1>> {};
template<> struct is_property_value<cluster_size_key<2>> : is_property_key<cluster_size_key<2>> {};
template<> struct is_property_value<cluster_size_key<3>> : is_property_key<cluster_size_key<3>> {};

template <typename O>
struct is_property_value_of<cluster_size_key<1>, O>
    : is_property_key_of<cluster_size_key<1>, O> {};

template <typename O>
struct is_property_value_of<cluster_size_key<2>, O>
    : is_property_key_of<cluster_size_key<2>, O> {};

template <typename O>
struct is_property_value_of<cluster_size_key<3>, O>
    : is_property_key_of<cluster_size_key<3>, O> {};

template <typename PropertiesT, int Dim> constexpr bool hasClusterDim() {
  return PropertiesT::template has_property<
      sycl::ext::oneapi::experimental::cluster_size_key<Dim>>();
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
