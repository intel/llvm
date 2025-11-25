//==---- syclbin_kernel_bundle.hpp - SYCLBIN-based kernel_bundle tooling ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/experimental/syclbin_properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/kernel_bundle.hpp>

#include <fstream>
#include <string>

#if __has_include(<filesystem>)
#include <filesystem>
#endif

#if __has_include(<span>)
#include <span>
#endif

namespace sycl {
inline namespace _V1 {

namespace detail {
__SYCL_EXPORT std::shared_ptr<detail::kernel_bundle_impl>
link_impl(const kernel_bundle<bundle_state::object> *ObjectBundles,
          size_t NumObjectBundles, const std::vector<device> &Devs,
          bool FastLink);

template <
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t,
    typename = std::enable_if_t<
        ext::oneapi::experimental::detail::all_are_properties_of_v<
            ext::oneapi::experimental::detail::link_props, PropertyListT>>>
kernel_bundle<bundle_state::executable>
link_common(const kernel_bundle<bundle_state::object> *ObjectBundles,
            size_t NumObjectBundles, const std::vector<device> &Devs,
            PropertyListT Props = {}) {
  std::vector<device> UniqueDevices = removeDuplicateDevices(Devs);

  bool UseFastLink = [&]() {
    if constexpr (Props.template has_property<
                      ext::oneapi::experimental::fast_link>())
      return Props.template get_property<ext::oneapi::experimental::fast_link>()
          .value;
    return false;
  }();

  KernelBundleImplPtr Impl =
      link_impl(ObjectBundles, NumObjectBundles, UniqueDevices, UseFastLink);
  return createSyclObjFromImpl<kernel_bundle<bundle_state::executable>>(
      std::move(Impl));
}
} // namespace detail

namespace ext::oneapi::experimental {

template <bundle_state State, typename PropertyListT = empty_properties_t>
std::enable_if_t<State != bundle_state::ext_oneapi_source, kernel_bundle<State>>
get_kernel_bundle(const context &Ctxt, const std::vector<device> &Devs,
                  const sycl::span<char> &Bytes, PropertyListT = {}) {
  std::vector<device> UniqueDevices =
      sycl::detail::removeDuplicateDevices(Devs);

  sycl::detail::KernelBundleImplPtr Impl =
      sycl::detail::get_kernel_bundle_impl(Ctxt, UniqueDevices, Bytes, State);
  return sycl::detail::createSyclObjFromImpl<kernel_bundle<State>>(Impl);
}

#if __cpp_lib_span
template <bundle_state State, typename PropertyListT = empty_properties_t>
std::enable_if_t<State != bundle_state::ext_oneapi_source, kernel_bundle<State>>
get_kernel_bundle(const context &Ctxt, const std::vector<device> &Devs,
                  const std::span<char> &Bytes, PropertyListT Props = {}) {
  return experimental::get_kernel_bundle(
      Ctxt, Devs, sycl::span<char>(Bytes.data(), Bytes.size()), Props);
}
#endif

#if __cpp_lib_filesystem
template <bundle_state State, typename PropertyListT = empty_properties_t>
std::enable_if_t<State != bundle_state::ext_oneapi_source, kernel_bundle<State>>
get_kernel_bundle(const context &Ctxt, const std::vector<device> &Devs,
                  const std::filesystem::path &Filename,
                  PropertyListT Props = {}) {
  std::vector<char> RawSYCLBINData;
  {
    std::ifstream FileStream{Filename, std::ios::binary};
    if (!FileStream.is_open())
      throw std::ios_base::failure("Failed to open SYCLBIN file: " +
                                   Filename.string());
    RawSYCLBINData =
        std::vector<char>{std::istreambuf_iterator<char>(FileStream),
                          std::istreambuf_iterator<char>()};
  }
  return experimental::get_kernel_bundle<State>(
      Ctxt, Devs, sycl::span<char>{RawSYCLBINData}, Props);
}

template <bundle_state State, typename PropertyListT = empty_properties_t>
std::enable_if_t<State != bundle_state::ext_oneapi_source, kernel_bundle<State>>
get_kernel_bundle(const context &Ctxt, const std::filesystem::path &Filename,
                  PropertyListT Props = {}) {
  return experimental::get_kernel_bundle<State>(Ctxt, Ctxt.get_devices(),
                                                Filename, Props);
}
#endif

template <typename PropertyListT = empty_properties_t,
          typename = std::enable_if_t<detail::all_are_properties_of_v<
              sycl::detail::link_props, PropertyListT>>>
kernel_bundle<bundle_state::executable>
link(const std::vector<kernel_bundle<bundle_state::object>> &ObjectBundles,
     const std::vector<device> &Devs, PropertyListT Props = {}) {
  return sycl::detail::link_common(ObjectBundles.data(), ObjectBundles.size(),
                                   Devs, Props);
}

template <typename PropertyListT = empty_properties_t,
          typename = std::enable_if_t<detail::all_are_properties_of_v<
              sycl::detail::link_props, PropertyListT>>>
kernel_bundle<bundle_state::executable>
link(const kernel_bundle<bundle_state::object> &ObjectBundle,
     const std::vector<device> &Devs, PropertyListT Props = {}) {
  return sycl::detail::link_common(&ObjectBundle, 1, Devs, Props);
}

template <typename PropertyListT = empty_properties_t,
          typename = std::enable_if_t<detail::all_are_properties_of_v<
              sycl::detail::link_props, PropertyListT>>>
kernel_bundle<bundle_state::executable>
link(const std::vector<kernel_bundle<bundle_state::object>> &ObjectBundles,
     PropertyListT Props = {}) {
  std::vector<sycl::device> IntersectDevices =
      sycl::detail::find_device_intersection(ObjectBundles);
  return link(ObjectBundles, IntersectDevices, Props);
}

template <typename PropertyListT = empty_properties_t,
          typename = std::enable_if_t<detail::all_are_properties_of_v<
              sycl::detail::link_props, PropertyListT>>>
kernel_bundle<bundle_state::executable>
link(const kernel_bundle<bundle_state::object> &ObjectBundle,
     PropertyListT Props = {}) {
  return link(std::vector<kernel_bundle<bundle_state::object>>{ObjectBundle},
              ObjectBundle.get_devices(), Props);
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
