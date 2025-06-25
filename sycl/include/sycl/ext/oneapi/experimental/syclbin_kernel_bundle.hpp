//==---- syclbin_kernel_bundle.hpp - SYCLBIN-based kernel_bundle tooling ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

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

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
