//==-------------- allowlist.hpp - SYCL_DEVICE_ALLOWLIST -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__impl/detail/defines_elementary.hpp>
#include <sycl/__impl/detail/pi.hpp>
#include <detail/plugin.hpp>

#include <map>
#include <vector>

#ifdef __SYCL_ENABLE_SYCL121_NAMESPACE
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
#else
namespace __sycl_internal {
inline namespace __v1 {
#endif
namespace detail {

using DeviceDescT = std::map<std::string, std::string>;
using AllowListParsedT = std::vector<DeviceDescT>;

AllowListParsedT parseAllowList(const std::string &AllowListRaw);

bool deviceIsAllowed(const DeviceDescT &DeviceDesc,
                     const AllowListParsedT &AllowListParsed);

void applyAllowList(std::vector<RT::PiDevice> &PiDevices,
                    RT::PiPlatform PiPlatform, const plugin &Plugin);

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
