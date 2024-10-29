//==-------------- allowlist.hpp - SYCL_DEVICE_ALLOWLIST -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/adapter.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/ur.hpp>

#include <map>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

using DeviceDescT = std::map<std::string, std::string>;
using AllowListParsedT = std::vector<DeviceDescT>;

AllowListParsedT parseAllowList(const std::string &AllowListRaw);

bool deviceIsAllowed(const DeviceDescT &DeviceDesc,
                     const AllowListParsedT &AllowListParsed);

void applyAllowList(std::vector<ur_device_handle_t> &UrDevices,
                    ur_platform_handle_t UrPlatform, const AdapterPtr &Adapter);

} // namespace detail
} // namespace _V1
} // namespace sycl
