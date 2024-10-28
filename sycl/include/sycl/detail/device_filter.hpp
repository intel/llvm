//==---------- device_filter.hpp - SYCL device filter descriptor -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/backend_types.hpp>
#include <sycl/detail/defines.hpp>
#include <sycl/info/info_desc.hpp>

#include <optional>
#include <ostream>
#include <string>

namespace sycl {
inline namespace _V1 {
namespace detail {

// ---------------------------------------
// ONEAPI_DEVICE_SELECTOR support

template <typename T>
std::ostream &operator<<(std::ostream &os, std::optional<T> const &opt) {
  return opt ? os << opt.value() : os << "not set ";
}

// the ONEAPI_DEVICE_SELECTOR string gets broken down into these targets
// will will match devices. If the target is negative, such as !opencl:*
// then matching devices will not be made available to the user.
struct ods_target {
public:
  std::optional<backend> Backend;
  std::optional<info::device_type> DeviceType;

  bool HasDeviceWildCard = false;
  std::optional<int> DeviceNum;

  bool HasSubDeviceWildCard = false;
  std::optional<unsigned> SubDeviceNum;

  bool HasSubSubDeviceWildCard = false; // two levels of sub-devices.
  std::optional<unsigned> SubSubDeviceNum;

  bool IsNegativeTarget = false; // used to represent negative filters.
  // used in filter selector to keep count of the number of devices with
  // the same Backend and DeviceType.
  int MatchesSeen = 0;

  ods_target(backend be) { Backend = be; };
  ods_target(){};
  friend std::ostream &operator<<(std::ostream &Out, const ods_target &Target);

#if __cplusplus >= 202002L
  bool operator==(const ods_target &Other) const = default;
#else
  bool operator==(const ods_target &Other) const {
    return Backend == Other.Backend && DeviceType == Other.DeviceType &&
           HasDeviceWildCard == Other.HasDeviceWildCard &&
           DeviceNum == Other.DeviceNum &&
           HasSubDeviceWildCard == Other.HasSubDeviceWildCard &&
           HasSubSubDeviceWildCard == Other.HasSubSubDeviceWildCard &&
           SubSubDeviceNum == Other.SubSubDeviceNum &&
           IsNegativeTarget == Other.IsNegativeTarget &&
           MatchesSeen == Other.MatchesSeen;
  }
#endif
};

class ods_target_list {
  std::vector<ods_target> TargetList;

public:
  ods_target_list() {}
  ods_target_list(const std::string &FilterString);
  std::vector<ods_target> &get() { return TargetList; }
  bool containsHost();
  bool backendCompatible(backend Backend);
};

std::ostream &operator<<(std::ostream &Out, const ods_target &Target);
std::vector<ods_target> Parse_ONEAPI_DEVICE_SELECTOR(const std::string &envStr);

} // namespace detail
} // namespace _V1
} // namespace sycl
