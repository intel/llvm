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
#include <sycl/detail/iostream_proxy.hpp>
#include <sycl/info/info_desc.hpp>

#include <string>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

struct device_filter {
  backend Backend = backend::all;
  info::device_type DeviceType = info::device_type::all;
  int DeviceNum = 0;
  bool HasBackend = false;
  bool HasDeviceType = false;
  bool HasDeviceNum = false;
  int MatchesSeen = 0;

  device_filter(){};
  device_filter(const std::string &FilterString);
  friend std::ostream &operator<<(std::ostream &Out,
                                  const device_filter &Filter);
};

class device_filter_list {
  std::vector<device_filter> FilterList;

public:
  device_filter_list() {}
  device_filter_list(const std::string &FilterString);
  device_filter_list(device_filter &Filter);
  void addFilter(device_filter &Filter);
  std::vector<device_filter> &get() { return FilterList; }
  bool backendCompatible(backend Backend);
  bool deviceTypeCompatible(info::device_type DeviceType);
  bool deviceNumberCompatible(int DeviceNum);
  friend std::ostream &operator<<(std::ostream &Out,
                                  const device_filter_list &List);
};

inline std::ostream &operator<<(std::ostream &Out,
                                const device_filter &Filter) {
  Out << Filter.Backend << ":";
  if (Filter.DeviceType == info::device_type::host) {
    Out << "host";
  } else if (Filter.DeviceType == info::device_type::cpu) {
    Out << "cpu";
  } else if (Filter.DeviceType == info::device_type::gpu) {
    Out << "gpu";
  } else if (Filter.DeviceType == info::device_type::accelerator) {
    Out << "accelerator";
  } else if (Filter.DeviceType == info::device_type::all) {
    Out << "*";
  } else {
    Out << "unknown";
  }
  if (Filter.HasDeviceNum) {
    Out << ":" << Filter.DeviceNum;
  }
  return Out;
}

inline std::ostream &operator<<(std::ostream &Out,
                                const device_filter_list &List) {
  for (const device_filter &Filter : List.FilterList) {
    Out << Filter;
    Out << ",";
  }
  return Out;
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
