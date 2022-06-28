//==---------- device_filter.hpp - SYCL device filter descriptor -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/info/info_desc.hpp>

#include <string>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
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
  inline operator std::string() const;
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
  bool containsHost();
  inline operator std::string() const;
};

inline device_filter::operator std::string() const {
  std::string Out{};
  Out += backend_to_string(this->Backend);
  Out += ":";
  switch (this->DeviceType) {
  case info::device_type::host:
    Out += "host";
    break;
  case info::device_type::cpu:
    Out += "cpu";
    break;
  case info::device_type::gpu:
    Out += "gpu";
    break;
  case info::device_type::accelerator:
    Out += "accelerator";
    break;
  case info::device_type::all:
    Out += "*";
    break;
  default:
    Out += "unknown";
    break;
  }

  if (this->HasDeviceNum) {
    Out += ":";
    Out += this->DeviceNum;
  }
  return Out;
}

inline device_filter_list::operator std::string() const {
  std::string Out;
  for (const device_filter &Filter : this->FilterList) {
    Out += Filter;
    Out += ",";
  }
  return Out;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
