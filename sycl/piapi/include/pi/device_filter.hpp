//==---------- pi_device_filter.hpp - PI device filter ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "pi.hpp"

#include <ostream>
#include <vector>

namespace pi {

struct device_filter {
  backend Backend = backend::all;
  pi::device_type DeviceType = pi::device_type::all;
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
  friend std::ostream &operator<<(std::ostream &Out,
                                  const device_filter_list &List);
};

} // namespace pi
