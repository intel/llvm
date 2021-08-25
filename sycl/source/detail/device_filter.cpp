//==------------------- device_filter.cpp ----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/device_filter.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <detail/config.hpp>
#include <detail/device_impl.hpp>

#include <cstring>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

device_filter::device_filter(const std::string &FilterString) {
  size_t Cursor = 0;
  size_t ColonPos = 0;
  auto findElement = [&](auto Element) {
    size_t Found = FilterString.find(Element.first, Cursor);
    if (Found == std::string::npos)
      return false;
    Cursor = Found;
    return true;
  };

  // Handle the optional 1st field of the filter, backend
  // Check if the first entry matches with a known backend type
  auto It = std::find_if(std::begin(getSyclBeMap()), std::end(getSyclBeMap()),
                         findElement);
  // If no match is found, set the backend type backend::all
  // which actually means 'any backend' will be a match.
  if (It == getSyclBeMap().end())
    Backend = backend::all;
  else {
    Backend = It->second;
    ColonPos = FilterString.find(":", Cursor);
    if (ColonPos != std::string::npos)
      Cursor = ColonPos + 1;
    else
      Cursor = Cursor + It->first.size();
  }
  // Handle the optional 2nd field of the filter - device type.
  // Check if the 2nd entry matches with any known device type.
  if (Cursor >= FilterString.size()) {
    DeviceType = info::device_type::all;
  } else {
    auto Iter = std::find_if(std::begin(getSyclDeviceTypeMap()),
                             std::end(getSyclDeviceTypeMap()), findElement);
    // If no match is found, set device_type 'all',
    // which actually means 'any device_type' will be a match.
    if (Iter == getSyclDeviceTypeMap().end())
      DeviceType = info::device_type::all;
    else {
      DeviceType = Iter->second;
      ColonPos = FilterString.find(":", Cursor);
      if (ColonPos != std::string::npos)
        Cursor = ColonPos + 1;
      else
        Cursor = Cursor + Iter->first.size();
    }
  }

  // Handle the optional 3rd field of the filter, device number
  // Try to convert the remaining string to an integer.
  // If succeessful, the converted integer is the desired device num.
  if (Cursor < FilterString.size()) {
    try {
      DeviceNum = stoi(FilterString.substr(Cursor));
      HasDeviceNum = true;
    } catch (...) {
      std::string Message =
          std::string("Invalid device filter: ") + FilterString +
          "\nPossible backend values are "
          "{host,opencl,level_zero,cuda,rocm,*}.\n"
          "Possible device types are {host,cpu,gpu,acc,*}.\n"
          "Device number should be an non-negative integer.\n";
      throw cl::sycl::invalid_parameter_error(Message, PI_INVALID_VALUE);
    }
  }
}

device_filter_list::device_filter_list(const std::string &FilterStr) {
  // First, change the string in all lowercase.
  // This means we allow the user to use both uppercase and lowercase strings.
  std::string FilterString = FilterStr;
  std::transform(FilterString.begin(), FilterString.end(), FilterString.begin(),
                 ::tolower);
  // SYCL_DEVICE_FILTER can set multiple filters separated by commas.
  // convert each filter triple string into an istance of device_filter class.
  size_t Pos = 0;
  while (Pos < FilterString.size()) {
    size_t CommaPos = FilterString.find(",", Pos);
    if (CommaPos == std::string::npos) {
      CommaPos = FilterString.size();
    }
    std::string SubString = FilterString.substr(Pos, CommaPos - Pos);
    FilterList.push_back(device_filter(SubString));
    Pos = CommaPos + 1;
  }
}

device_filter_list::device_filter_list(device_filter &Filter) {
  FilterList.push_back(Filter);
}

void device_filter_list::addFilter(device_filter &Filter) {
  FilterList.push_back(Filter);
}

// Backend is compatible with the SYCL_DEVICE_FILTER in the following cases.
// 1. Filter backend is '*' which means ANY backend.
// 2. Filter backend match exactly with the given 'Backend'
bool device_filter_list::backendCompatible(backend Backend) {
  for (const device_filter &Filter : FilterList) {
    backend FilterBackend = Filter.Backend;
    if (FilterBackend == Backend || FilterBackend == backend::all)
      return true;
  }
  return false;
}

bool device_filter_list::deviceTypeCompatible(info::device_type DeviceType) {
  for (const device_filter &Filter : FilterList) {
    info::device_type FilterDevType = Filter.DeviceType;
    if (FilterDevType == DeviceType || FilterDevType == info::device_type::all)
      return true;
  }
  return false;
}

bool device_filter_list::deviceNumberCompatible(int DeviceNum) {
  for (const device_filter &Filter : FilterList) {
    int FilterDevNum = Filter.DeviceNum;
    if (!Filter.HasDeviceNum || FilterDevNum == DeviceNum)
      return true;
  }
  return false;
}

bool device_filter_list::containsHost() {
  for (const device_filter &Filter : FilterList) {
    if (Filter.Backend == backend::host || Filter.Backend == backend::all)
      if (Filter.DeviceType == info::device_type::host ||
          Filter.DeviceType == info::device_type::all)
        // SYCL RT never creates more than one HOST device.
        // All device numbers other than 0 are rejected.
        if (!Filter.HasDeviceNum || Filter.DeviceNum == 0)
          return true;
  }
  return false;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
