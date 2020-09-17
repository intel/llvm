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
  const std::array<std::pair<std::string, info::device_type>, 5>
      SyclDeviceTypeMap = {{{"host", info::device_type::host},
                            {"cpu", info::device_type::cpu},
                            {"gpu", info::device_type::gpu},
                            {"acc", info::device_type::accelerator},
                            {"*", info::device_type::all}}};
  const std::array<std::pair<std::string, backend>, 5> SyclBeMap = {
      {{"host", backend::host},
       {"opencl", backend::opencl},
       {"level_zero", backend::level_zero},
       {"cuda", backend::cuda},
       {"*", backend::all}}};

  size_t Cursor = 0;
  size_t ColonPos = 0;
  auto findElement = [&](auto Element) {
    size_t Found = FilterString.find(Element.first, Cursor);
    if (Found == std::string::npos)
      return false;
    Cursor = Found;
    return true;
  };
  auto selectElement = [&](auto It, auto Map, auto EltIfNotFound) {
    if (It == Map.end())
      return EltIfNotFound;
    ColonPos = FilterString.find(":", Cursor);
    if (ColonPos != std::string::npos)
      Cursor = ColonPos + 1;
    else
      Cursor = Cursor + It->first.size();
    return It->second;
  };

  // Handle the optional 1st field of the filter, backend
  // Check if the first entry matches with a known backend type
  auto It =
      std::find_if(std::begin(SyclBeMap), std::end(SyclBeMap), findElement);
  // If no match is found, set the backend type backend::all
  // which actually means 'any backend' will be a match.
  Backend = selectElement(It, SyclBeMap, backend::all);

  // Handle the optional 2nd field of the filter - device type.
  // Check if the 2nd entry matches with any known device type.
  if (Cursor >= FilterString.size()) {
    DeviceType = info::device_type::all;
  } else {
    auto Iter = std::find_if(std::begin(SyclDeviceTypeMap),
                             std::end(SyclDeviceTypeMap), findElement);
    // If no match is found, set device_type 'all',
    // which actually means 'any device_type' will be a match.
    DeviceType = selectElement(Iter, SyclDeviceTypeMap, info::device_type::all);
  }

  // Handle the optional 3rd field of the filter, device number
  // Try to convert the remaining string to an integer.
  // If succeessful, the converted integer is the desired device num.
  if (Cursor < FilterString.size()) {
    try {
      DeviceNum = stoi(FilterString.substr(ColonPos + 1));
      HasDeviceNum = true;
    } catch (...) {
      std::string Message =
          std::string("Invalid device filter: ") + FilterString +
          "\nPossible backend values are {host,opencl,level_zero,cuda,*}.\n"
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

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
