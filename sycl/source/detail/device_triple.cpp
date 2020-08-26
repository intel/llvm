//==------------------- device_triple.cpp ----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/device_triple.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <detail/config.hpp>
#include <detail/device_impl.hpp>

#include <cstring>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

device_triple::device_triple(std::string &TripleString) {
  const std::array<std::pair<std::string, info::device_type>, 5>
      SyclDeviceTypeMap = {{{"host", info::device_type::host},
                            {"cpu", info::device_type::cpu},
                            {"gpu", info::device_type::gpu},
                            {"acc", info::device_type::accelerator},
                            {"*", info::device_type::all}}};
  const std::array<std::pair<std::string, backend>, 4> SyclBeMap = {
      {{"opencl", backend::opencl},
       {"level_zero", backend::level_zero},
       {"cuda", backend::cuda},
       {"*", backend::all}}};

  // handle the optional 1st entry, backend
  size_t Cursor = 0;
  size_t ColonPos = TripleString.find(":", Cursor);
  auto It = std::find_if(
      std::begin(SyclBeMap), std::end(SyclBeMap),
      [=, &Cursor](const std::pair<std::string, backend> &Element) {
        size_t Found = TripleString.find(Element.first, Cursor);
        if (Found != std::string::npos) {
          Cursor = Found;
          return true;
        }
        return false;
      });
  if (It == SyclBeMap.end()) {
    Backend = backend::all;
  } else {
    Backend = It->second;
    if (ColonPos != std::string::npos) {
      Cursor = ColonPos + 1;
    } else {
      Cursor = Cursor + It->first.size();
    }
  }

  // handle the optional 2nd entry, device type
  auto Iter = std::find_if(
      std::begin(SyclDeviceTypeMap), std::end(SyclDeviceTypeMap),
      [=, &Cursor](const std::pair<std::string, info::device_type> &Element) {
        size_t Found = TripleString.find(Element.first, Cursor);
        if (Found != std::string::npos) {
          Cursor = Found;
          return true;
        }
        return false;
      });
  if (Iter == SyclDeviceTypeMap.end()) {
    DeviceType = info::device_type::all;
  } else {
    DeviceType = Iter->second;
    ColonPos = TripleString.find(":", Cursor);
    if (ColonPos != std::string::npos) {
      Cursor = ColonPos + 1;
    } else {
      Cursor = Cursor + Iter->first.size();
    }
  }

  // handle the optional 3rd entry, device number
  if (Cursor < TripleString.size()) {
    try {
      DeviceNum = stoi(TripleString.substr(ColonPos + 1));
    } catch (...) {
      char message[100];
      strcpy(message, "Invalid device triple: ");
      std::strcat(message, TripleString.c_str());
      std::strcat(message,
                  "\nPossible backend values are {opencl,level_zero,cuda,*}.");
      std::strcat(message, "\nPossible device types are {host,cpu,gpu,acc,*}.");
      std::strcat(message,
                  "\nDevice number should be an non-negative integer.\n");
      throw cl::sycl::invalid_parameter_error(message, PI_INVALID_VALUE);
    }
  } else {
    DeviceNum = DEVICE_NUM_UNSPECIFIED;
  }
}

device_triple_list::device_triple_list(std::string &TripleString) {
  std::transform(TripleString.begin(), TripleString.end(), TripleString.begin(),
                 ::tolower);
  size_t Pos = 0;
  while (Pos < TripleString.size()) {
    size_t CommaPos = TripleString.find(",", Pos);
    if (CommaPos == std::string::npos) {
      CommaPos = TripleString.size();
    }
    std::string SubString = TripleString.substr(Pos, CommaPos - Pos);
    TripleList.push_back(device_triple(SubString));
    Pos = CommaPos + 1;
  }
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
