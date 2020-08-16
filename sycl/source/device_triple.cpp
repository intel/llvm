//==------------------- device_triple.cpp ----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/device_triple.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <detail/config.hpp>
#include <detail/device_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

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

  // device_type is a required entry
  size_t Pos = 0;
  auto It = std::find_if(
      std::begin(SyclDeviceTypeMap), std::end(SyclDeviceTypeMap),
      [=, &Pos](const std::pair<std::string, info::device_type> &Element) {
        size_t Found = TripleString.find(Element.first, Pos);
        if (Found != std::string::npos) {
          Pos = Found;
          return true;
        }
        return false;
      });
  if (It == SyclDeviceTypeMap.end())
    throw cl::sycl::invalid_parameter_error(
        "Invalid device_type. Valid values are host/cpu/gpu/acc/*",
        PI_INVALID_VALUE);

  DeviceType = It->second;
  // initialize optional entries with default values
  if (DeviceType == info::device_type::all) {
    Backend = backend::all;
  } else if (DeviceType == info::device_type::gpu) {
    Backend = backend::level_zero;
  } else {
    Backend = backend::opencl;
  }
  DeviceNum = DEVICE_NUM_UNSPECIFIED;

  // update the optional 2nd entry, backend
  size_t ColonPos = TripleString.find(":", Pos);
  if (ColonPos != std::string::npos) {
    Pos = ColonPos + 1;
    auto It =
        std::find_if(std::begin(SyclBeMap), std::end(SyclBeMap),
                     [=, &Pos](const std::pair<std::string, backend> &Element) {
                       size_t Found = TripleString.find(Element.first, Pos);
                       if (Found != std::string::npos) {
                         Pos = Found;
                         return true;
                       }
                       return false;
                     });
    if (It == SyclBeMap.end())
      throw cl::sycl::invalid_parameter_error(
          "Invalid backend. Valid values are opencl/level_zero/cuda/*",
          PI_INVALID_VALUE);
    Backend = It->second;
  }

  // update the optional 3rd entry, device number
  ColonPos = TripleString.find(":", Pos);
  if (ColonPos != std::string::npos && (ColonPos + 1) < TripleString.size()) {
    try {
      DeviceNum = stoi(TripleString.substr(ColonPos + 1));
    } catch (...) {
      throw cl::sycl::invalid_parameter_error(
          "Invalid device number. An integer is needed.", PI_INVALID_VALUE);
    }
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

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
