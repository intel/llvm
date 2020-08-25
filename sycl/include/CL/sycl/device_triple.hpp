//==---------- device_triple.hpp - SYCL device triple descriptor -----------==//
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

#include <iostream>
#include <string>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

class device_triple {
  backend Backend;
  info::device_type DeviceType;
  int32_t DeviceNum;
  static constexpr int DEVICE_NUM_UNSPECIFIED = -1;

public:
  device_triple(std::string &TripleString);
  backend getBackend() const { return Backend; }
  info::device_type getDeviceType() const { return DeviceType; }
  int32_t getDeviceNum() const { return DeviceNum; }
  friend std::ostream &operator<<(std::ostream &Out, const device_triple &Trp);
};

class device_triple_list {
  std::vector<device_triple> TripleList;

public:
  device_triple_list(std::string &TripleString);
  device_triple_list(device_triple &Trp);
  std::vector<device_triple> &get() { return TripleList; }
  friend std::ostream &operator<<(std::ostream &Out,
                                  const device_triple_list &List);
};

inline std::ostream &operator<<(std::ostream &Out, const device_triple &Trp) {
  switch (Trp.Backend) {
  case backend::host:
    Out << std::string("host");
    break;
  case backend::opencl:
    Out << std::string("opencl");
    break;
  case backend::level_zero:
    Out << std::string("level-zero");
    break;
  case backend::cuda:
    Out << std::string("cuda");
    break;
  case backend::all:
    Out << std::string("*");
  }
  Out << std::string(":");
  if (Trp.DeviceType == info::device_type::host) {
    Out << std::string("host");
  } else if (Trp.DeviceType == info::device_type::cpu) {
    Out << std::string("cpu");
  } else if (Trp.DeviceType == info::device_type::gpu) {
    Out << std::string("gpu");
  } else if (Trp.DeviceType == info::device_type::accelerator) {
    Out << std::string("acceclerator");
  } else if (Trp.DeviceType == info::device_type::all) {
    Out << std::string("*");
  }
  if (Trp.DeviceNum != Trp.DEVICE_NUM_UNSPECIFIED) {
    Out << std::string(":") << Trp.DeviceNum;
  }
  return Out;
}

inline std::ostream &operator<<(std::ostream &Out,
                                const device_triple_list &List) {
  for (const device_triple &Trp : List.TripleList) {
    Out << Trp;
    Out << ",";
  }
  return Out;
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
