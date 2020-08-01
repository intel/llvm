//==-------------- backend_types.hpp - SYCL backend types ------------------==//
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

#include <fstream>
#include <iostream>
#include <istream>
#include <string>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

#define DEVICE_NUM_UNSPECIFIED -1

struct triple {
  info::device_type DeviceType;
  backend Backend;
  int32_t DeviceNum;
};

inline std::ostream &operator<<(std::ostream &Out, triple Trp) {
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
  Out << std::string(":");
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
  }
  if (Trp.DeviceNum != DEVICE_NUM_UNSPECIFIED) {
    Out << std::string(":") << Trp.DeviceNum;
  }
  return Out;
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
