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
#include <pi/device_filter.hpp>

#include <iostream>
#include <string>

namespace pi {

inline std::ostream &operator<<(std::ostream &Out,
                                const device_filter &Filter) {
  namespace info = cl::sycl::info;
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

} // namespace pi

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

using pi::device_filter;
using pi::device_filter_list;

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
