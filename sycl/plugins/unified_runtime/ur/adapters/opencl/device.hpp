//===--------- device.hpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include "common.hpp"

#include <sycl/detail/cl.h>
#include <ur/ur.hpp>

namespace cl_adapter {
ur_result_t getDeviceVersion(cl_device_id dev, OCLV::OpenCLVersion &version);

ur_result_t checkDeviceExtensions(cl_device_id dev,
                                  const std::vector<std::string> &exts,
                                  bool &supported);
} // namespace cl_adapter
