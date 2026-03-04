//===--------- platform.hpp - CUDA Adapter --------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef UR_CUDA_PLATFORM_HPP_INCLUDED
#define UR_CUDA_PLATFORM_HPP_INCLUDED

#include "common.hpp"
#include "device.hpp"
#include <ur/ur.hpp>

#include <memory>
#include <vector>

struct ur_platform_handle_t_ : ur::cuda::handle_base {
  std::vector<std::unique_ptr<ur_device_handle_t_>> Devices;
};

#endif // UR_CUDA_PLATFORM_HPP_INCLUDED
