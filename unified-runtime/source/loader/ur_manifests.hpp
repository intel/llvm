/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions.
 * See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_manifests.hpp
 *
 */

// Auto-generated file, do not edit.

#pragma once

#include <string>
#include <vector>

#include "ur_util.hpp"
#include <ur_api.h>

namespace ur_loader {
struct ur_adapter_manifest {
  std::string name;
  std::string library;
  ur_backend_t backend;
  std::vector<ur_device_type_t> device_types;
};

const std::vector<ur_adapter_manifest> ur_adapter_manifests = {
    {"opencl",
     MAKE_LIBRARY_NAME("ur_adapter_opencl", "0"),
     UR_BACKEND_OPENCL,
     {
         UR_DEVICE_TYPE_CPU,
         UR_DEVICE_TYPE_GPU,
         UR_DEVICE_TYPE_FPGA,
         UR_DEVICE_TYPE_MCA,
         UR_DEVICE_TYPE_VPU,
     }},
    {"cuda",
     MAKE_LIBRARY_NAME("ur_adapter_cuda", "0"),
     UR_BACKEND_CUDA,
     {
         UR_DEVICE_TYPE_GPU,
     }},
    {"hip",
     MAKE_LIBRARY_NAME("ur_adapter_hip", "0"),
     UR_BACKEND_HIP,
     {
         UR_DEVICE_TYPE_GPU,
     }},
    {"level_zero",
     MAKE_LIBRARY_NAME("ur_adapter_level_zero", "0"),
     UR_BACKEND_LEVEL_ZERO,
     {
         UR_DEVICE_TYPE_CPU,
         UR_DEVICE_TYPE_GPU,
         UR_DEVICE_TYPE_FPGA,
         UR_DEVICE_TYPE_MCA,
         UR_DEVICE_TYPE_VPU,
     }},
    {"level_zero_v2",
     MAKE_LIBRARY_NAME("ur_adapter_level_zero_v2", "0"),
     UR_BACKEND_LEVEL_ZERO,
     {
         UR_DEVICE_TYPE_CPU,
         UR_DEVICE_TYPE_GPU,
         UR_DEVICE_TYPE_FPGA,
         UR_DEVICE_TYPE_MCA,
         UR_DEVICE_TYPE_VPU,
     }},
    {"native_cpu",
     MAKE_LIBRARY_NAME("ur_adapter_native_cpu", "0"),
     UR_BACKEND_NATIVE_CPU,
     {
         UR_DEVICE_TYPE_CPU,
     }},
};
} // namespace ur_loader
