//==--------- compiler.h - Interface between compiler and runtime ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// TODO: as of time of the file creation, it is only used within SYCL library,
// consider moving it to library includes.

/// Target identification strings
///
/// A device type represented by a particular target
/// triple requires specific binary images. We need
/// to map the image type onto the device target triple
///
#define __SYCL_DEVICE_BINARY_TARGET_UNKNOWN "<unknown>"
/// SPIR-V 32-bit image <-> "spir", 32-bit OpenCL device
#define __SYCL_DEVICE_BINARY_TARGET_SPIRV32 "spir"
/// SPIR-V 64-bit image <-> "spir64", 64-bit OpenCL device
#define __SYCL_DEVICE_BINARY_TARGET_SPIRV64 "spir64"
/// Device-specific binary images produced from SPIR-V 64-bit <->
/// various "spir64_*" triples for specific 64-bit OpenCL devices
#define __SYCL_DEVICE_BINARY_TARGET_SPIRV64_X86_64 "spir64_x86_64"
#define __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN "spir64_gen"
#define __SYCL_DEVICE_BINARY_TARGET_SPIRV64_FPGA "spir64_fpga"
/// PTX 64-bit image <-> "nvptx64", 64-bit NVIDIA PTX device
#define __SYCL_DEVICE_BINARY_TARGET_NVPTX64 "nvptx64"
#define __SYCL_DEVICE_BINARY_TARGET_AMDGCN "amdgcn"
#define __SYCL_DEVICE_BINARY_TARGET_NATIVE_CPU "native_cpu"
