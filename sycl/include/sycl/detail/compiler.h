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

/// Device binary image property set names recognized by the SYCL runtime.
/// Name must be consistent with
/// PropertySetRegistry::SYCL_SPECIALIZATION_CONSTANTS defined in
/// PropertySetIO.h
#define __SYCL_PROPERTY_SET_SPEC_CONST_MAP "SYCL/specialization constants"
/// PropertySetRegistry::SYCL_SPEC_CONSTANTS_DEFAULT_VALUES defined in
/// PropertySetIO.h
#define __SYCL_PROPERTY_SET_SPEC_CONST_DEFAULT_VALUES_MAP                   \
  "SYCL/specialization constants default values"
/// PropertySetRegistry::SYCL_DEVICELIB_REQ_MASK defined in PropertySetIO.h
#define __SYCL_PROPERTY_SET_DEVICELIB_REQ_MASK "SYCL/devicelib req mask"
/// PropertySetRegistry::SYCL_KERNEL_PARAM_OPT_INFO defined in PropertySetIO.h
#define __SYCL_PROPERTY_SET_KERNEL_PARAM_OPT_INFO "SYCL/kernel param opt"
/// PropertySetRegistry::SYCL_KERNEL_PROGRAM_METADATA defined in PropertySetIO.h
#define __SYCL_PROPERTY_SET_PROGRAM_METADATA "SYCL/program metadata"
/// PropertySetRegistry::SYCL_MISC_PROP defined in PropertySetIO.h
#define __SYCL_PROPERTY_SET_SYCL_MISC_PROP "SYCL/misc properties"
/// PropertySetRegistry::SYCL_ASSERT_USED defined in PropertySetIO.h
#define __SYCL_PROPERTY_SET_SYCL_ASSERT_USED "SYCL/assert used"
/// PropertySetRegistry::SYCL_EXPORTED_SYMBOLS defined in PropertySetIO.h
#define __SYCL_PROPERTY_SET_SYCL_EXPORTED_SYMBOLS "SYCL/exported symbols"
/// PropertySetRegistry::SYCL_DEVICE_GLOBALS defined in PropertySetIO.h
#define __SYCL_PROPERTY_SET_SYCL_DEVICE_GLOBALS "SYCL/device globals"
/// PropertySetRegistry::SYCL_DEVICE_REQUIREMENTS defined in PropertySetIO.h
#define __SYCL_PROPERTY_SET_SYCL_DEVICE_REQUIREMENTS                        \
  "SYCL/device requirements"
/// PropertySetRegistry::SYCL_HOST_PIPES defined in PropertySetIO.h
#define __SYCL_PROPERTY_SET_SYCL_HOST_PIPES "SYCL/host pipes"
/// PropertySetRegistry::SYCL_VIRTUAL_FUNCTIONS defined in PropertySetIO.h
#define __SYCL_PROPERTY_SET_SYCL_VIRTUAL_FUNCTIONS "SYCL/virtual functions"

/// Program metadata tags recognized by the PI backends. For kernels the tag
/// must appear after the kernel name.
#define __SYCL_PROGRAM_METADATA_TAG_REQD_WORK_GROUP_SIZE                    \
  "@reqd_work_group_size"
#define __SYCL_PROGRAM_METADATA_GLOBAL_ID_MAPPING "@global_id_mapping"

#define __SYCL_PROGRAM_METADATA_TAG_NEED_FINALIZATION "Requires finalization"
