//===----- SYCLRequiredDeviceLibs.h - get SYCL devicelib required Info ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This function goes through input module's function list to detect all SYCL
// devicelib functions invoked. Each devicelib function invoked is included in
// one 'fallback' SPIR-V library loaded by SYCL runtime. After scanning all
// functions in input module, a mask telling which SPIR-V libraries are needed
// by input module indeed will be returned. This mask will be saved and used by
// SYCL runtime later.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace llvm {

class Module;

// DeviceLibExt is shared between sycl-post-link tool and sycl runtime.
// If any change is made here, need to sync with DeviceLibExt definition
// in sycl/source/detail/program_manager/program_manager.hpp
enum class DeviceLibExt : std::uint32_t {
  cl_intel_devicelib_assert,
  cl_intel_devicelib_math,
  cl_intel_devicelib_math_fp64,
  cl_intel_devicelib_complex,
  cl_intel_devicelib_complex_fp64,
  cl_intel_devicelib_cstring,
  cl_intel_devicelib_imf,
  cl_intel_devicelib_imf_fp64,
  cl_intel_devicelib_imf_bf16,
  cl_intel_devicelib_bfloat16,
};

enum class DeviceLibIsNative : std::uint32_t { Yes, No, Ignore };

struct SYCLDeviceLibSPVMeta {
  DeviceLibExt SPVExt;
  const char *SPVFileName;
  DeviceLibIsNative IsNative;
};

struct SYCLDeviceLibSPVBinary {
  typedef uint8_t value_type;
  value_type *SPVRawBytes;
  size_t SPVBytesNum;
  SYCLDeviceLibSPVBinary(value_type *RawB, size_t BNum) {
    SPVRawBytes = RawB;
    SPVBytesNum = BNum;
  }
  value_type *data() const { return SPVRawBytes; }
  size_t size() const { return SPVBytesNum; }
};

void getRequiredSYCLDeviceLibs(const Module &M,
                               SmallVector<SYCLDeviceLibSPVMeta, 16> &ReqLibs);

} // namespace llvm
