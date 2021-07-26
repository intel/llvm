//===----- SYCLDeviceLibReqMask.h - get SYCL devicelib required Info -----=-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass goes through input module's function list to detect all SYCL
// devicelib functions invoked. Each devicelib function invoked is included in
// one 'fallback' SPIR-V library loaded by SYCL runtime. After scanning all
// functions in input module, a mask telling which SPIR-V libraries are needed
// by input module indeed will be returned. This mask will be saved and used by
// SYCL runtime later.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Pass.h"
#include <unordered_map>
using namespace llvm;

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
};

using SYCLDeviceLibFuncMap = std::unordered_map<std::string, DeviceLibExt>;
class SYCLDeviceLibReqMaskPass : public ModulePass {
public:
  static char ID;
  SYCLDeviceLibReqMaskPass() : ModulePass(ID) { MReqMask = 0; }
  bool runOnModule(Module &M) override;
  uint32_t getSYCLDeviceLibReqMask() { return MReqMask; }

private:
  uint32_t MReqMask;
};
