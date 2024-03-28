//==--- DeviceConfigFile.hpp - Device Config File for SYCL  ------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <map>
#include <string>
#include <vector>

namespace llvm {
class StringRef;
}

namespace DeviceConfigFile {

// This struct is used in DeviceConfigFile.td. Both the fields and the name of
// this struct must match the definition in DeviceConfigFile.td. Thus, any
// modification to this struct in this file must be mirrored in
// DeviceConfigFile.td.
struct TargetInfo {
  bool maySupportOtherAspects;
  std::vector<llvm::StringRef> aspects;
  std::vector<unsigned> subGroupSizes;
  std::string aotToolchain;
  std::string aotToolchainOptions;
};
using TargetTable_t = std::map<std::string, TargetInfo>;

// AspectTable maps aspect names to their corresponding enum value
// defined in the SYCL headers
using AspectTable_t = std::map<std::string, int>;

#define GET_TargetTable_IMPL
#define GET_AspectTable_IMPL
#include "llvm/SYCLLowerIR/DeviceConfigFile.inc"
#undef GET_TargetTable_IMPL
#undef GET_AspectTable_IMPL
} // namespace DeviceConfigFile
