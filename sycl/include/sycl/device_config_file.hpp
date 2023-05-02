//===- device_config_file.h - Device Config File for SYCL  ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <map>

namespace DeviceConfigFile {

struct TargetInfo {
  int maySupportOtherAspects;
  std::vector<std::string> aspects;
  std::vector<unsigned> subGroupSizes;
  std::string aotToolchain;
  std::string aotToolchainOptions;
};

#define GET_TargetTable_IMPL
#include "sycl/device_config_file.inc"
}; // namespace DeviceConfigFile
