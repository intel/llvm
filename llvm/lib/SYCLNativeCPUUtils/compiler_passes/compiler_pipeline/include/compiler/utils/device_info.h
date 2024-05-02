// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// @file
///
/// @brief Information about compiler device information.

#ifndef COMPILER_UTILS_DEVICE_INFO_H_INCLUDED
#define COMPILER_UTILS_DEVICE_INFO_H_INCLUDED

#include <llvm/IR/PassManager.h>

#include <optional>

namespace compiler {
namespace utils {

/// @brief Bitfield of all possible floating point capabilities.
///
/// Each Mux device struct has a member which denotes the floating point
/// capabilities of that device, as a bitfield of the following enum.
///
/// NOTE: Must be kept in sync with mux_floating_point_capabilities_e in
/// mux/include/mux/mux.h! This should probably be placed in an intermediary
/// mux/compiler library and shared as part of CA-4236.
enum device_floating_point_capabilities_e {
  /// @brief Denormals supported.
  device_floating_point_capabilities_denorm = 0x1,
  /// @brief INF and NaN are supported.
  device_floating_point_capabilities_inf_nan = 0x2,
  /// @brief Round to nearest even supported.
  device_floating_point_capabilities_rte = 0x4,
  /// @brief Round to zero supported.
  device_floating_point_capabilities_rtz = 0x8,
  /// @brief Round to positive infinity supported.
  device_floating_point_capabilities_rtp = 0x10,
  /// @brief Round to negative infinity supported.
  device_floating_point_capabilities_rtn = 0x20,
  /// @brief Fused multiply add supported.
  device_floating_point_capabilities_fma = 0x40,
  /// @brief Floating point operations are written in software.
  device_floating_point_capabilities_soft = 0x80,
  /// @brief Binary format conforms to the IEEE-754 specification.
  device_floating_point_capabilities_full = 0x100
};

struct DeviceInfo {
  DeviceInfo() = default;

  /// @brief Construct a DeviceInfo from individual properties
  ///
  /// @param h Enumeration of half-precision floating-point capabilities
  /// @param f Enumeration of single-precision floating-point capabilities
  /// @param d Enumeration of double-precision floating-point capabilities
  /// @param max_work_width  The maximum number of work-items of a work-group
  /// allowed to execute in one invocation of a kernel.
  DeviceInfo(uint32_t h, uint32_t f, uint32_t d, uint32_t max_work_width)
      : half_capabilities(h),
        float_capabilities(f),
        double_capabilities(d),
        max_work_width(max_work_width) {}

  uint32_t half_capabilities = 0;
  uint32_t float_capabilities = 0;
  uint32_t double_capabilities = 0;
  uint32_t max_work_width = 0;

  /// @brief List of supported 'required' sub-group sizes reported by this
  /// device.
  ///
  /// These are only the sub-group sizes that can be requested as 'required' for
  /// a kernel; the compiler may produce a wide range of other sub-group sizes
  /// on undecorated kernels, assuming sub-groups are supported by the device.
  std::vector<uint32_t> reqd_sub_group_sizes;

  /// @brief Handle invalidation events from the new pass manager.
  ///
  /// @return false, as this analysis can never be invalidated.
  bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                  llvm::ModuleAnalysisManager::Invalidator &) {
    return false;
  }
};

/// @brief Caches and returns the device information for a Module.
class DeviceInfoAnalysis : public llvm::AnalysisInfoMixin<DeviceInfoAnalysis> {
  friend AnalysisInfoMixin<DeviceInfoAnalysis>;

 public:
  using Result = DeviceInfo;

  DeviceInfoAnalysis() = default;
  DeviceInfoAnalysis(Result res) : Info(res) {}

  /// @brief Retrieve the DeviceInfo for the requested module.
  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) {
    return Info ? *Info : Result();
  }

  /// @brief Return the name of the pass.
  static llvm::StringRef name() { return "Device info analysis"; }

 private:
  /// @brief Optional device information
  std::optional<Result> Info;

  /// @brief Unique pass identifier.
  static llvm::AnalysisKey Key;
};

}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_DEVICE_INFO_H_INCLUDED
