//==--- TargetInfo.h - Model SYCL queue-related target information ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "JITBinaryInfo.h"

namespace jit_compiler {

/// Unique ID for each supported architecture in the SYCL implementation.
///
/// Values of this type will only be used in the kernel fusion non-persistent
/// JIT. There is no guarantee for backwards compatibility, so this should not
/// be used in persistent caches.
using DeviceArchitecture = unsigned;

class TargetInfo {
public:
  static constexpr TargetInfo get(BinaryFormat Format,
                                  DeviceArchitecture Arch) {
    if (Format == BinaryFormat::SPIRV) {
      /// As an exception, SPIR-V targets have a single common ID (-1), as fused
      /// kernels will be reused across SPIR-V devices.
      return {Format, DeviceArchitecture(-1)};
    }
    return {Format, Arch};
  }

  TargetInfo() = default;

  constexpr BinaryFormat getFormat() const { return Format; }
  constexpr DeviceArchitecture getArch() const { return Arch; }

private:
  constexpr TargetInfo(BinaryFormat Format, DeviceArchitecture Arch)
      : Format(Format), Arch(Arch) {}

  BinaryFormat Format;
  DeviceArchitecture Arch;
};

} // namespace jit_compiler
