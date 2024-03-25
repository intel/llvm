//===----- SYCLDeviceRequirements.h - collect data for used aspects ------=-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <vector>

namespace llvm {

class StringRef;

namespace module_split {
class ModuleDesc;
}
namespace util {
class PropertyValue;
}

struct SYCLDeviceRequirements {
  std::set<uint32_t> Aspects;
  std::set<uint32_t> FixedTarget;
  std::optional<llvm::SmallVector<uint64_t, 3>> ReqdWorkGroupSize;
  std::optional<llvm::SmallString<256>> JointMatrix;
  std::optional<llvm::SmallString<256>> JointMatrixMad;
  std::optional<uint32_t> SubGroupSize;

  std::map<StringRef, util::PropertyValue> asMap() const;
};

SYCLDeviceRequirements
computeDeviceRequirements(const module_split::ModuleDesc &M);

} // namespace llvm
