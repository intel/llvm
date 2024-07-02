//===----- SYCLDeviceRequirements.h - collect data for used aspects ------=-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <vector>

namespace llvm {
class Function;
class Module;
class StringRef;

namespace util {
class PropertyValue;
}

struct SYCLDeviceRequirements {
  struct AspectNameValuePair {
    llvm::SmallString<64> Name;
    uint32_t Value;
    AspectNameValuePair(StringRef Name, uint32_t Value)
        : Name(Name), Value(Value) {}
    bool operator<(const AspectNameValuePair &rhs) const {
      return Value < rhs.Value;
    }
    bool operator==(const AspectNameValuePair &rhs) const {
      return Value == rhs.Value;
    }
  };
  std::set<AspectNameValuePair> Aspects;
  std::set<uint32_t> FixedTarget;
  std::optional<llvm::SmallVector<uint64_t, 3>> ReqdWorkGroupSize;
  std::optional<uint32_t> WorkGroupNumDim;
  std::optional<llvm::SmallString<256>> JointMatrix;
  std::optional<llvm::SmallString<256>> JointMatrixMad;
  std::optional<uint32_t> SubGroupSize;

  std::map<StringRef, util::PropertyValue> asMap() const;
};

SYCLDeviceRequirements
computeDeviceRequirements(const Module &M,
                          const SetVector<Function *> &EntryPoints);

} // namespace llvm
