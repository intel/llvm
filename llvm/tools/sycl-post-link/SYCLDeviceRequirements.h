//===----- SYCLDeviceRequirements.h - collect data for used aspects ------=-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <map>
#include <vector>

namespace llvm {

class StringRef;

namespace module_split {
class ModuleDesc;
}
namespace util {
class PropertyValue;
}

void getSYCLDeviceRequirements(
    const module_split::ModuleDesc &M,
    std::map<StringRef, util::PropertyValue> &Requirements);

} // namespace llvm
