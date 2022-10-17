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

class Module;
class StringRef;

std::map<StringRef, std::vector<uint32_t>>
getSYCLDeviceRequirements(const Module &M);

} // namespace llvm
