//===------- HostPipes.h - get required info about FPGA Host Pipes --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The file contains a number of functions to extract corresponding attributes
// of the host pipe global variables and save them as a property set for the
// runtime.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/MapVector.h"

#include <cstdint>
#include <vector>

namespace llvm {

class GlobalVariable;
class Module;
class StringRef;

// Represents a host pipe variable - at SYCL RT level host pipe
// variables are being represented as a byte-array.
struct HostPipeProperty {
  HostPipeProperty(uint32_t Size) : Size(Size) {}

  // Encodes size of the underlying type T of the host pipe variable.
  uint32_t Size;
};

using HostPipePropertyMapTy =
    MapVector<StringRef, std::vector<HostPipeProperty>>;

/// Return \c true if the variable @GV is a host pipe variable.
///
/// The function checks whether the variable has the LLVM IR attribute \c
/// sycl-host-pipe
/// @param GV [in] A variable to test.
///
/// @return \c true if the variable is a host pipe variable, \c false
/// otherwise.
bool isHostPipeVariable(const GlobalVariable &GV);

/// Searches given module for occurrences of host pipe variable-specific
/// metadata and builds "host pipe variable name" ->
/// vector<"variable properties"> map.
///
/// @param M [in] LLVM Module.
///
/// @returns the "host pipe variable name" -> vector<"variable properties">
/// map.
HostPipePropertyMapTy collectHostPipeProperties(const Module &M);

/// Returns the unique id for the host pipe variable.
///
/// @param GV [in] Device Global variable.
///
/// @returns the unique id of the host pipe variable represented
/// in the LLVM IR by \c GV.
StringRef getHostPipeVariableUniqueId(const GlobalVariable &GV);

} // end namespace llvm
