//===--- DeviceGlobals.h - get required into about SYCL Device Globals ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The file contains a number of functions to extract corresponding attributes
// of the device global variables and save them as a property set for the
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

// Represents a device global variable - at SYCL RT level device global
// variables are being represented as a byte-array.
struct DeviceGlobalProperty {
  DeviceGlobalProperty(uint32_t Size, uint8_t DeviceImageScope)
      : Size(Size), DeviceImageScope(DeviceImageScope) {}

  // Encodes size of the underlying type T of the device global variable.
  uint32_t Size;

  // Either 1 (true) or 0 (false), telling whether the device global variable
  // was declared with the device_image_scope property.
  uint8_t DeviceImageScope;
};

using DeviceGlobalPropertyMapTy =
    MapVector<StringRef, std::vector<DeviceGlobalProperty>>;

/// Searches given module for occurrences of device global variable-specific
/// metadata and builds "device global variable name" ->
/// vector<"variable properties"> map.
///
/// @param M [in] LLVM Module.
///
/// @returns the "device global variable name" -> vector<"variable properties">
/// map.
DeviceGlobalPropertyMapTy collectDeviceGlobalProperties(const Module &M);

/// Return \c true if the variable @GV is a device global variable.
///
/// @param GV [in] A variable to test.
///
/// @return \c true if the variable is a device global variable, \c false
/// otherwise.
bool isDeviceGlobalVariable(const GlobalVariable &GV);

/// Returns the unique id for the device global variable.
///
/// @param GV [in] Device Global variable.
///
/// @returns the unique id of the device global variable represented
/// in the LLVM IR by \c GV.
StringRef getGlobalVariableUniqueId(const GlobalVariable &GV);

} // end namespace llvm
