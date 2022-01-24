//===----- DeviceGlobals.h - SYCL Device Globals Pass ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass which converts symbolic device globals attributes
// to integer id-based ones to later map to SPIRV device globals. The class
// allso contains a number of static methods to extract corresponding
// attributes of the device global variables and save them as a property set
// for the runtime.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/MapVector.h"

#include <cstdint>
#include <vector>

namespace llvm {

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

class DeviceGlobalsPass {
public:
  // Searches given module for occurrences of device global variable-specific
  // metadata and builds "device global variable name" ->
  // vector<"variable properties"> map.
  static DeviceGlobalPropertyMapTy
  collectDeviceGlobalProperties(const Module &M);
};

} // end namespace llvm
