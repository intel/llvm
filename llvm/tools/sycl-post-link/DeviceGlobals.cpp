//===----- DeviceGlobals.cpp - SYCL Device Globals Pass -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//

#include "DeviceGlobals.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"

#include <cassert>

using namespace llvm;

namespace {

constexpr StringRef SYCL_DEVICE_GLOBAL_SIZE_ATTR = "sycl-device-global-size";
constexpr StringRef SYCL_UNIQUE_ID_ATTR = "sycl-unique-id";
constexpr StringRef SYCL_DEVICE_IMAGE_SCOPE_ATTR = "device_image_scope";

/// Converts the string into a boolean value. If the string is equal to "false"
/// we consider its value as /c false, /true otherwise.
///
/// @param Value [in] "boolean as string" value.
///
/// @returns \c false if the value of \c Value equals to "false", \c true
/// otherwise.
bool toBool(StringRef Value) { return !Value.equals("false"); }

/// Checks whether the device global variable has the \c device_image_scope
/// property. The variable has the property if the \c sycl-device-image-scope
/// attribute is defined for the variable and the attribute value is not
/// represented as \c false.
///
/// @param GV [in] Device Global variable.
///
/// @returns \c true if variable \c GV has the \c device_image_scope property,
/// \c false otherwise.
bool hasDeviceImageScope(const GlobalVariable &GV) {
  return GV.hasAttribute(SYCL_DEVICE_IMAGE_SCOPE_ATTR) &&
         toBool(
             GV.getAttribute(SYCL_DEVICE_IMAGE_SCOPE_ATTR).getValueAsString());
}

/// Returns the size (in bytes) of the underlying type \c T of the device
/// global variable.
///
/// The function gets this value from the LLVM IR attribute \c
/// sycl-device-global-size.
///
/// @param GV [in] Device Global variable.
///
/// @returns the size (int bytes) of the underlying type \c T of the
/// device global variable represented in the LLVM IR by \c GV.
uint32_t getUnderlyingTypeSize(const GlobalVariable &GV) {
  assert(GV.hasAttribute(SYCL_DEVICE_GLOBAL_SIZE_ATTR) &&
         "The device global variable must have the 'sycl-device-global-size' "
         "attribute");
  uint32_t value;
  bool error = GV.getAttribute(SYCL_DEVICE_GLOBAL_SIZE_ATTR)
                   .getValueAsString()
                   .getAsInteger(10, value);
  assert(!error &&
         "The 'sycl-device-global-size' attribute must contain a number"
         " representing the size of the underlying type T of the device"
         " global variable");
  (void)error;
  return value;
}

/// Returns the unique id for the device global variable.
///
/// The function gets this value from the LLVM IR attribute \c
/// sycl-unique-id. If the attribute is not found for the variable
/// an error should occur even in the release build.
///
/// @param GV [in] Device Global variable.
///
/// @returns the unique id of the device global variable represented
/// in the LLVM IR by \c GV.
StringRef getUniqueId(const GlobalVariable &GV) {
  assert(GV.hasAttribute(SYCL_UNIQUE_ID_ATTR) &&
         "a 'sycl-unique-id' string must be associated with every device "
         "global variable");
  return GV.getAttribute(SYCL_UNIQUE_ID_ATTR).getValueAsString();
}

/// Checks whether the variable is a device global one.
///
/// A variable is device global if and only if it contains the LLVM IR
/// attribute \c sycl-device-global-size.
///
/// @param GV [in] a global variable.
///
/// @returns \c true if variable \c GV is device global, \c false otherwise.
bool isDeviceGlobalVariable(const GlobalVariable &GV) {
  return GV.hasAttribute(SYCL_DEVICE_GLOBAL_SIZE_ATTR);
}

} // namespace

DeviceGlobalPropertyMapTy
DeviceGlobalsPass::collectDeviceGlobalProperties(const Module &M) {
  DeviceGlobalPropertyMapTy DGM;
  auto DevGlobalNum = countDeviceGlobals(M);
  if (DevGlobalNum == 0)
    return DGM;

  DGM.reserve(DevGlobalNum);

  for (auto &GV : M.globals()) {
    if (!isDeviceGlobalVariable(GV))
      continue;

    DGM[getUniqueId(GV)] = {
        {{getUnderlyingTypeSize(GV), hasDeviceImageScope(GV)}}};
  }

  return DGM;
}

ptrdiff_t
DeviceGlobalsPass::countDeviceGlobals(const Module &M) {
  return count_if(M.globals(), isDeviceGlobalVariable);
}
