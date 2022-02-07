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
#include "CompileTimePropertiesPass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"

#include <cassert>

using namespace llvm;

namespace {

constexpr StringRef SYCL_DEVICE_GLOBAL_SIZE_ATTR = "sycl-device-global-size";
constexpr StringRef SYCL_UNIQUE_ID_ATTR = "sycl-unique-id";
constexpr StringRef SYCL_DEVICE_IMAGE_SCOPE_ATTR = "device_image_scope";

/// Returns the size (in bytes) of the underlying type \c T of the device
/// global variable.
///
/// The function gets this value from the LLVM IR attribute \c
/// sycl-device-global-size.
///
/// @param GV [in] Device Global variable.
///
/// @returns the size (int bytes) of the underlying type \c T of the
/// device global variable represented in the LLVM IR by  @GV.
uint32_t getUnderlyingTypeSize(const GlobalVariable &GV) {
  assert(GV.hasAttribute(SYCL_DEVICE_GLOBAL_SIZE_ATTR) &&
         "The device global variable must have the 'sycl-device-global-size' "
         "attribute that must contain a number representing the size of the "
         "underlying type T of the device global variable");
  return getAttributeAsInteger<uint32_t>(GV, SYCL_DEVICE_GLOBAL_SIZE_ATTR);
}

} // anonymous namespace

namespace llvm {

/// Return \c true if the variable @GV is a device global variable.
///
/// The function checks whether the variable has the LLVM IR attribute \c
/// sycl-device-global-size.
/// @param GV [in] A variable to test.
///
/// @return \c true if the variable is a device global variable, \c false
/// otherwise.
bool isDeviceGlobalVariable(const GlobalVariable &GV) {
  return GV.hasAttribute(SYCL_DEVICE_GLOBAL_SIZE_ATTR);
}

/// Returns the unique id for the device global variable.
///
/// The function gets this value from the LLVM IR attribute \c
/// sycl-unique-id.
///
/// @param GV [in] Device Global variable.
///
/// @returns the unique id of the device global variable represented
/// in the LLVM IR by \c GV.
StringRef getGlobalVariableUniqueId(const GlobalVariable &GV) {
  assert(GV.hasAttribute(SYCL_UNIQUE_ID_ATTR) &&
         "a 'sycl-unique-id' string must be associated with every device "
         "global variable");
  return GV.getAttribute(SYCL_UNIQUE_ID_ATTR).getValueAsString();
}

DeviceGlobalPropertyMapTy collectDeviceGlobalProperties(const Module &M) {
  DeviceGlobalPropertyMapTy DGM;
  auto DevGlobalNum = count_if(M.globals(), isDeviceGlobalVariable);
  if (DevGlobalNum == 0)
    return DGM;

  DGM.reserve(DevGlobalNum);

  for (auto &GV : M.globals()) {
    if (!isDeviceGlobalVariable(GV))
      continue;

    DGM[getGlobalVariableUniqueId(GV)] = {
        {{getUnderlyingTypeSize(GV),
          hasProperty(GV, SYCL_DEVICE_IMAGE_SCOPE_ATTR)}}};
  }

  return DGM;
}

} // namespace llvm
