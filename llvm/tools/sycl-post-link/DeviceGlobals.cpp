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

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>

using namespace llvm;

namespace {

constexpr StringRef SYCL_DEVICE_GLOBAL_SIZE_ATTR = "sycl-device-global-size";
constexpr StringRef SYCL_UNIQUE_ID_ATTR = "sycl-unique-id";
constexpr StringRef SYCL_DEVICE_IMAGE_SCOPE_ATTR = "device_image_scope";

void AssertRelease(bool Cond, const char *Msg) {
  if (!Cond)
    report_fatal_error((Twine("DeviceGlobals.cpp: ") + Msg).str().c_str());
}

/// Checks whether the string that represents a boolean value ("true"/"false"
/// or "1"/"0" actually represents \c false.
///
/// @param Value [in] "boolean as string" value.
///
/// @returns \c true if the value of \c Value represents \c false, \c false
/// otherwise.
bool isFalsey(StringRef Value) {
  return Value.equals("false") || !Value.compare_numeric("0");
}

/// Checks whether the device global variable has the \c device_image_scope
/// property. The variable has the property if the \c sycl-device-image-scope
/// attribute is defined for the variable and the attribute value cannot be
/// represented as \c false.
///
/// @param GV [in] Device Global variable.
///
/// @returns \c true if variable \c GV has the \c device_image_scope property,
/// \c false otherwise.
bool hasDeviceImageScope(const GlobalVariable &GV) {
  return GV.hasAttribute(SYCL_DEVICE_IMAGE_SCOPE_ATTR) &&
      !isFalsey(GV.getAttribute(SYCL_DEVICE_IMAGE_SCOPE_ATTR)
                  .getValueAsString());
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
  return static_cast<uint32_t>(std::stoul(GV.getAttribute(
      SYCL_DEVICE_GLOBAL_SIZE_ATTR).getValueAsString().str()));
}

/// Returns the unique id for the device global variable.
///
/// The function gets this value from the LLVM IR attribute \c
/// sycl-unique-id. If the attribute is not found for the variable
/// an error should occur even in the release built.
///
/// @param GV [in] Device Global variable.
///
/// @returns the unique id of the device global variable represented
/// in the LLVM IR by \c GV.
StringRef getUniqueId(const GlobalVariable &GV) {
  AssertRelease(GV.hasAttribute(SYCL_UNIQUE_ID_ATTR),
      "a 'sycl-unique-id' string must be associated with every device "
      "global variable");
  return GV.getAttribute(SYCL_UNIQUE_ID_ATTR).getValueAsString();
}

} // namespace

DeviceGlobalPropertyMapTy
DeviceGlobalsPass::collectDeviceGlobalProperties(const Module &M) {
  auto DevGlobalFilter = [](auto &GV) {
    return GV.hasAttribute(SYCL_DEVICE_GLOBAL_SIZE_ATTR);
  };
  auto DevGlobalNum = std::count_if(M.globals().begin(), M.globals().end(),
      DevGlobalFilter);

  DeviceGlobalPropertyMapTy DGM;
  DGM.reserve(DevGlobalNum);

  for (auto &GV : M.globals()) {
    if (!DevGlobalFilter(GV))
      continue;

    DGM[getUniqueId(GV)] = {{{getUnderlyingTypeSize(GV),
        hasDeviceImageScope(GV)}}};
  }

  return DGM;
}
