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

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"

#include <cassert>

using namespace llvm;

namespace {

constexpr StringRef SYCL_DEVICE_GLOBAL_SIZE_ATTR = "sycl-device-global-size";
constexpr StringRef SYCL_UNIQUE_ID_ATTR = "sycl-unique-id";
constexpr StringRef SYCL_DEVICE_IMAGE_SCOPE_ATTR = "device_image_scope";
constexpr StringRef SYCL_HOST_ACCESS_ATTR = "host_access";
constexpr StringRef SYCL_INIT_MODE_ATTR = "init_mode";
constexpr StringRef SYCL_IMPLEMENT_IN_CSR_ATTR = "implement_in_csr";

constexpr StringRef SPIRV_DECOR_MD_KIND = "spirv.Decorations";
constexpr uint32_t SPIRV_HOST_ACCESS_DECOR = 6147;
constexpr uint32_t SPIRV_INIT_MODE_DECOR = 6148;
constexpr uint32_t SPIRV_IMPLEMENT_IN_CSR_DECOR = 6149;

/// Converts the string into a boolean value. If the string is equal to "false"
/// we consider its value as /c false, /true otherwise.
///
/// @param Value [in] "boolean as string" value.
///
/// @returns \c false if the value of \c Value equals to "false", \c true
/// otherwise.
bool toBool(StringRef Value) { return !Value.equals("false"); }

/// Checks whether the device global variable has the \c AttributeName
/// property. The variable has the property if the \c AttributeName
/// attribute is defined for the variable and its value is not
/// represented as \c false.
///
/// @param GV            [in] Device Global variable.
/// @param AttributeName [in] Property name.
///
/// @returns \c true if variable \c GV has the \c AttributeName property,
/// \c false otherwise.
bool hasProperty(const GlobalVariable &GV, StringRef AttributeName) {
  return GV.hasAttribute(AttributeName) &&
         toBool(GV.getAttribute(AttributeName).getValueAsString());
}

/// Returns the value of the \c AttributeName attribute of the \c GV global
/// variable as an integer
///
/// @param GV            [in] Device Global variable.
/// @param AttributeName [in] Property name.
///
/// @returns \c the attribute's value as an integer.
uint32_t getAttributeAsInteger(const GlobalVariable &GV,
                               StringRef AttributeName) {
  assert(GV.hasAttribute(AttributeName) &&
         "The global variable GV must have the requested attribute");
  uint32_t value;
  bool error = GV.getAttribute(AttributeName)
                   .getValueAsString()
                   .getAsInteger(10, value);
  assert(!error && "The attribute's value is not a number");
  (void)error;
  return value;
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
         "attribute that must contain a number representing the size of the "
         "underlying type T of the device global variable");
  return getAttributeAsInteger(GV, SYCL_DEVICE_GLOBAL_SIZE_ATTR);
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

/// Builds a metadata node for a SPIR-V decoration.
///
/// @param Ctx   [in] the LLVM Context.
/// @param Ty    [in] the type of the decoration code and value.
/// @param Decor [in] the decoration code.
/// @param Value [in] the decoration value.
///
/// @returns a pointer to the metadata node created for the required decoration
/// and its value.
MDNode *buildSpirvDecorMetadata(LLVMContext &Ctx, Type *Ty, uint32_t Decor,
                                uint32_t Value) {
  SmallVector<Metadata *, 2> MD;
  MD.push_back(ConstantAsMetadata::get(
        Constant::getIntegerValue(Ty, APInt(32, Decor))));
  MD.push_back(ConstantAsMetadata::get(
        Constant::getIntegerValue(Ty, APInt(32, Value))));
  return MDNode::get(Ctx, MD);
}

} // namespace

PreservedAnalyses
DeviceGlobalsPass::run(Module &M, ModuleAnalysisManager &MAM) {
  LLVMContext &Ctx = M.getContext();
  unsigned MDKindID = Ctx.getMDKindID(SPIRV_DECOR_MD_KIND);
  auto *Int32Ty = Type::getInt32Ty(Ctx);

  for (auto &GV : M.globals()) {
    if (!isDeviceGlobalVariable(GV))
      continue;

    // we suppose the enumeration orders in the host_access and init_mode
    // enumerations in the SYCL headers are the same as in the descriptions of
    // the HostAccessINTEL and InitModeINTEL decorations in the
    // SPV_INTEL_global_variable_decorations extension.
    SmallVector<Metadata *, 4> MDOps;                                             
    if (GV.hasAttribute(SYCL_HOST_ACCESS_ATTR)) {
      uint32_t HostAccessValue =
          getAttributeAsInteger(GV, SYCL_HOST_ACCESS_ATTR);
      MDOps.push_back(buildSpirvDecorMetadata(Ctx, Int32Ty,
                                              SPIRV_HOST_ACCESS_DECOR,
                                              HostAccessValue));
    }

    if (GV.hasAttribute(SYCL_INIT_MODE_ATTR)) {
      uint32_t InitModeValue = getAttributeAsInteger(GV, SYCL_INIT_MODE_ATTR);
      MDOps.push_back(buildSpirvDecorMetadata(Ctx, Int32Ty,
                                              SPIRV_INIT_MODE_DECOR,
                                              InitModeValue));
    }

    if (GV.hasAttribute(SYCL_IMPLEMENT_IN_CSR_ATTR)) {
      uint32_t InCSRValue =
          hasProperty(GV, SYCL_IMPLEMENT_IN_CSR_ATTR) ? 1 : 0;
      MDOps.push_back(buildSpirvDecorMetadata(Ctx, Int32Ty,
                                              SPIRV_IMPLEMENT_IN_CSR_DECOR,
                                              InCSRValue));    
    }

    GV.addMetadata(MDKindID, *MDNode::get(Ctx, MDOps));
  }

  // The pass just adds some metadata to the module, it should not ruine
  // any analysis.
  return PreservedAnalyses::all();
}

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
        {{getUnderlyingTypeSize(GV),
          hasProperty(GV, SYCL_DEVICE_IMAGE_SCOPE_ATTR)}}};
  }

  return DGM;
}

ptrdiff_t
DeviceGlobalsPass::countDeviceGlobals(const Module &M) {
  return count_if(M.globals(), isDeviceGlobalVariable);
}
