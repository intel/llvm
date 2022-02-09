//===---- CompileTimePropertiesPass.cpp - SYCL Compile Time Props Pass ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//

#include "CompileTimePropertiesPass.h"
#include "DeviceGlobals.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace {

constexpr StringRef SYCL_HOST_ACCESS_ATTR = "host_access";

constexpr StringRef SPIRV_DECOR_MD_KIND = "spirv.Decorations";
// The corresponding SPIR-V OpCode for the host_access property is documented
// in the SPV_INTEL_global_variable_decorations design document:
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/DeviceGlobal/SPV_INTEL_global_variable_decorations.asciidoc#decoration
constexpr uint32_t SPIRV_HOST_ACCESS_DECOR = 6147;
constexpr uint32_t SPIRV_HOST_ACCESS_DEFAULT_VALUE = 2; // Read/Write

enum class DecorValueTy {
  uint32,
  boolean,
};

struct Decor {
  uint32_t Code;
  DecorValueTy Type;
};

#define SYCL_COMPILE_TIME_PROPERTY(PropertyName, Decoration, ValueType)        \
  {PropertyName, {Decoration, ValueType}},

const StringMap<Decor> SpirvDecorMap = {
#include "CompileTimeProperties.def"
};
#undef SYCL_COMPILE_TIME_PROPERTY

/// Builds a metadata node for a SPIR-V decoration (both decoration code
/// and value are \c uint32_t integers).
///
/// @param Ctx    [in] the LLVM Context.
/// @param OpCode [in] the SPIR-V OpCode code.
/// @param Value  [in] the SPIR-V decoration value.
///
/// @returns a pointer to the metadata node created for the required decoration
/// and its value.
MDNode *buildSpirvDecorMetadata(LLVMContext &Ctx, uint32_t OpCode,
                                uint32_t Value) {
  auto *Ty = Type::getInt32Ty(Ctx);
  SmallVector<Metadata *, 2> MD;
  MD.push_back(ConstantAsMetadata::get(
      Constant::getIntegerValue(Ty, APInt(32, OpCode))));
  MD.push_back(
      ConstantAsMetadata::get(Constant::getIntegerValue(Ty, APInt(32, Value))));
  return MDNode::get(Ctx, MD);
}

/// Builds a metadata node for a SPIR-V decoration (both decoration code
/// and value are \c uint32_t integers, and the secondary extra operand is a
/// string).
///
/// @param Ctx        [in] the LLVM Context.
/// @param OpCode     [in] the SPIR-V OpCode code.
/// @param Value      [in] the SPIR-V decoration value.
/// @param Secondary  [in] the secondary "extra operands" (\c StringRef).
///
/// @returns a pointer to the metadata node created for the required decoration
/// and its value.
MDNode *buildSpirvDecorMetadata(LLVMContext &Ctx, uint32_t OpCode,
                                uint32_t Value, StringRef Secondary) {
  auto *Ty = Type::getInt32Ty(Ctx);
  SmallVector<Metadata *, 3> MD;
  MD.push_back(ConstantAsMetadata::get(
      Constant::getIntegerValue(Ty, APInt(32, OpCode))));
  MD.push_back(
      ConstantAsMetadata::get(Constant::getIntegerValue(Ty, APInt(32, Value))));
  MD.push_back(MDString::get(Ctx, Secondary));
  return MDNode::get(Ctx, MD);
}

} // anonymous namespace

PreservedAnalyses CompileTimePropertiesPass::run(Module &M,
                                                 ModuleAnalysisManager &MAM) {
  LLVMContext &Ctx = M.getContext();
  unsigned MDKindID = Ctx.getMDKindID(SPIRV_DECOR_MD_KIND);
  bool CompileTimePropertiesMet = false;

  // Let's process all the globals
  for (auto &GV : M.globals()) {
    // we suppose the enumeration orders in every enumeration in the SYCL
    // headers are the same as in the descriptions of the corresponding
    // decorations in the SPV_INTEL_* extensions.
    SmallVector<Metadata *, 8> MDOps;
    for (auto &Attribute : GV.getAttributes()) {
      // Currently, only string attributes are supported
      if (!Attribute.isStringAttribute())
        continue;
      auto DecorIt = SpirvDecorMap.find(Attribute.getKindAsString());
      if (DecorIt == SpirvDecorMap.end())
        continue;
      auto Decor = DecorIt->second;
      auto DecorCode = Decor.Code;
      auto DecorValue = Decor.Type == DecorValueTy::uint32
                            ? getAttributeAsInteger<uint32_t>(Attribute)
                            : hasProperty(Attribute);
      MDOps.push_back(buildSpirvDecorMetadata(Ctx, DecorCode, DecorValue));
    }

    // Some properties should be handled specially.

    // The host_access property is handled specially for device global variables
    // because the SPIR-V decoration requires two "extra operands". The second
    // SPIR-V operand is the "name" (the value of the "sycl-unique-id" property)
    // of the variable.
    if (isDeviceGlobalVariable(GV)) {
      auto HostAccessDecorValue =
          GV.hasAttribute(SYCL_HOST_ACCESS_ATTR)
              ? getAttributeAsInteger<uint32_t>(GV, SYCL_HOST_ACCESS_ATTR)
              : SPIRV_HOST_ACCESS_DEFAULT_VALUE;
      auto VarName = getGlobalVariableUniqueId(GV);
      MDOps.push_back(buildSpirvDecorMetadata(Ctx, SPIRV_HOST_ACCESS_DECOR,
                                              HostAccessDecorValue, VarName));
    }

    // Add the generated metadata to the variable
    if (!MDOps.empty()) {
      GV.addMetadata(MDKindID, *MDNode::get(Ctx, MDOps));
      CompileTimePropertiesMet = true;
    }
  }

  // The pass just adds some metadata to the module, it should not ruin
  // any analysis, but we need return PreservedAnalyses::none() to inform
  // the caller that at least one compile-time property was met.
  return CompileTimePropertiesMet ? PreservedAnalyses::none()
                                  : PreservedAnalyses::all();
}
