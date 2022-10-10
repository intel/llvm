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
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"

using namespace llvm;

namespace {

constexpr StringRef SYCL_HOST_ACCESS_ATTR = "sycl-host-access";

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

Optional<StringRef> getGlobalVariableString(const Value *StringV) {
  if (const auto *StringGV = dyn_cast<GlobalVariable>(StringV))
    if (const auto *StringData =
            dyn_cast<ConstantDataSequential>(StringGV->getInitializer()))
      if (StringData->isCString())
        return StringData->getAsCString();
  return {};
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

  // Check pointer annotations.
  SmallVector<IntrinsicInst *, 4> RemovableAnnots;
  for (Function &F : M)
    for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I)
      if (auto *IntrInst = dyn_cast<IntrinsicInst>(&*I))
        if (IntrInst->getIntrinsicID() == Intrinsic::ptr_annotation &&
            transformSYCLPropertiesAnnotation(M, IntrInst, RemovableAnnots))
          CompileTimePropertiesMet = true;

  // Remove irrelevant "sycl-properties" annotations after the transformations.
  for (IntrinsicInst *IntrInst : RemovableAnnots) {
    assert(IntrInst->getNumUses() == 0);
    IntrInst->eraseFromParent();
  }

  // The pass just adds some metadata to the module, it should not ruin
  // any analysis, but we need return PreservedAnalyses::none() to inform
  // the caller that at least one compile-time property was met.
  return CompileTimePropertiesMet ? PreservedAnalyses::none()
                                  : PreservedAnalyses::all();
}

// Returns true if the transformation changed IntrInst.
bool CompileTimePropertiesPass::transformSYCLPropertiesAnnotation(
    Module &M, IntrinsicInst *IntrInst,
    SmallVectorImpl<IntrinsicInst *> &RemovableAnnotations) {
  assert(IntrInst->getIntrinsicID() == Intrinsic::ptr_annotation &&
         "Intrinsic is not a pointer annotation.");
  assert(IntrInst->arg_size() == 5 &&
         "Unexpected number of arguments in annotation intrinsic.");

  // Get the global variable with the annotation string.
  const GlobalVariable *AnnotStrArgGV = nullptr;
  const Value *IntrAnnotStringArg = IntrInst->getArgOperand(1);
  if (auto *GEP = dyn_cast<GEPOperator>(IntrAnnotStringArg))
    if (auto *C = dyn_cast<Constant>(GEP->getOperand(0)))
      AnnotStrArgGV = dyn_cast<GlobalVariable>(C);
  if (!AnnotStrArgGV)
    return false;

  // We only need to consider annotations with "sycl-properties" annotation
  // string.
  Optional<StringRef> AnnotStr = getGlobalVariableString(AnnotStrArgGV);
  if (!AnnotStr || AnnotStr->str() != "sycl-properties")
    return false;

  // Read the annotation values and create the new annotation string.
  std::string NewAnnotString = "";
  if (const auto *Cast =
          dyn_cast<BitCastOperator>(IntrInst->getArgOperand(4))) {
    if (const auto *AnnotValsGV =
            dyn_cast<GlobalVariable>(Cast->getOperand(0))) {
      if (const auto *AnnotValsAggr =
              dyn_cast<ConstantAggregate>(AnnotValsGV->getInitializer())) {
        assert(
            (AnnotValsAggr->getNumOperands() & 1) == 0 &&
            "sycl-properties annotation must have an even number of annotation "
            "values.");

        // Iterate over the pairs of property meta-names and meta-values.
        for (size_t I = 0; I < AnnotValsAggr->getNumOperands(); I += 2) {
          Optional<StringRef> PropMetaName =
              getGlobalVariableString(AnnotValsAggr->getOperand(I));
          Optional<StringRef> PropMetaValue =
              getGlobalVariableString(AnnotValsAggr->getOperand(I + 1));

          assert(PropMetaName &&
                 "Unexpected format for property name in annotation.");

          auto DecorIt = SpirvDecorMap.find(*PropMetaName);
          if (DecorIt == SpirvDecorMap.end())
            continue;
          uint32_t DecorCode = DecorIt->second.Code;

          // Expected format is '{X}' or '{X:Y}' where X is decoration ID and
          // Y is the value if present. It encloses Y in " to ensure that
          // string values are handled correctly. Note that " around values are
          // always valid, even if the decoration parameters are not strings.
          NewAnnotString += "{" + std::to_string(DecorCode);
          if (PropMetaValue)
            NewAnnotString += ":\"" + PropMetaValue->str() + "\"";
          NewAnnotString += "}";
        }
      }
    }
  }

  // If the new annotation string is empty there is no reason to keep it, so
  // replace it with the first operand and mark it for removal.
  if (NewAnnotString.empty()) {
    IntrInst->replaceAllUsesWith(IntrInst->getOperand(0));
    RemovableAnnotations.push_back(IntrInst);
    return true;
  }

  // Either reuse a previously generated one or create a new global variable
  // with the new annotation string.
  GlobalVariable *NewAnnotStringGV = nullptr;
  auto ExistingNewAnnotStringIt = ReusableAnnotStrings.find(NewAnnotString);
  if (ExistingNewAnnotStringIt != ReusableAnnotStrings.end()) {
    NewAnnotStringGV = ExistingNewAnnotStringIt->second;
  } else {
    Constant *NewAnnotStringData =
        ConstantDataArray::getString(M.getContext(), NewAnnotString);
    NewAnnotStringGV = new GlobalVariable(M, NewAnnotStringData->getType(),
                                          true, GlobalValue::PrivateLinkage,
                                          NewAnnotStringData, ".str");
    NewAnnotStringGV->setSection(AnnotStrArgGV->getSection());
    NewAnnotStringGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    ReusableAnnotStrings.insert({NewAnnotString, NewAnnotStringGV});
  }

  // Replace the annotation string with a bitcast of the new global variable.
  IntrInst->setArgOperand(
      1, ConstantExpr::getBitCast(NewAnnotStringGV,
                                  IntrAnnotStringArg->getType()));

  // The values are not in the annotation string, so we can remove the original
  // annotation value.
  unsigned DefaultAS = M.getDataLayout().getDefaultGlobalsAddressSpace();
  Type *Int8Ty = IntegerType::getInt8Ty(M.getContext());
  PointerType *Int8DefaultASPtrTy = Int8Ty->getPointerTo(DefaultAS);
  IntrInst->setArgOperand(4, ConstantPointerNull::get(Int8DefaultASPtrTy));
  return true;
}
