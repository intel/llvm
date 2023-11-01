//===---- CompileTimePropertiesPass.cpp - SYCL Compile Time Props Pass ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/CompileTimePropertiesPass.h"
#include "llvm/SYCLLowerIR/DeviceGlobals.h"
#include "llvm/SYCLLowerIR/ESIMD/ESIMDUtils.h"
#include "llvm/SYCLLowerIR/HostPipes.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

namespace {

constexpr StringRef SYCL_HOST_ACCESS_ATTR = "sycl-host-access";
constexpr StringRef SYCL_PIPELINED_ATTR = "sycl-pipelined";
constexpr StringRef SYCL_REGISTER_ALLOC_MODE_ATTR = "sycl-register-alloc-mode";
constexpr StringRef SYCL_GRF_SIZE_ATTR = "sycl-grf-size";

constexpr StringRef SPIRV_DECOR_MD_KIND = "spirv.Decorations";
constexpr StringRef SPIRV_PARAM_DECOR_MD_KIND = "spirv.ParameterDecorations";
// The corresponding SPIR-V OpCode for the host_access property is documented
// in the SPV_INTEL_global_variable_decorations design document:
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/DeviceGlobal/SPV_INTEL_global_variable_decorations.asciidoc#decoration
constexpr uint32_t SPIRV_HOST_ACCESS_DECOR = 6147;
constexpr uint32_t SPIRV_HOST_ACCESS_DEFAULT_VALUE = 2; // Read/Write

constexpr uint32_t SPIRV_INITIATION_INTERVAL_DECOR = 5917;
constexpr uint32_t SPIRV_PIPELINE_ENABLE_DECOR = 5919;

enum class DecorValueTy {
  uint32,
  boolean,
  string,
  none,
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

/// Builds a metadata node for a SPIR-V decoration (decoration code is
/// \c uint32_t integers) with no value.
///
/// @param Ctx    [in] the LLVM Context.
/// @param OpCode [in] the SPIR-V OpCode code.
///
/// @returns a pointer to the metadata node created for the required decoration
MDNode *buildSpirvDecorMetadata(LLVMContext &Ctx, uint32_t OpCode) {
  auto *Ty = Type::getInt32Ty(Ctx);
  SmallVector<Metadata *, 2> MD;
  MD.push_back(ConstantAsMetadata::get(
      Constant::getIntegerValue(Ty, APInt(32, OpCode))));
  return MDNode::get(Ctx, MD);
}

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

/// Builds a metadata node for a SPIR-V decoration (decoration code
/// is \c uint32_t integer and value is a string).
///
/// @param Ctx    [in] the LLVM Context.
/// @param OpCode [in] the SPIR-V OpCode code.
/// @param Value  [in] the SPIR-V decoration value.
///
/// @returns a pointer to the metadata node created for the required decoration
/// and its value.
MDNode *buildSpirvDecorMetadata(LLVMContext &Ctx, uint32_t OpCode,
                                StringRef Value) {
  auto *Ty = Type::getInt32Ty(Ctx);
  SmallVector<Metadata *, 2> MD;
  MD.push_back(ConstantAsMetadata::get(
      Constant::getIntegerValue(Ty, APInt(32, OpCode))));
  MD.push_back(MDString::get(Ctx, Value));
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

/// Gets the string value in a global variable. If the parameter is not a global
/// variable or it does not contain string data, then \c None is returned.
///
/// @param StringV  [in] the LLVM value of supposed \c GlobalVariable type with
///                 a string value.
///
/// @returns a \c StringRef with the string contained in \c StringV and \c None
///          if \c StringV is not a \c GlobalVariable or does not contain string
///          data.
std::optional<StringRef> getGlobalVariableString(const Value *StringV) {
  if (const auto *StringGV = dyn_cast<GlobalVariable>(StringV))
    if (const auto *StringData =
            dyn_cast<ConstantDataSequential>(StringGV->getInitializer()))
      if (StringData->isCString())
        return StringData->getAsCString();
  return {};
}

/// Tries to generate a SPIR-V decorate metadata node from an attribute. If
/// the attribute is unknown \c nullptr will be returned.
///
/// @param Ctx   [in] the LLVM context.
/// @param Attr  [in] the LLVM attribute to generate metadata for.
///
/// @returns a pointer to a new metadata node if \c Attr is an attribute with a
///          known corresponding SPIR-V decorate and the arguments are valid.
///          Otherwise \c nullptr is returned.
MDNode *attributeToDecorateMetadata(LLVMContext &Ctx, const Attribute &Attr) {
  // Currently, only string attributes are supported
  if (!Attr.isStringAttribute())
    return nullptr;
  auto DecorIt = SpirvDecorMap.find(Attr.getKindAsString());
  if (DecorIt == SpirvDecorMap.end())
    return nullptr;
  Decor DecorFound = DecorIt->second;
  uint32_t DecorCode = DecorFound.Code;
  switch (DecorFound.Type) {
  case DecorValueTy::uint32:
    return buildSpirvDecorMetadata(Ctx, DecorCode,
                                   getAttributeAsInteger<uint32_t>(Attr));
  case DecorValueTy::boolean:
    return buildSpirvDecorMetadata(Ctx, DecorCode, hasProperty(Attr));
  case DecorValueTy::string:
    return buildSpirvDecorMetadata(Ctx, DecorCode, Attr.getValueAsString());
  case DecorValueTy::none:
    return buildSpirvDecorMetadata(Ctx, DecorCode);
  default:
    llvm_unreachable("Unhandled decorator type.");
  }
}

/// Tries to generate a SPIR-V execution mode metadata node from an attribute.
/// If the attribute is unknown \c None will be returned.
///
/// @param Attr  [in] the LLVM attribute to generate metadata for.
/// @param F     [in] the LLVM function.
///
/// @returns a pair with the name of the resulting metadata and a pointer to
///          the metadata node with its values if the attribute has a
///          corresponding SPIR-V execution mode. Otherwise \c None is returned.
std::optional<std::pair<std::string, MDNode *>>
attributeToExecModeMetadata(const Attribute &Attr, Function &F) {
  Module &M = *F.getParent();
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DLayout = M.getDataLayout();

  // Currently, only string attributes are supported
  if (!Attr.isStringAttribute())
    return std::nullopt;
  StringRef AttrKindStr = Attr.getKindAsString();
  // Early exit if it is not a sycl-* attribute.
  if (!AttrKindStr.startswith("sycl-"))
    return std::nullopt;

  if (AttrKindStr == "sycl-work-group-size" ||
      AttrKindStr == "sycl-work-group-size-hint") {
    // Split values in the comma-separated list integers.
    SmallVector<StringRef, 3> ValStrs;
    Attr.getValueAsString().split(ValStrs, ',');

    assert(ValStrs.size() <= 3 &&
           "sycl-work-group-size and sycl-work-group-size-hint currently only "
           "support up to three values");

    // SYCL work-group sizes must be reversed for SPIR-V.
    std::reverse(ValStrs.begin(), ValStrs.end());

    // Use integer pointer size as closest analogue to size_t.
    IntegerType *IntPtrTy = DLayout.getIntPtrType(Ctx);
    IntegerType *SizeTTy = Type::getIntNTy(Ctx, IntPtrTy->getBitWidth());
    unsigned SizeTBitSize = SizeTTy->getBitWidth();

    // Get the integers from the strings.
    SmallVector<Metadata *, 3> MDVals;
    for (StringRef ValStr : ValStrs)
      MDVals.push_back(ConstantAsMetadata::get(
          Constant::getIntegerValue(SizeTTy, APInt(SizeTBitSize, ValStr, 10))));

    const char *MDName = (AttrKindStr == "sycl-work-group-size")
                             ? "reqd_work_group_size"
                             : "work_group_size_hint";
    return std::pair<std::string, MDNode *>(MDName, MDNode::get(Ctx, MDVals));
  }

  if (AttrKindStr == "sycl-sub-group-size") {
    uint32_t SubGroupSize = getAttributeAsInteger<uint32_t>(Attr);
    IntegerType *Ty = Type::getInt32Ty(Ctx);
    Metadata *MDVal = ConstantAsMetadata::get(
        Constant::getIntegerValue(Ty, APInt(32, SubGroupSize)));
    SmallVector<Metadata *, 1> MD{MDVal};
    return std::pair<std::string, MDNode *>("intel_reqd_sub_group_size",
                                            MDNode::get(Ctx, MD));
  }

  // The sycl-single-task attribute currently only has an effect when targeting
  // SPIR FPGAs, in which case it will generate a "max_global_work_dim" MD node
  // with a 0 value, similar to applying [[intel::max_global_work_dim(0)]] to
  // a SYCL single_target kernel.
  if (AttrKindStr == "sycl-single-task" &&
      Triple(M.getTargetTriple()).getSubArch() == Triple::SPIRSubArch_fpga) {
    IntegerType *Ty = Type::getInt32Ty(Ctx);
    Metadata *MDVal = ConstantAsMetadata::get(Constant::getNullValue(Ty));
    SmallVector<Metadata *, 1> MD{MDVal};
    return std::pair<std::string, MDNode *>("max_global_work_dim",
                                            MDNode::get(Ctx, MD));
  }

  if (AttrKindStr == "sycl-streaming-interface") {
    // generate either:
    //   !N = !{!"streaming"} or
    //   !N = !{!"streaming", !"stall_free_return"}
    SmallVector<Metadata *, 2> MD;
    MD.push_back(MDString::get(Ctx, "streaming"));
    if (getAttributeAsInteger<uint32_t>(Attr))
      MD.push_back(MDString::get(Ctx, "stall_free_return"));
    return std::pair<std::string, MDNode *>("ip_interface",
                                            MDNode::get(Ctx, MD));
  }

  if (AttrKindStr == "sycl-register-map-interface") {
    // generate either:
    //   !N = !{!"csr"} or
    //   !N = !{!"csr", !"wait_for_done_write"}
    SmallVector<Metadata *, 2> MD;
    MD.push_back(MDString::get(Ctx, "csr"));
    if (getAttributeAsInteger<uint32_t>(Attr))
      MD.push_back(MDString::get(Ctx, "wait_for_done_write"));
    return std::pair<std::string, MDNode *>("ip_interface",
                                            MDNode::get(Ctx, MD));
  }

  if ((AttrKindStr == SYCL_REGISTER_ALLOC_MODE_ATTR ||
       AttrKindStr == SYCL_GRF_SIZE_ATTR) &&
      !llvm::esimd::isESIMD(F)) {
    // TODO: Remove SYCL_REGISTER_ALLOC_MODE_ATTR support in next ABI break.
    uint32_t PropVal = getAttributeAsInteger<uint32_t>(Attr);
    if (AttrKindStr == SYCL_GRF_SIZE_ATTR) {
      assert((PropVal == 0 || PropVal == 128 || PropVal == 256) &&
             "Unsupported GRF Size");
      // Map sycl-grf-size values to RegisterAllocMode values used in SPIR-V.
      static constexpr int SMALL_GRF_REGALLOCMODE_VAL = 1;
      static constexpr int LARGE_GRF_REGALLOCMODE_VAL = 2;
      if (PropVal == 128)
        PropVal = SMALL_GRF_REGALLOCMODE_VAL;
      else if (PropVal == 256)
        PropVal = LARGE_GRF_REGALLOCMODE_VAL;
    }
    Metadata *AttrMDArgs[] = {ConstantAsMetadata::get(
        Constant::getIntegerValue(Type::getInt32Ty(Ctx), APInt(32, PropVal)))};
    return std::pair<std::string, MDNode *>("RegisterAllocMode",
                                            MDNode::get(Ctx, AttrMDArgs));
  }

  return std::nullopt;
}

SmallVector<std::pair<std::optional<StringRef>, std::optional<StringRef>>, 8>
parseSYCLPropertiesString(Module &M, IntrinsicInst *IntrInst) {
  SmallVector<std::pair<std::optional<StringRef>, std::optional<StringRef>>, 8>
      result;

  auto AnnotValsIntrOpd = IntrInst->getArgOperand(4);
  const GlobalVariable *AnnotValsGV = nullptr;
  if (AnnotValsIntrOpd->getType()->isPointerTy())
    AnnotValsGV = dyn_cast<GlobalVariable>(AnnotValsIntrOpd);
  else if (const auto *Cast = dyn_cast<BitCastOperator>(AnnotValsIntrOpd))
    AnnotValsGV = dyn_cast<GlobalVariable>(Cast->getOperand(0));
  if (AnnotValsGV) {
    if (const auto *AnnotValsAggr =
            dyn_cast<ConstantAggregate>(AnnotValsGV->getInitializer())) {
      assert(
          (AnnotValsAggr->getNumOperands() & 1) == 0 &&
          "sycl-properties annotation must have an even number of annotation "
          "values.");

      // Iterate over the pairs of property meta-names and meta-values.
      for (size_t I = 0; I < AnnotValsAggr->getNumOperands(); I += 2) {
        std::optional<StringRef> PropMetaName =
            getGlobalVariableString(AnnotValsAggr->getOperand(I));
        std::optional<StringRef> PropMetaValue =
            getGlobalVariableString(AnnotValsAggr->getOperand(I + 1));

        assert(PropMetaName &&
               "Unexpected format for property name in annotation.");

        result.push_back(std::make_pair(PropMetaName, PropMetaValue));
      }
    }
  }
  return result;
}

// Collect UserList if User isa<T>. Skip BitCast and AddrSpace
template <typename T>
void getUserListIgnoringCast(
    Value *V, SmallVector<std::pair<Instruction *, int>, 8> &List) {
  for (auto *User : V->users()) {
    if (auto *Inst = dyn_cast<T>(User)) {
      int Op_num = -1;
      for (unsigned i = 0; i < Inst->getNumOperands(); i++) {
        if (V == Inst->getOperand(i)) {
          Op_num = i;
          break;
        }
      }
      List.push_back(std::make_pair(Inst, Op_num));
    } else if (isa<BitCastInst>(User) || isa<AddrSpaceCastInst>(User))
      getUserListIgnoringCast<T>(User, List);
  }
}

} // anonymous namespace

PreservedAnalyses CompileTimePropertiesPass::run(Module &M,
                                                 ModuleAnalysisManager &MAM) {
  LLVMContext &Ctx = M.getContext();
  unsigned MDKindID = Ctx.getMDKindID(SPIRV_DECOR_MD_KIND);
  bool CompileTimePropertiesMet = false;
  unsigned MDParamKindID = Ctx.getMDKindID(SPIRV_PARAM_DECOR_MD_KIND);

  // Let's process all the globals
  for (auto &GV : M.globals()) {
    // we suppose the enumeration orders in every enumeration in the SYCL
    // headers are the same as in the descriptions of the corresponding
    // decorations in the SPV_INTEL_* extensions.
    SmallVector<Metadata *, 8> MDOps;
    for (auto &Attribute : GV.getAttributes()) {
      MDNode *SPIRVMetadata = attributeToDecorateMetadata(Ctx, Attribute);
      if (SPIRVMetadata)
        MDOps.push_back(SPIRVMetadata);
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

    if (isHostPipeVariable(GV)) {
      auto VarName = getGlobalVariableUniqueId(GV);
      MDOps.push_back(buildSpirvDecorMetadata(Ctx, SPIRV_HOST_ACCESS_DECOR,
                                              SPIRV_HOST_ACCESS_DEFAULT_VALUE,
                                              VarName));
    }

    // Add the generated metadata to the variable
    if (!MDOps.empty()) {
      GV.addMetadata(MDKindID, *MDNode::get(Ctx, MDOps));
      CompileTimePropertiesMet = true;
    }
  }

  // Process all properties on kernels.
  for (Function &F : M) {
    // Only consider kernels.
    if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;

    // Compile time properties on kernel arguments
    {
      SmallVector<Metadata *, 8> MDOps;
      MDOps.reserve(F.arg_size());
      bool FoundKernelProperties = false;
      for (unsigned I = 0; I < F.arg_size(); I++) {
        SmallVector<Metadata *, 8> MDArgOps;
        for (auto &Attribute : F.getAttributes().getParamAttrs(I)) {
          if (MDNode *SPIRVMetadata =
                  attributeToDecorateMetadata(Ctx, Attribute)) {
            if (Attribute.getKindAsString() == "sycl-alignment") {
              // apply alignment on kernel argument
              uint32_t AttrVal = getAttributeAsInteger<uint32_t>(Attribute);
              assert(llvm::isPowerOf2_64(AttrVal) &&
                     "sycl-alignment attribute is not a power of 2");
              // sycl-alignment is not collected to SPIRV.ParamDecoration
              // Convert sycl-alignment to general align
              auto Attr =
                  Attribute::getWithAlignment(Ctx, llvm::Align(AttrVal));
              F.addParamAttr(I, Attr);
              F.removeParamAttr(I, Attribute.getKindAsString());
              continue;
            }
            MDArgOps.push_back(SPIRVMetadata);
          }
        }
        if (!MDArgOps.empty())
          FoundKernelProperties = true;
        MDOps.push_back(MDNode::get(Ctx, MDArgOps));
      }
      // Add the generated metadata to the kernel function.
      if (FoundKernelProperties) {
        F.addMetadata(MDParamKindID, *MDNode::get(Ctx, MDOps));
        CompileTimePropertiesMet = true;
      }
    }

    SmallVector<Metadata *, 8> MDOps;
    SmallVector<std::pair<std::string, MDNode *>, 8> NamedMDOps;
    for (const Attribute &Attribute : F.getAttributes().getFnAttrs()) {
      // Handle pipelined attribute as a special case.
      if (Attribute.isStringAttribute() &&
          Attribute.getKindAsString() == SYCL_PIPELINED_ATTR) {
        auto PipelineOrInitiationInterval =
            getAttributeAsInteger<int32_t>(Attribute);
        MDNode *SPIRVMetadata;
        if (PipelineOrInitiationInterval < 0) {
          // Default pipelining desired
          SPIRVMetadata =
              buildSpirvDecorMetadata(Ctx, SPIRV_PIPELINE_ENABLE_DECOR, 1);
        } else if (PipelineOrInitiationInterval == 0) {
          // No pipelining desired
          SPIRVMetadata =
              buildSpirvDecorMetadata(Ctx, SPIRV_PIPELINE_ENABLE_DECOR, 0);
        } else {
          // Pipelining desired, with specified Initiation Interval
          SPIRVMetadata =
              buildSpirvDecorMetadata(Ctx, SPIRV_PIPELINE_ENABLE_DECOR, 1);
          MDOps.push_back(SPIRVMetadata);
          SPIRVMetadata =
              buildSpirvDecorMetadata(Ctx, SPIRV_INITIATION_INTERVAL_DECOR,
                                      PipelineOrInitiationInterval);
        }
        MDOps.push_back(SPIRVMetadata);
      } else if (MDNode *SPIRVMetadata =
                     attributeToDecorateMetadata(Ctx, Attribute))
        MDOps.push_back(SPIRVMetadata);
      else if (auto NamedMetadata = attributeToExecModeMetadata(Attribute, F))
        NamedMDOps.push_back(*NamedMetadata);
    }

    // Add the generated metadata to the kernel function.
    if (!MDOps.empty()) {
      F.addMetadata(MDKindID, *MDNode::get(Ctx, MDOps));
      CompileTimePropertiesMet = true;
    }

    // Add the new named metadata to the kernel function.
    for (std::pair<std::string, MDNode *> NamedMD : NamedMDOps) {
      // If multiple sources defined this metadata, prioritize the existing one.
      if (F.hasMetadata(NamedMD.first))
        continue;
      F.addMetadata(NamedMD.first, *NamedMD.second);
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

void CompileTimePropertiesPass::parseAlignmentAndApply(
    Module &M, IntrinsicInst *IntrInst) {
  // Get the global variable with the annotation string.
  const GlobalVariable *AnnotStrArgGV = nullptr;
  const Value *IntrAnnotStringArg = IntrInst->getArgOperand(1);
  if (IntrAnnotStringArg->getType()->isPointerTy())
    AnnotStrArgGV = dyn_cast<GlobalVariable>(IntrAnnotStringArg);
  else if (auto *GEP = dyn_cast<GEPOperator>(IntrAnnotStringArg))
    AnnotStrArgGV = dyn_cast<GlobalVariable>(GEP->getOperand(0));
  if (!AnnotStrArgGV)
    return;

  std::optional<StringRef> AnnotStr = getGlobalVariableString(AnnotStrArgGV);
  if (!AnnotStr)
    return;

  // parse properties string to decoration-value pairs
  auto Properties = parseSYCLPropertiesString(M, IntrInst);

  SmallVector<std::pair<Instruction *, int>, 8> TargetedInstList;
  // search ptr.annotation followed by Load/Store
  getUserListIgnoringCast<LoadInst>(IntrInst, TargetedInstList);
  getUserListIgnoringCast<StoreInst>(IntrInst, TargetedInstList);
  getUserListIgnoringCast<MemTransferInst>(IntrInst, TargetedInstList);

  for (auto &Property : Properties) {
    auto DecorStr = Property.first->str();
    auto DecorValue = Property.second;
    uint32_t AttrVal;

    if (DecorStr == "sycl-alignment") {
      assert(DecorValue && "sycl-alignment attribute is missing");

      bool DecorValueIntConvFailed = DecorValue->getAsInteger(0, AttrVal);

      std::ignore = DecorValueIntConvFailed;
      assert(!DecorValueIntConvFailed &&
             "sycl-alignment attribute is not an integer");
      assert(llvm::isPowerOf2_64(AttrVal) &&
             "sycl-alignment attribute is not a power of 2");

      auto Align_val = Align(AttrVal);
      // apply alignment attributes to load/store
      for (const auto &Pair : TargetedInstList) {
        auto *Inst = Pair.first;
        auto Op_num = Pair.second;
        if (auto *LInst = dyn_cast<LoadInst>(Inst)) {
          LInst->setAlignment(Align_val);
        } else if (auto *SInst = dyn_cast<StoreInst>(Inst)) {
          if (Op_num == 1)
            SInst->setAlignment(Align_val);
        } else if (auto *MI = dyn_cast<MemTransferInst>(Inst)) {
          if (Op_num == 0)
            MI->setDestAlignment(Align_val);
          else if (Op_num == 1)
            MI->setSourceAlignment(Align_val);
        }
      }
    }
  }
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
  if (IntrAnnotStringArg->getType()->isPointerTy())
    AnnotStrArgGV = dyn_cast<GlobalVariable>(IntrAnnotStringArg);
  else if (auto *GEP = dyn_cast<GEPOperator>(IntrAnnotStringArg))
    AnnotStrArgGV = dyn_cast<GlobalVariable>(GEP->getOperand(0));
  if (!AnnotStrArgGV)
    return false;

  // We only need to consider annotations with "sycl-properties" annotation
  // string.
  std::optional<StringRef> AnnotStr = getGlobalVariableString(AnnotStrArgGV);
  if (!AnnotStr || AnnotStr->str() != "sycl-properties")
    return false;

  // check alignment annotation and apply it to load/store
  parseAlignmentAndApply(M, IntrInst);

  // Read the annotation values and create the new annotation string.
  std::string NewAnnotString = "";
  auto Properties = parseSYCLPropertiesString(M, IntrInst);
  for (const auto &[PropName, PropVal] : Properties) {
    // sycl-alignment is converted to align on
    // previous parseAlignmentAndApply(), dropping here
    if (PropName == "sycl-alignment")
      continue;

    auto DecorIt = SpirvDecorMap.find(*PropName);
    if (DecorIt == SpirvDecorMap.end())
      continue;
    uint32_t DecorCode = DecorIt->second.Code;

    // Expected format is '{X}' or '{X:Y}' where X is decoration ID and
    // Y is the value if present. It encloses Y in " to ensure that
    // string values are handled correctly. Note that " around values are
    // always valid, even if the decoration parameters are not strings.
    NewAnnotString += "{" + std::to_string(DecorCode);
    if (PropVal)
      NewAnnotString += ":\"" + PropVal->str();

    if (PropName == "sycl-prefetch-hint")
      NewAnnotString += ",1"; // CachedINTEL
    if (PropName == "sycl-prefetch-hint-nt")
      NewAnnotString += ",3"; // InvalidateAfterReadINTEL

    if (PropVal)
      NewAnnotString += "\"";
    NewAnnotString += "}";
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
    NewAnnotStringGV = new GlobalVariable(
        M, NewAnnotStringData->getType(), true, GlobalValue::PrivateLinkage,
        NewAnnotStringData, ".str", nullptr, llvm::GlobalValue::NotThreadLocal,
        IntrAnnotStringArg->getType()->getPointerAddressSpace());
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
  PointerType *Arg4PtrTy =
      cast<PointerType>(IntrInst->getArgOperand(4)->getType());
  IntrInst->setArgOperand(4, ConstantPointerNull::get(Arg4PtrTy));
  return true;
}
