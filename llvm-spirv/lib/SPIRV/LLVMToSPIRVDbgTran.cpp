//===- LLVMToSPIRVDbgTran.cpp - Converts debug info to SPIR-V ---*- C++ -*-===//
//
//                     The LLVM/SPIR-V Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2018 Intel Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Intel Corporation, nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file implements translation of debug info from LLVM metadata to SPIR-V
//
//===----------------------------------------------------------------------===//
#include "LLVMToSPIRVDbgTran.h"
#include "SPIRVWriter.h"

#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"

using namespace SPIRV;

// Public interface

/// This function is looking for debug information in the LLVM module
/// and translates it to SPIRV
void LLVMToSPIRVDbgTran::transDebugMetadata() {
  DIF.processModule(*M);
  if (DIF.compile_unit_count() == 0)
    return;

  for (DICompileUnit *CU : DIF.compile_units()) {
    transDbgEntry(CU);
    for (DIImportedEntity *IE : CU->getImportedEntities())
      transDbgEntry(IE);
  }

  for (const DIType *T : DIF.types())
    transDbgEntry(T);

  // When translating a debug lexical block, we expect the translation of its
  // parent scope (say it's a subprogram) already been created in MDMap.
  // Otherwise, we have to dive into the details of subprogram translation
  // first. During this process, we will try to resolve all retainedNodes
  // (aka, variables) owned by this subprogram.
  // And local variable's scope could be the original lexical block that we
  // haven't finish translating yet. In other words, the block hasn't been
  // inserted into MDMap cache yet.
  // So we try to invoke transDbgEntryImpl on the same lexical block again,
  // then we get a duplicated lexical block messing up the debug info.
  //
  // Scheduling the translation of subprograms ahead of scopes (lexical blocks)
  // solves this dependency cycle issue.
  for (const DISubprogram *F : DIF.subprograms())
    transDbgEntry(F);

  for (const DIScope *S : DIF.scopes())
    transDbgEntry(S);

  for (const DIGlobalVariableExpression *G : DIF.global_variables()) {
    transDbgEntry(G->getVariable());
  }

  for (const DbgVariableIntrinsic *DDI : DbgDeclareIntrinsics)
    finalizeDebugDeclare(DDI);

  for (const DbgVariableIntrinsic *DVI : DbgValueIntrinsics)
    finalizeDebugValue(DVI);

  transLocationInfo();
}

// llvm.dbg.declare intrinsic.

SPIRVValue *LLVMToSPIRVDbgTran::createDebugDeclarePlaceholder(
    const DbgVariableIntrinsic *DbgDecl, SPIRVBasicBlock *BB) {
  DbgDeclareIntrinsics.push_back(DbgDecl);
  using namespace SPIRVDebug::Operand::DebugDeclare;
  SPIRVWordVec Ops(OperandCount, getDebugInfoNoneId());
  SPIRVId ExtSetId = BM->getExtInstSetId(BM->getDebugInfoEIS());
  return BM->addExtInst(getVoidTy(), ExtSetId, SPIRVDebug::Declare, Ops, BB);
}

void LLVMToSPIRVDbgTran::finalizeDebugDeclare(
    const DbgVariableIntrinsic *DbgDecl) {
  SPIRVValue *V = SPIRVWriter->getTranslatedValue(DbgDecl);
  assert(V && "llvm.dbg.declare intrinsic isn't mapped to a SPIRV instruction");
  assert(V->isExtInst(BM->getDebugInfoEIS(), SPIRVDebug::Declare) &&
         "llvm.dbg.declare intrinsic has been translated wrong!");
  if (!V || !V->isExtInst(BM->getDebugInfoEIS(), SPIRVDebug::Declare))
    return;
  SPIRVExtInst *DD = static_cast<SPIRVExtInst *>(V);
  SPIRVBasicBlock *BB = DD->getBasicBlock();
  llvm::Value *Alloca = DbgDecl->getVariableLocationOp(0);

  using namespace SPIRVDebug::Operand::DebugDeclare;
  SPIRVWordVec Ops(OperandCount);
  Ops[DebugLocalVarIdx] = transDbgEntry(DbgDecl->getVariable())->getId();
  Ops[VariableIdx] = Alloca ? SPIRVWriter->transValue(Alloca, BB)->getId()
                            : getDebugInfoNoneId();
  Ops[ExpressionIdx] = transDbgEntry(DbgDecl->getExpression())->getId();
  DD->setArguments(Ops);
}

// llvm.dbg.value intrinsic.

SPIRVValue *LLVMToSPIRVDbgTran::createDebugValuePlaceholder(
    const DbgVariableIntrinsic *DbgValue, SPIRVBasicBlock *BB) {
  if (!DbgValue->getVariableLocationOp(0))
    return nullptr; // It is pointless without new value

  DbgValueIntrinsics.push_back(DbgValue);
  using namespace SPIRVDebug::Operand::DebugValue;
  SPIRVWordVec Ops(MinOperandCount, getDebugInfoNone()->getId());
  SPIRVId ExtSetId = BM->getExtInstSetId(BM->getDebugInfoEIS());
  return BM->addExtInst(getVoidTy(), ExtSetId, SPIRVDebug::Value, Ops, BB);
}

void LLVMToSPIRVDbgTran::finalizeDebugValue(
    const DbgVariableIntrinsic *DbgValue) {
  SPIRVValue *V = SPIRVWriter->getTranslatedValue(DbgValue);
  assert(V && "llvm.dbg.value intrinsic isn't mapped to a SPIRV instruction");
  assert(V->isExtInst(BM->getDebugInfoEIS(), SPIRVDebug::Value) &&
         "llvm.dbg.value intrinsic has been translated wrong!");
  if (!V || !V->isExtInst(BM->getDebugInfoEIS(), SPIRVDebug::Value))
    return;
  SPIRVExtInst *DV = static_cast<SPIRVExtInst *>(V);
  SPIRVBasicBlock *BB = DV->getBasicBlock();
  Value *Val = DbgValue->getVariableLocationOp(0);
  DIExpression *Expr = DbgValue->getExpression();
  if (!isNonSemanticDebugInfo()) {
    if (DbgValue->getNumVariableLocationOps() > 1) {
      Val = UndefValue::get(Val->getType());
      Expr = DIExpression::get(M->getContext(), {});
    }
  }
  using namespace SPIRVDebug::Operand::DebugValue;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[DebugLocalVarIdx] = transDbgEntry(DbgValue->getVariable())->getId();
  Ops[ValueIdx] = SPIRVWriter->transValue(Val, BB)->getId();
  Ops[ExpressionIdx] = transDbgEntry(Expr)->getId();
  DV->setArguments(Ops);
}

// Emitting DebugScope and OpLine instructions

void LLVMToSPIRVDbgTran::transLocationInfo() {
  for (const Function &F : *M) {
    for (const BasicBlock &BB : F) {
      SPIRVValue *V = SPIRVWriter->getTranslatedValue(&BB);
      assert(V && V->isBasicBlock() &&
             "Basic block is expected to be translated");
      SPIRVBasicBlock *SBB = static_cast<SPIRVBasicBlock *>(V);
      MDNode *DbgScope = nullptr;
      MDNode *InlinedAt = nullptr;
      SPIRVString *File = nullptr;
      unsigned LineNo = 0;
      unsigned Col = 0;
      for (const Instruction &I : BB) {
        if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
          if (II->getIntrinsicID() == Intrinsic::dbg_label) {
            // SPIR-V doesn't support llvm.dbg.label intrinsic translation
            continue;
          }
          if (II->getIntrinsicID() == Intrinsic::annotation ||
              II->getIntrinsicID() == Intrinsic::var_annotation ||
              II->getIntrinsicID() == Intrinsic::ptr_annotation) {
            // llvm call instruction for llvm .*annotation intrinsics
            // is translated into SPIR-V instruction only if it represents
            // call of __builtin_intel_fpga_reg() builtin. In other cases this
            // instruction is dropped. In these cases debug info for this call
            // should be skipped too.
            // TODO: Remove skipping of debug info when *.annotation call will
            //       be handled in a better way during SPIR-V translation.
            V = SPIRVWriter->getTranslatedValue(&I);
            if (!V || V->getOpCode() != OpFPGARegINTEL)
              continue;
          }
        }
        V = SPIRVWriter->getTranslatedValue(&I);
        if (!V || isConstantOpCode(V->getOpCode()))
          continue;
        const DebugLoc &DL = I.getDebugLoc();
        if (!DL.get()) {
          if (DbgScope || InlinedAt) { // Emit DebugNoScope
            DbgScope = nullptr;
            InlinedAt = nullptr;
            transDebugLoc(DL, SBB, static_cast<SPIRVInstruction *>(V));
          }
          continue;
        }
        // Once scope or inlining has changed emit another DebugScope
        if (DL.getScope() != DbgScope || DL.getInlinedAt() != InlinedAt) {
          DbgScope = DL.getScope();
          InlinedAt = DL.getInlinedAt();
          transDebugLoc(DL, SBB, static_cast<SPIRVInstruction *>(V));
        }
        // If any component of OpLine has changed emit another OpLine
        SPIRVString *DirAndFile = BM->getString(getFullPath(DL.get()));
        if (File != DirAndFile || LineNo != DL.getLine() ||
            Col != DL.getCol()) {
          File = DirAndFile;
          LineNo = DL.getLine();
          Col = DL.getCol();
          // According to the spec, OpLine for an OpBranch/OpBranchConditional
          // must precede the merge instruction and not the branch instruction
          if (V->getOpCode() == OpBranch ||
              V->getOpCode() == OpBranchConditional) {
            auto *VPrev = static_cast<SPIRVInstruction *>(V)->getPrevious();
            if (VPrev && (VPrev->getOpCode() == OpLoopMerge ||
                          VPrev->getOpCode() == OpLoopControlINTEL)) {
              V = VPrev;
            }
          }
          if (BM->getDebugInfoEIS() ==
                  SPIRVEIS_NonSemantic_Shader_DebugInfo_100 ||
              BM->getDebugInfoEIS() ==
                  SPIRVEIS_NonSemantic_Shader_DebugInfo_200)
            BM->addDebugLine(V, getVoidTy(),
                             File ? File->getId() : getDebugInfoNoneId(),
                             LineNo, LineNo, Col, Col + 1);
          else
            BM->addLine(V, File ? File->getId() : getDebugInfoNoneId(), LineNo,
                        Col);
        }
      } // Instructions
    }   // Basic Blocks
  }     // Functions
}

// Translation of single debug entry

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgEntry(const MDNode *DIEntry) {
  // Caching
  auto It = MDMap.find(DIEntry);
  if (It != MDMap.end()) {
    assert(It->second && "Invalid SPIRVEntry is cached!");
    return It->second;
  }
  SPIRVEntry *Res = transDbgEntryImpl(DIEntry);
  assert(Res && "Translation failure");
  // We might end up having a recursive debug info generation like the
  // following:
  // translation of DIDerivedType (member) calls DICompositeType translation
  // as its parent scope;
  // translation of DICompositeType calls translation of its members
  // (DIDerivedType with member tag).
  // Here we make only the latest of these instructions be cached and hence
  // reused
  // FIXME: find a way to not create dead instruction
  if (MDMap[DIEntry])
    return MDMap[DIEntry];
  MDMap[DIEntry] = Res;
  return Res;
}

// Dispatcher implementation

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgEntryImpl(const MDNode *MDN) {
  if (!MDN)
    return BM->addDebugInfo(SPIRVDebug::DebugInfoNone, getVoidTy(),
                            SPIRVWordVec());
  if (isNonSemanticDebugInfo())
    BM->addExtension(SPIRV::ExtensionID::SPV_KHR_non_semantic_info);
  if (const DINode *DIEntry = dyn_cast<DINode>(MDN)) {
    switch (DIEntry->getTag()) {
    // Types
    case dwarf::DW_TAG_base_type:
    case dwarf::DW_TAG_unspecified_type:
      return transDbgBaseType(cast<DIBasicType>(DIEntry));

    case dwarf::DW_TAG_reference_type:
    case dwarf::DW_TAG_rvalue_reference_type:
    case dwarf::DW_TAG_pointer_type:
      return transDbgPointerType(cast<DIDerivedType>(DIEntry));

    case dwarf::DW_TAG_array_type:
      return transDbgArrayType(cast<DICompositeType>(DIEntry));

    case dwarf::DW_TAG_subrange_type:
      if (BM->getDebugInfoEIS() == SPIRVEIS_NonSemantic_Shader_DebugInfo_200)
        return transDbgSubrangeType(cast<DISubrange>(DIEntry));
      else
        return getDebugInfoNone();

    case dwarf::DW_TAG_string_type: {
      if (BM->getDebugInfoEIS() == SPIRVEIS_NonSemantic_Shader_DebugInfo_200)
        return transDbgStringType(cast<DIStringType>(DIEntry));
      return getDebugInfoNone();
    }

    case dwarf::DW_TAG_const_type:
    case dwarf::DW_TAG_restrict_type:
    case dwarf::DW_TAG_volatile_type:
    case dwarf::DW_TAG_atomic_type:
      return transDbgQualifiedType(cast<DIDerivedType>(DIEntry));

    case dwarf::DW_TAG_subroutine_type:
      return transDbgSubroutineType(cast<DISubroutineType>(DIEntry));

    case dwarf::DW_TAG_class_type:
    case dwarf::DW_TAG_structure_type:
    case dwarf::DW_TAG_union_type:
      return transDbgCompositeType(cast<DICompositeType>(DIEntry));

    case dwarf::DW_TAG_member:
      return transDbgMemberType(cast<DIDerivedType>(DIEntry));

    case dwarf::DW_TAG_inheritance:
      return transDbgInheritance(cast<DIDerivedType>(DIEntry));

    case dwarf::DW_TAG_enumeration_type:
      return transDbgEnumType(cast<DICompositeType>(DIEntry));

    case dwarf::DW_TAG_file_type:
      return transDbgFileType(cast<DIFile>(DIEntry));

    case dwarf::DW_TAG_typedef:
      return transDbgTypeDef(cast<DIDerivedType>(DIEntry));

    case dwarf::DW_TAG_ptr_to_member_type:
      return transDbgPtrToMember(cast<DIDerivedType>(DIEntry));

    // Scope
    case dwarf::DW_TAG_namespace:
    case dwarf::DW_TAG_lexical_block:
      return transDbgScope(cast<DIScope>(DIEntry));

    // Function
    case dwarf::DW_TAG_subprogram:
      return transDbgFunction(cast<DISubprogram>(DIEntry));

    // Variables
    case dwarf::DW_TAG_variable:
      if (const DILocalVariable *LV = dyn_cast<DILocalVariable>(DIEntry))
        return transDbgLocalVariable(LV);
      if (const DIGlobalVariable *GV = dyn_cast<DIGlobalVariable>(DIEntry))
        return transDbgGlobalVariable(GV);
      if (const DIDerivedType *MT = dyn_cast<DIDerivedType>(DIEntry))
        if (M->getDwarfVersion() >= 5 && MT->isStaticMember())
          return transDbgMemberType(MT);
      llvm_unreachable("Unxpected debug info type for variable");
    case dwarf::DW_TAG_formal_parameter:
      return transDbgLocalVariable(cast<DILocalVariable>(DIEntry));

    // Compilation unit
    case dwarf::DW_TAG_compile_unit:
      return transDbgCompileUnit(cast<DICompileUnit>(DIEntry));

    // Templates
    case dwarf::DW_TAG_template_type_parameter:
    case dwarf::DW_TAG_template_value_parameter:
      return transDbgTemplateParameter(cast<DITemplateParameter>(DIEntry));
    case dwarf::DW_TAG_GNU_template_template_param:
      return transDbgTemplateTemplateParameter(
          cast<DITemplateValueParameter>(DIEntry));
    case dwarf::DW_TAG_GNU_template_parameter_pack:
      return transDbgTemplateParameterPack(
          cast<DITemplateValueParameter>(DIEntry));

    case dwarf::DW_TAG_imported_module:
    case dwarf::DW_TAG_imported_declaration:
      return transDbgImportedEntry(cast<DIImportedEntity>(DIEntry));

    case dwarf::DW_TAG_module: {
      if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_debug_module) ||
          BM->getDebugInfoEIS() == SPIRVEIS_NonSemantic_Shader_DebugInfo_200)
        return transDbgModule(cast<DIModule>(DIEntry));
      return getDebugInfoNone();
    }

    default:
      return getDebugInfoNone();
    }
  }
  if (const DIExpression *Expr = dyn_cast<DIExpression>(MDN))
    return transDbgExpression(Expr);

  if (const DILocation *Loc = dyn_cast<DILocation>(MDN)) {
    return transDbgInlinedAt(Loc);
  }
  llvm_unreachable("Not implemented debug info entry!");
}

// Helper methods

SPIRVType *LLVMToSPIRVDbgTran::getVoidTy() {
  if (!VoidT) {
    assert(M && "Pointer to LLVM Module is expected to be initialized!");
    // Cache void type in a member.
    VoidT = SPIRVWriter->transType(Type::getVoidTy(M->getContext()));
  }
  return VoidT;
}

SPIRVType *LLVMToSPIRVDbgTran::getInt32Ty() {
  if (!Int32T) {
    assert(M && "Pointer to LLVM Module is expected to be initialized!");
    // Cache int32 type in a member.
    Int32T = SPIRVWriter->transType(Type::getInt32Ty(M->getContext()));
  }
  return Int32T;
}

SPIRVEntry *LLVMToSPIRVDbgTran::getScope(DIScope *S) {
  if (S)
    return transDbgEntry(S);
  assert(!SPIRVCUMap.empty() &&
         "Compile units are expected to be already translated");
  return SPIRVCUMap.begin()->second;
}

SPIRVEntry *LLVMToSPIRVDbgTran::getGlobalVariable(const DIGlobalVariable *GV) {
  for (GlobalVariable &V : M->globals()) {
    SmallVector<DIGlobalVariableExpression *, 4> GVs;
    V.getDebugInfo(GVs);
    for (DIGlobalVariableExpression *GVE : GVs) {
      if (GVE->getVariable() == GV)
        return SPIRVWriter->transValue(&V, nullptr);
    }
  }
  return getDebugInfoNone();
}

inline bool LLVMToSPIRVDbgTran::isNonSemanticDebugInfo() {
  return (BM->getDebugInfoEIS() == SPIRVEIS_NonSemantic_Shader_DebugInfo_100 ||
          BM->getDebugInfoEIS() == SPIRVEIS_NonSemantic_Shader_DebugInfo_200);
}

void LLVMToSPIRVDbgTran::transformToConstant(std::vector<SPIRVWord> &Ops,
                                             std::vector<SPIRVWord> Idxs) {
  for (const auto Idx : Idxs) {
    SPIRVValue *Const = BM->addIntegerConstant(
        static_cast<SPIRVTypeInt *>(getInt32Ty()), Ops[Idx]);
    Ops[Idx] = Const->getId();
  }
}

SPIRVWord LLVMToSPIRVDbgTran::mapDebugFlags(DINode::DIFlags DFlags) {
  SPIRVWord Flags = 0;
  if ((DFlags & DINode::FlagAccessibility) == DINode::FlagPublic)
    Flags |= SPIRVDebug::FlagIsPublic;
  if ((DFlags & DINode::FlagAccessibility) == DINode::FlagProtected)
    Flags |= SPIRVDebug::FlagIsProtected;
  if ((DFlags & DINode::FlagAccessibility) == DINode::FlagPrivate)
    Flags |= SPIRVDebug::FlagIsPrivate;

  if (DFlags & DINode::FlagFwdDecl)
    Flags |= SPIRVDebug::FlagIsFwdDecl;
  if (DFlags & DINode::FlagArtificial)
    Flags |= SPIRVDebug::FlagIsArtificial;
  if (DFlags & DINode::FlagExplicit)
    Flags |= SPIRVDebug::FlagIsExplicit;
  if (DFlags & DINode::FlagPrototyped)
    Flags |= SPIRVDebug::FlagIsPrototyped;
  if (DFlags & DINode::FlagObjectPointer)
    Flags |= SPIRVDebug::FlagIsObjectPointer;
  if (DFlags & DINode::FlagStaticMember)
    Flags |= SPIRVDebug::FlagIsStaticMember;
  // inderect variable flag ?
  if (DFlags & DINode::FlagLValueReference)
    Flags |= SPIRVDebug::FlagIsLValueReference;
  if (DFlags & DINode::FlagRValueReference)
    Flags |= SPIRVDebug::FlagIsRValueReference;
  if (DFlags & DINode::FlagTypePassByValue)
    Flags |= SPIRVDebug::FlagTypePassByValue;
  if (DFlags & DINode::FlagTypePassByReference)
    Flags |= SPIRVDebug::FlagTypePassByReference;
  if (BM->getDebugInfoEIS() == SPIRVEIS_NonSemantic_Shader_DebugInfo_200)
    if (DFlags & DINode::FlagBitField)
      Flags |= SPIRVDebug::FlagBitField;
  return Flags;
}

SPIRVWord LLVMToSPIRVDbgTran::transDebugFlags(const DINode *DN) {
  SPIRVWord Flags = 0;
  if (const DIGlobalVariable *GV = dyn_cast<DIGlobalVariable>(DN)) {
    if (GV->isLocalToUnit())
      Flags |= SPIRVDebug::FlagIsLocal;
    if (GV->isDefinition())
      Flags |= SPIRVDebug::FlagIsDefinition;
  }
  if (const DISubprogram *DS = dyn_cast<DISubprogram>(DN)) {
    if (DS->isLocalToUnit())
      Flags |= SPIRVDebug::FlagIsLocal;
    if (DS->isOptimized())
      Flags |= SPIRVDebug::FlagIsOptimized;
    if (DS->isDefinition())
      Flags |= SPIRVDebug::FlagIsDefinition;
    Flags |= mapDebugFlags(DS->getFlags());
  }
  if (DN->getTag() == dwarf::DW_TAG_reference_type)
    Flags |= SPIRVDebug::FlagIsLValueReference;
  if (DN->getTag() == dwarf::DW_TAG_rvalue_reference_type)
    Flags |= SPIRVDebug::FlagIsRValueReference;
  if (const DIType *DT = dyn_cast<DIType>(DN))
    Flags |= mapDebugFlags(DT->getFlags());
  if (const DILocalVariable *DLocVar = dyn_cast<DILocalVariable>(DN))
    Flags |= mapDebugFlags(DLocVar->getFlags());

  return Flags;
}

/// Clang doesn't emit access flags for members with default access specifier
/// See clang/lib/CodeGen/CGDebugInfo.cpp: getAccessFlag()
/// In SPIR-V we set the flags even for members with default access specifier
SPIRVWord adjustAccessFlags(DIScope *Scope, SPIRVWord Flags) {
  if (Scope && (Flags & SPIRVDebug::FlagAccess) == 0) {
    unsigned Tag = Scope->getTag();
    if (Tag == dwarf::DW_TAG_class_type)
      Flags |= SPIRVDebug::FlagIsPrivate;
    else if (Tag == dwarf::DW_TAG_structure_type ||
             Tag == dwarf::DW_TAG_union_type)
      Flags |= SPIRVDebug::FlagIsPublic;
  }
  return Flags;
}

// Fortran dynamic arrays can have following 'dataLocation', 'associated'
// 'allocated' and 'rank' debug metadata. Such arrays are being mapped on
// DebugTypeArrayDynamic from NonSemantic.Shader.200 debug spec
inline bool isFortranArrayDynamic(const DICompositeType *AT) {
  return (AT->getRawDataLocation() || AT->getRawAssociated() ||
          AT->getRawAllocated() || AT->getRawRank());
}

/// The following methods (till the end of the file) implement translation of
/// debug instrtuctions described in the spec.

// Absent Debug Info

SPIRVEntry *LLVMToSPIRVDbgTran::getDebugInfoNone() {
  if (!DebugInfoNone) {
    DebugInfoNone = transDbgEntry(nullptr);
  }
  return DebugInfoNone;
}

SPIRVId LLVMToSPIRVDbgTran::getDebugInfoNoneId() {
  return getDebugInfoNone()->getId();
}

// Compilation unit

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgCompileUnit(const DICompileUnit *CU) {
  using namespace SPIRVDebug::Operand::CompilationUnit;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[SPIRVDebugInfoVersionIdx] = SPIRVDebug::DebugInfoVersion;
  Ops[DWARFVersionIdx] = M->getDwarfVersion();
  Ops[SourceIdx] = getSource(CU)->getId();

  if (isNonSemanticDebugInfo())
    generateBuildIdentifierAndStoragePath(CU);

  auto DwarfLang =
      static_cast<llvm::dwarf::SourceLanguage>(CU->getSourceLanguage());
  Ops[LanguageIdx] =
      BM->getDebugInfoEIS() == SPIRVEIS_NonSemantic_Shader_DebugInfo_200
          ? convertDWARFSourceLangToSPIRVNonSemanticDbgInfo(DwarfLang)
          : convertDWARFSourceLangToSPIRV(DwarfLang);
  if (isNonSemanticDebugInfo())
    transformToConstant(
        Ops, {SPIRVDebugInfoVersionIdx, DWARFVersionIdx, LanguageIdx});

  if (isNonSemanticDebugInfo()) {
    if (BM->getDebugInfoEIS() == SPIRVEIS_NonSemantic_Shader_DebugInfo_200) {
      Ops.push_back(BM->getString(CU->getProducer().str())->getId());
    }
  } else {
    // TODO: Remove this workaround once we switch to NonSemantic.Shader.* debug
    // info by default
    BM->addModuleProcessed(SPIRVDebug::ProducerPrefix +
                           CU->getProducer().str());
  }
  // Cache CU in a member.
  SPIRVCUMap[CU] = static_cast<SPIRVExtInst *>(
      BM->addDebugInfo(SPIRVDebug::CompilationUnit, getVoidTy(), Ops));
  return SPIRVCUMap[CU];
}

// Types

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgBaseType(const DIBasicType *BT) {
  using namespace SPIRVDebug::Operand::TypeBasic;
  SPIRVWordVec Ops(OperandCountOCL);
  Ops[NameIdx] = BM->getString(BT->getName().str())->getId();
  ConstantInt *Size = getUInt(M, BT->getSizeInBits());
  Ops[SizeIdx] = SPIRVWriter->transValue(Size, nullptr)->getId();
  auto Encoding = static_cast<dwarf::TypeKind>(BT->getEncoding());
  SPIRVDebug::EncodingTag EncTag = SPIRVDebug::Unspecified;
  SPIRV::DbgEncodingMap::find(Encoding, &EncTag);
  // Unset encoding if it's complex and NonSemantic.Shader.DebugInfo.200 is not
  // enabled
  if (EncTag == SPIRVDebug::Complex &&
      BM->getDebugInfoEIS() != SPIRVEIS_NonSemantic_Shader_DebugInfo_200)
    EncTag = SPIRVDebug::Unspecified;
  Ops[EncodingIdx] = EncTag;
  if (isNonSemanticDebugInfo()) {
    transformToConstant(Ops, {EncodingIdx});
    // Flags value could not be generated by clang or by LLVM environment.
    Ops.push_back(getDebugInfoNoneId());
  }
  return BM->addDebugInfo(SPIRVDebug::TypeBasic, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgPointerType(const DIDerivedType *PT) {
  using namespace SPIRVDebug::Operand::TypePointer;
  SPIRVWordVec Ops(OperandCount);
  SPIRVEntry *Base = transDbgEntry(PT->getBaseType());
  Ops[BaseTypeIdx] = Base->getId();
  Ops[StorageClassIdx] = ~0U; // all ones denote no address space
  std::optional<unsigned> AS = PT->getDWARFAddressSpace();
  if (AS.has_value()) {
    SPIRAddressSpace SPIRAS = static_cast<SPIRAddressSpace>(AS.value());
    Ops[StorageClassIdx] = SPIRSPIRVAddrSpaceMap::map(SPIRAS);
  }
  Ops[FlagsIdx] = transDebugFlags(PT);
  if (isNonSemanticDebugInfo())
    transformToConstant(Ops, {StorageClassIdx, FlagsIdx});
  SPIRVEntry *Res = BM->addDebugInfo(SPIRVDebug::TypePointer, getVoidTy(), Ops);
  return Res;
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgQualifiedType(const DIDerivedType *QT) {
  using namespace SPIRVDebug::Operand::TypeQualifier;
  SPIRVWordVec Ops(OperandCount);
  SPIRVEntry *Base = transDbgEntry(QT->getBaseType());
  Ops[BaseTypeIdx] = Base->getId();
  Ops[QualifierIdx] = SPIRV::DbgTypeQulifierMap::map(
      static_cast<llvm::dwarf::Tag>(QT->getTag()));
  if (isNonSemanticDebugInfo())
    transformToConstant(Ops, {QualifierIdx});
  return BM->addDebugInfo(SPIRVDebug::TypeQualifier, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgArrayType(const DICompositeType *AT) {
  if (BM->getDebugInfoEIS() == SPIRVEIS_NonSemantic_Shader_DebugInfo_200) {
    if (isFortranArrayDynamic(AT))
      return transDbgArrayTypeDynamic(AT);
    return transDbgArrayTypeNonSemantic(AT);
  }

  return transDbgArrayTypeOpenCL(AT);
}

SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgArrayTypeOpenCL(const DICompositeType *AT) {
  using namespace SPIRVDebug::Operand::TypeArray;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[BaseTypeIdx] = transDbgEntry(AT->getBaseType())->getId();

  DINodeArray AR(AT->getElements());
  // For N-dimensianal arrays AR.getNumElements() == N
  const unsigned N = AR.size();
  Ops.resize(ComponentCountIdx + N);
  SPIRVWordVec LowerBounds(N);
  for (unsigned I = 0; I < N; ++I) {
    DISubrange *SR = cast<DISubrange>(AR[I]);
    ConstantInt *Count = SR->getCount().get<ConstantInt *>();
    if (AT->isVector()) {
      assert(N == 1 && "Multidimensional vector is not expected!");
      Ops[ComponentCountIdx] = static_cast<SPIRVWord>(Count->getZExtValue());
      if (isNonSemanticDebugInfo())
        transformToConstant(Ops, {ComponentCountIdx});
      return BM->addDebugInfo(SPIRVDebug::TypeVector, getVoidTy(), Ops);
    }
    if (Count) {
      Ops[ComponentCountIdx + I] =
          SPIRVWriter->transValue(Count, nullptr)->getId();
    } else {
      if (auto *UpperBound = dyn_cast<MDNode>(SR->getRawUpperBound()))
        Ops[ComponentCountIdx + I] = transDbgEntry(UpperBound)->getId();
      else
        Ops[ComponentCountIdx + I] = getDebugInfoNoneId();
    }
    if (auto *RawLB = SR->getRawLowerBound()) {
      if (auto *DIExprLB = dyn_cast<MDNode>(RawLB))
        LowerBounds[I] = transDbgEntry(DIExprLB)->getId();
      else {
        ConstantInt *ConstIntLB = SR->getLowerBound().get<ConstantInt *>();
        LowerBounds[I] = SPIRVWriter->transValue(ConstIntLB, nullptr)->getId();
      }
    } else {
      LowerBounds[I] = getDebugInfoNoneId();
    }
  }
  Ops.insert(Ops.end(), LowerBounds.begin(), LowerBounds.end());
  return BM->addDebugInfo(SPIRVDebug::TypeArray, getVoidTy(), Ops);
}

SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgArrayTypeNonSemantic(const DICompositeType *AT) {
  using namespace SPIRVDebug::Operand::TypeArray;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[BaseTypeIdx] = transDbgEntry(AT->getBaseType())->getId();

  DINodeArray AR(AT->getElements());
  // For N-dimensianal arrays AR.getNumElements() == N
  const unsigned N = AR.size();
  Ops.resize(SubrangesIdx + N);
  for (unsigned I = 0; I < N; ++I) {
    DISubrange *SR = cast<DISubrange>(AR[I]);
    ConstantInt *Count = SR->getCount().get<ConstantInt *>();
    if (AT->isVector()) {
      assert(N == 1 && "Multidimensional vector is not expected!");
      Ops[ComponentCountIdx] = static_cast<SPIRVWord>(Count->getZExtValue());
      if (isNonSemanticDebugInfo())
        transformToConstant(Ops, {ComponentCountIdx});
      return BM->addDebugInfo(SPIRVDebug::TypeVector, getVoidTy(), Ops);
    }
    Ops[SubrangesIdx + I] = transDbgEntry(SR)->getId();
  }
  return BM->addDebugInfo(SPIRVDebug::TypeArray, getVoidTy(), Ops);
}

// The function is used to translate Fortran's dynamic arrays
SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgArrayTypeDynamic(const DICompositeType *AT) {
  using namespace SPIRVDebug::Operand::TypeArrayDynamic;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[BaseTypeIdx] = transDbgEntry(AT->getBaseType())->getId();

  // DataLocation, Associated, Allocated and Rank can be either DIExpression
  // metadata or DIVariable
  auto TransOperand = [&](llvm::Metadata *DIMD) -> SPIRVWord {
    if (auto *DIExpr = dyn_cast_or_null<DIExpression>(DIMD))
      return transDbgExpression(DIExpr)->getId();
    if (auto *DIVar = dyn_cast_or_null<DIVariable>(DIMD)) {
      if (const DILocalVariable *LV = dyn_cast<DILocalVariable>(DIVar))
        return transDbgLocalVariable(LV)->getId();
      if (const DIGlobalVariable *GV = dyn_cast<DIGlobalVariable>(DIVar))
        return transDbgGlobalVariable(GV)->getId();
    }
    return getDebugInfoNoneId();
  };

  Ops[DataLocationIdx] = TransOperand(AT->getRawDataLocation());
  Ops[AssociatedIdx] = TransOperand(AT->getRawAssociated());
  Ops[AllocatedIdx] = TransOperand(AT->getRawAllocated());
  Ops[RankIdx] = TransOperand(AT->getRawRank());

  DINodeArray AR(AT->getElements());
  // For N-dimensianal arrays AR.getNumElements() == N
  const unsigned N = AR.size();
  Ops.resize(SubrangesIdx + N);
  for (unsigned I = 0; I < N; ++I) {
    DISubrange *SR = cast<DISubrange>(AR[I]);
    Ops[SubrangesIdx + I] = transDbgEntry(SR)->getId();
  }
  return BM->addDebugInfo(SPIRVDebug::TypeArrayDynamic, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgSubrangeType(const DISubrange *ST) {
  using namespace SPIRVDebug::Operand::TypeSubrange;
  SPIRVWordVec Ops(MinOperandCount);
  auto TransOperand = [&Ops, this, ST](int Idx) -> void {
    Metadata *RawNode = nullptr;
    switch (Idx) {
    case LowerBoundIdx:
      RawNode = ST->getRawLowerBound();
      break;
    case UpperBoundIdx:
      RawNode = ST->getRawUpperBound();
      break;
    case CountIdx:
      RawNode = ST->getRawCountNode();
      break;
    }
    if (!RawNode) {
      Ops[Idx] = getDebugInfoNoneId();
      return;
    }
    if (auto *Node = dyn_cast<MDNode>(RawNode)) {
      Ops[Idx] = transDbgEntry(Node)->getId();
    } else {
      ConstantInt *IntNode = nullptr;
      switch (Idx) {
      case LowerBoundIdx:
        IntNode = ST->getLowerBound().get<ConstantInt *>();
        break;
      case UpperBoundIdx:
        IntNode = ST->getUpperBound().get<ConstantInt *>();
        break;
      case CountIdx:
        IntNode = ST->getCount().get<ConstantInt *>();
        break;
      }
      Ops[Idx] = IntNode ? SPIRVWriter->transValue(IntNode, nullptr)->getId()
                         : getDebugInfoNoneId();
    }
  };
  for (int Idx = 0; Idx < MinOperandCount; ++Idx)
    TransOperand(Idx);
  if (auto *RawNode = ST->getRawStride()) {
    Ops.resize(MaxOperandCount);
    if (auto *Node = dyn_cast<MDNode>(RawNode))
      Ops[StrideIdx] = transDbgEntry(Node)->getId();
    else
      Ops[StrideIdx] =
          SPIRVWriter->transValue(ST->getStride().get<ConstantInt *>(), nullptr)
              ->getId();
  }
  return BM->addDebugInfo(SPIRVDebug::TypeSubrange, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgStringType(const DIStringType *ST) {
  using namespace SPIRVDebug::Operand::TypeString;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[NameIdx] = BM->getString(ST->getName().str())->getId();

  Ops[BaseTypeIdx] = ST->getEncoding()
                         ? getDebugInfoNoneId() /*TODO: replace with basetype*/
                         : getDebugInfoNoneId();

  auto TransOperand = [&](llvm::Metadata *DIMD) -> SPIRVWord {
    if (auto *DIExpr = dyn_cast_or_null<DIExpression>(DIMD))
      return transDbgExpression(DIExpr)->getId();
    if (auto *DIVar = dyn_cast_or_null<DIVariable>(DIMD)) {
      if (const DILocalVariable *LV = dyn_cast<DILocalVariable>(DIVar))
        return transDbgLocalVariable(LV)->getId();
      if (const DIGlobalVariable *GV = dyn_cast<DIGlobalVariable>(DIVar))
        return transDbgGlobalVariable(GV)->getId();
    }
    return getDebugInfoNoneId();
  };

  Ops[DataLocationIdx] = TransOperand(ST->getRawStringLocationExp());

  ConstantInt *Size = getUInt(M, ST->getSizeInBits());
  Ops[SizeIdx] = SPIRVWriter->transValue(Size, nullptr)->getId();

  if (auto *StrLengthExp = ST->getRawStringLengthExp()) {
    Ops[LengthAddrIdx] = TransOperand(StrLengthExp);
  } else if (auto *StrLengthVar = ST->getRawStringLength()) {
    Ops[LengthAddrIdx] = TransOperand(StrLengthVar);
  } else {
    Ops[LengthAddrIdx] = getDebugInfoNoneId();
  }

  return BM->addDebugInfo(SPIRVDebug::TypeString, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgTypeDef(const DIDerivedType *DT) {
  using namespace SPIRVDebug::Operand::Typedef;
  SPIRVWordVec Ops(OperandCount);
  Ops[NameIdx] = BM->getString(DT->getName().str())->getId();
  SPIRVEntry *BaseTy = transDbgEntry(DT->getBaseType());
  assert(BaseTy && "Couldn't translate base type!");
  Ops[BaseTypeIdx] = BaseTy->getId();
  Ops[SourceIdx] = getSource(DT)->getId();
  Ops[LineIdx] = 0;   // This version of DIDerivedType has no line number
  Ops[ColumnIdx] = 0; // This version of DIDerivedType has no column number
  SPIRVEntry *Scope = getScope(DT->getScope());
  assert(Scope && "Couldn't translate scope!");
  Ops[ParentIdx] = Scope->getId();
  if (isNonSemanticDebugInfo())
    transformToConstant(Ops, {LineIdx, ColumnIdx});
  return BM->addDebugInfo(SPIRVDebug::Typedef, getVoidTy(), Ops);
}

SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgSubroutineType(const DISubroutineType *FT) {
  using namespace SPIRVDebug::Operand::TypeFunction;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[FlagsIdx] = transDebugFlags(FT);

  DITypeRefArray Types = FT->getTypeArray();
  const size_t NumElements = Types.size();
  if (NumElements) {
    Ops.resize(1 + NumElements);
    // First element of the TypeArray is the type of the return value,
    // followed by types of the function arguments' types.
    // The same order is preserved in SPIRV.
    for (unsigned I = 0; I < NumElements; ++I)
      Ops[ReturnTypeIdx + I] = transDbgEntry(Types[I])->getId();
  } else { // void foo();
    Ops[ReturnTypeIdx] = getVoidTy()->getId();
  }

  if (isNonSemanticDebugInfo())
    transformToConstant(Ops, {FlagsIdx});
  return BM->addDebugInfo(SPIRVDebug::TypeFunction, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgEnumType(const DICompositeType *ET) {
  using namespace SPIRVDebug::Operand::TypeEnum;
  SPIRVWordVec Ops(MinOperandCount);

  SPIRVEntry *UnderlyingType = getVoidTy();
  if (DIType *DerivedFrom = ET->getBaseType())
    UnderlyingType = transDbgEntry(DerivedFrom);
  ConstantInt *Size = getUInt(M, ET->getSizeInBits());

  Ops[NameIdx] = BM->getString(ET->getName().str())->getId();
  Ops[UnderlyingTypeIdx] = UnderlyingType->getId();
  Ops[SourceIdx] = getSource(ET)->getId();
  Ops[LineIdx] = ET->getLine();
  Ops[ColumnIdx] = 0; // This version of DICompositeType has no column number
  Ops[ParentIdx] = getScope(ET->getScope())->getId();
  Ops[SizeIdx] = SPIRVWriter->transValue(Size, nullptr)->getId();
  Ops[FlagsIdx] = transDebugFlags(ET);

  DINodeArray Elements = ET->getElements();
  size_t ElemCount = Elements.size();
  for (unsigned I = 0; I < ElemCount; ++I) {
    DIEnumerator *E = cast<DIEnumerator>(Elements[I]);
    ConstantInt *EnumValue = getInt(M, E->getValue().getSExtValue());
    SPIRVValue *Val = SPIRVWriter->transValue(EnumValue, nullptr);
    assert(Val->getOpCode() == OpConstant &&
           "LLVM constant must be translated to SPIRV constant");
    Ops.push_back(Val->getId());
    SPIRVString *Name = BM->getString(E->getName().str());
    Ops.push_back(Name->getId());
  }
  if (isNonSemanticDebugInfo())
    transformToConstant(Ops, {LineIdx, ColumnIdx, FlagsIdx});
  return BM->addDebugInfo(SPIRVDebug::TypeEnum, getVoidTy(), Ops);
}

SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgCompositeType(const DICompositeType *CT) {
  using namespace SPIRVDebug::Operand::TypeComposite;
  SPIRVWordVec Ops(MinOperandCount);

  SPIRVForward *Tmp = BM->addForward(nullptr);
  MDMap.insert(std::make_pair(CT, Tmp));

  auto Tag = static_cast<dwarf::Tag>(CT->getTag());
  SPIRVId UniqId = getDebugInfoNoneId();
  StringRef Identifier = CT->getIdentifier();
  if (!Identifier.empty())
    UniqId = BM->getString(Identifier.str())->getId();
  ConstantInt *Size = getUInt(M, CT->getSizeInBits());

  Ops[NameIdx] = BM->getString(CT->getName().str())->getId();
  Ops[TagIdx] = SPIRV::DbgCompositeTypeMap::map(Tag);
  Ops[SourceIdx] = getSource(CT)->getId();
  Ops[LineIdx] = CT->getLine();
  Ops[ColumnIdx] = 0; // This version of DICompositeType has no column number
  Ops[ParentIdx] = getScope(CT->getScope())->getId();
  Ops[LinkageNameIdx] = UniqId;
  Ops[SizeIdx] = SPIRVWriter->transValue(Size, nullptr)->getId();
  Ops[FlagsIdx] = transDebugFlags(CT);

  for (DINode *N : CT->getElements()) {
    Ops.push_back(transDbgEntry(N)->getId());
  }

  if (isNonSemanticDebugInfo())
    transformToConstant(Ops, {TagIdx, LineIdx, ColumnIdx, FlagsIdx});
  SPIRVEntry *Res =
      BM->addDebugInfo(SPIRVDebug::TypeComposite, getVoidTy(), Ops);

  // Translate template parameters.
  if (DITemplateParameterArray TP = CT->getTemplateParams()) {
    const unsigned int NumTParams = TP.size();
    SPIRVWordVec Args(1 + NumTParams);
    Args[0] = Res->getId();
    for (unsigned int I = 0; I < NumTParams; ++I) {
      Args[I + 1] = transDbgEntry(TP[I])->getId();
    }
    Res = BM->addDebugInfo(SPIRVDebug::TypeTemplate, getVoidTy(), Args);
  }
  BM->replaceForward(Tmp, Res);
  MDMap[CT] = Res;
  return Res;
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgMemberType(const DIDerivedType *MT) {
  if (isNonSemanticDebugInfo())
    return transDbgMemberTypeNonSemantic(MT);
  return transDbgMemberTypeOpenCL(MT);
}

SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgMemberTypeOpenCL(const DIDerivedType *MT) {
  using namespace SPIRVDebug::Operand::TypeMember::OpenCL;
  SPIRVWordVec Ops(MinOperandCount);

  Ops[NameIdx] = BM->getString(MT->getName().str())->getId();
  Ops[TypeIdx] = transDbgEntry(MT->getBaseType())->getId();
  Ops[SourceIdx] = getSource(MT)->getId();
  Ops[LineIdx] = MT->getLine();
  Ops[ColumnIdx] = 0; // This version of DIDerivedType has no column number
  Ops[ParentIdx] = transDbgEntry(MT->getScope())->getId();
  ConstantInt *Offset = getUInt(M, MT->getOffsetInBits());
  Ops[OffsetIdx] = SPIRVWriter->transValue(Offset, nullptr)->getId();
  ConstantInt *Size = getUInt(M, MT->getSizeInBits());
  Ops[SizeIdx] = SPIRVWriter->transValue(Size, nullptr)->getId();
  Ops[FlagsIdx] = adjustAccessFlags(MT->getScope(), transDebugFlags(MT));
  if (MT->isStaticMember()) {
    if (llvm::Constant *C = MT->getConstant()) {
      SPIRVValue *Val = SPIRVWriter->transValue(C, nullptr);
      assert(isConstantOpCode(Val->getOpCode()) &&
             "LLVM constant must be translated to SPIRV constant");
      Ops.push_back(Val->getId());
    }
  }
  return BM->addDebugInfo(SPIRVDebug::TypeMember, getVoidTy(), Ops);
}

SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgMemberTypeNonSemantic(const DIDerivedType *MT) {
  using namespace SPIRVDebug::Operand::TypeMember::NonSemantic;
  SPIRVWordVec Ops(MinOperandCount);

  Ops[NameIdx] = BM->getString(MT->getName().str())->getId();
  Ops[TypeIdx] = transDbgEntry(MT->getBaseType())->getId();
  Ops[SourceIdx] = getSource(MT)->getId();
  Ops[LineIdx] = MT->getLine();
  Ops[ColumnIdx] = 0; // This version of DIDerivedType has no column number
  ConstantInt *Offset = getUInt(M, MT->getOffsetInBits());
  Ops[OffsetIdx] = SPIRVWriter->transValue(Offset, nullptr)->getId();
  ConstantInt *Size = getUInt(M, MT->getSizeInBits());
  Ops[SizeIdx] = SPIRVWriter->transValue(Size, nullptr)->getId();
  Ops[FlagsIdx] = adjustAccessFlags(MT->getScope(), transDebugFlags(MT));
  transDbgEntry(MT->getScope())->getId();
  if (MT->isStaticMember()) {
    if (llvm::Constant *C = MT->getConstant()) {
      SPIRVValue *Val = SPIRVWriter->transValue(C, nullptr);
      assert(isConstantOpCode(Val->getOpCode()) &&
             "LLVM constant must be translated to SPIRV constant");
      Ops.push_back(Val->getId());
    }
  }
  transformToConstant(Ops, {LineIdx, ColumnIdx, FlagsIdx});
  return BM->addDebugInfo(SPIRVDebug::TypeMember, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgInheritance(const DIDerivedType *DT) {
  using namespace SPIRVDebug::Operand::TypeInheritance;
  const SPIRVWord Offset = isNonSemanticDebugInfo() ? 1 : 0;
  SPIRVWordVec Ops(OperandCount - Offset);
  // There is no Child operand in NonSemantic debug spec
  if (!isNonSemanticDebugInfo())
    Ops[ChildIdx] = transDbgEntry(DT->getScope())->getId();
  Ops[ParentIdx - Offset] = transDbgEntry(DT->getBaseType())->getId();
  ConstantInt *OffsetInBits = getUInt(M, DT->getOffsetInBits());
  Ops[OffsetIdx - Offset] =
      SPIRVWriter->transValue(OffsetInBits, nullptr)->getId();
  ConstantInt *Size = getUInt(M, DT->getSizeInBits());
  Ops[SizeIdx - Offset] = SPIRVWriter->transValue(Size, nullptr)->getId();
  Ops[FlagsIdx - Offset] = transDebugFlags(DT);
  if (isNonSemanticDebugInfo())
    transformToConstant(Ops, {FlagsIdx - Offset});
  return BM->addDebugInfo(SPIRVDebug::TypeInheritance, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgPtrToMember(const DIDerivedType *DT) {
  using namespace SPIRVDebug::Operand::TypePtrToMember;
  SPIRVWordVec Ops(OperandCount);
  Ops[MemberTypeIdx] = transDbgEntry(DT->getBaseType())->getId();
  Ops[ParentIdx] = transDbgEntry(DT->getClassType())->getId();
  return BM->addDebugInfo(SPIRVDebug::TypePtrToMember, getVoidTy(), Ops);
}

// Templates
SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgTemplateParams(DITemplateParameterArray TPA,
                                           const SPIRVEntry *Target) {
  using namespace SPIRVDebug::Operand::TypeTemplate;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[TargetIdx] = Target->getId();
  for (DITemplateParameter *TP : TPA) {
    Ops.push_back(transDbgEntry(TP)->getId());
  }
  return BM->addDebugInfo(SPIRVDebug::TypeTemplate, getVoidTy(), Ops);
}

SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgTemplateParameter(const DITemplateParameter *TP) {
  using namespace SPIRVDebug::Operand::TypeTemplateParameter;
  SPIRVWordVec Ops(OperandCount);
  Ops[NameIdx] = BM->getString(TP->getName().str())->getId();
  Ops[TypeIdx] = transDbgEntry(TP->getType())->getId();
  Ops[ValueIdx] = getDebugInfoNoneId();
  if (TP->getTag() == dwarf::DW_TAG_template_value_parameter) {
    const DITemplateValueParameter *TVP = cast<DITemplateValueParameter>(TP);
    if (auto *TVVal = TVP->getValue()) {
      Constant *C = cast<ConstantAsMetadata>(TVVal)->getValue();
      Ops[ValueIdx] = SPIRVWriter->transValue(C, nullptr)->getId();
    } else {
      SPIRVType *TyPtr = SPIRVWriter->transType(
          PointerType::get(Type::getInt8Ty(M->getContext()), 0));
      Ops[ValueIdx] = BM->addNullConstant(TyPtr)->getId();
    }
  }
  Ops[SourceIdx] = getDebugInfoNoneId();
  Ops[LineIdx] = 0;   // This version of DITemplateParameter has no line number
  Ops[ColumnIdx] = 0; // This version of DITemplateParameter has no column info
  if (isNonSemanticDebugInfo())
    transformToConstant(Ops, {LineIdx, ColumnIdx});
  return BM->addDebugInfo(SPIRVDebug::TypeTemplateParameter, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgTemplateTemplateParameter(
    const DITemplateValueParameter *TVP) {
  using namespace SPIRVDebug::Operand::TypeTemplateTemplateParameter;
  SPIRVWordVec Ops(OperandCount);
  assert(isa<MDString>(TVP->getValue()));
  MDString *Val = cast<MDString>(TVP->getValue());
  Ops[NameIdx] = BM->getString(TVP->getName().str())->getId();
  Ops[TemplateNameIdx] = BM->getString(Val->getString().str())->getId();
  Ops[SourceIdx] = getDebugInfoNoneId();
  Ops[LineIdx] = 0; // This version of DITemplateValueParameter has no line info
  Ops[ColumnIdx] = 0; // This version of DITemplateValueParameter has no column
  if (isNonSemanticDebugInfo())
    transformToConstant(Ops, {LineIdx, ColumnIdx});
  return BM->addDebugInfo(SPIRVDebug::TypeTemplateTemplateParameter,
                          getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgTemplateParameterPack(
    const DITemplateValueParameter *TVP) {
  using namespace SPIRVDebug::Operand::TypeTemplateParameterPack;
  SPIRVWordVec Ops(MinOperandCount);
  assert(isa<MDNode>(TVP->getValue()));
  MDNode *Params = cast<MDNode>(TVP->getValue());

  Ops[NameIdx] = BM->getString(TVP->getName().str())->getId();
  Ops[SourceIdx] = getDebugInfoNoneId();
  Ops[LineIdx] = 0; // This version of DITemplateValueParameter has no line info
  Ops[ColumnIdx] = 0; // This version of DITemplateValueParameter has no column

  for (const MDOperand &Op : Params->operands()) {
    SPIRVEntry *P = transDbgEntry(cast<DINode>(Op.get()));
    Ops.push_back(P->getId());
  }
  if (isNonSemanticDebugInfo())
    transformToConstant(Ops, {LineIdx, ColumnIdx});
  return BM->addDebugInfo(SPIRVDebug::TypeTemplateParameterPack, getVoidTy(),
                          Ops);
}

// Global objects

SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgGlobalVariable(const DIGlobalVariable *GV) {
  using namespace SPIRVDebug::Operand::GlobalVariable;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[NameIdx] = BM->getString(GV->getName().str())->getId();
  Ops[TypeIdx] = transDbgEntry(GV->getType())->getId();
  Ops[SourceIdx] = getSource(GV)->getId();
  Ops[LineIdx] = GV->getLine();
  Ops[ColumnIdx] = 0; // This version of DIGlobalVariable has no column number

  // Parent scope
  DIScope *Context = GV->getScope();
  SPIRVEntry *Parent = SPIRVCUMap.begin()->second;
  // Global variable may be declared in scope of a namespace or imported module,
  // it may also be a static variable declared in scope of a function.
  if (Context && (isa<DINamespace>(Context) || isa<DISubprogram>(Context) ||
                  isa<DIModule>(Context)))
    Parent = transDbgEntry(Context);
  Ops[ParentIdx] = Parent->getId();

  Ops[LinkageNameIdx] = BM->getString(GV->getLinkageName().str())->getId();
  Ops[VariableIdx] = getGlobalVariable(GV)->getId();
  Ops[FlagsIdx] = transDebugFlags(GV);

  // Check if GV is the definition of previously declared static member
  if (DIDerivedType *StaticMember = GV->getStaticDataMemberDeclaration())
    Ops.push_back(transDbgEntry(StaticMember)->getId());

  // Check if Ops[VariableIdx] has no information
  if (Ops[VariableIdx] == getDebugInfoNoneId()) {
    // Check if GV has an associated GVE with a non-empty DIExpression.
    // The non-empty DIExpression gives the initial value of the GV.
    for (const DIGlobalVariableExpression *GVE : DIF.global_variables()) {
      if ( // GVE matches GV
          GVE->getVariable() == GV &&
          // DIExpression is non-empty
          GVE->getExpression()->getNumElements()) {
        // Repurpose VariableIdx operand to hold the initial value held in the
        // GVE's DIExpression
        Ops[VariableIdx] = transDbgExpression(GVE->getExpression())->getId();
        break;
      }
    }
  }

  if (isNonSemanticDebugInfo())
    transformToConstant(Ops, {LineIdx, ColumnIdx, FlagsIdx});
  return BM->addDebugInfo(SPIRVDebug::GlobalVariable, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgFunction(const DISubprogram *Func) {
  auto It = MDMap.find(Func);
  if (It != MDMap.end())
    return static_cast<SPIRVValue *>(It->second);

  // As long as indexes of FunctionDeclaration operands match with Function
  using namespace SPIRVDebug::Operand::FunctionDeclaration;
  SPIRVWordVec Ops(OperandCount);
  Ops[NameIdx] = BM->getString(Func->getName().str())->getId();
  Ops[TypeIdx] = transDbgEntry(Func->getType())->getId();
  Ops[SourceIdx] = getSource(Func)->getId();
  Ops[LineIdx] = Func->getLine();
  Ops[ColumnIdx] = 0; // This version of DISubprogram has no column number
  auto *Scope = Func->getScope();
  if (Scope && !isa<DIFile>(Scope)) {
    Ops[ParentIdx] = getScope(Scope)->getId();
  } else {
    if (auto *Unit = Func->getUnit())
      Ops[ParentIdx] = SPIRVCUMap[Unit]->getId();
    else
      // it might so happen, that DISubprogram is missing Unit parameter
      Ops[ParentIdx] = SPIRVCUMap.begin()->second->getId();
  }
  Ops[LinkageNameIdx] = BM->getString(Func->getLinkageName().str())->getId();
  Ops[FlagsIdx] = adjustAccessFlags(Scope, transDebugFlags(Func));
  if (isNonSemanticDebugInfo())
    transformToConstant(Ops, {LineIdx, ColumnIdx, FlagsIdx});

  SPIRVEntry *DebugFunc = nullptr;
  SPIRVValue *FuncDef = nullptr;
  bool IsEntryPointKernel = false;
  if (!Func->isDefinition()) {
    DebugFunc =
        BM->addDebugInfo(SPIRVDebug::FunctionDeclaration, getVoidTy(), Ops);
  } else {
    // Here we add operands specific function definition
    using namespace SPIRVDebug::Operand::Function;
    Ops.resize(MinOperandCount);
    Ops[ScopeLineIdx] = Func->getScopeLine();
    if (isNonSemanticDebugInfo())
      transformToConstant(Ops, {ScopeLineIdx});

    Ops[FunctionIdIdx] = getDebugInfoNoneId();
    for (const llvm::Function &F : M->functions()) {
      if (Func->describes(&F)) {
        // Function definition of spir_kernel can have no "spir_kernel" calling
        // convention because SPIRVRegularizeLLVMBase::addKernelEntryPoint pass
        // could have turned it to spir_func. The "true" entry point is a
        // wrapper kernel function, which can be found further in the module.
        if (FuncDef) {
          if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
            IsEntryPointKernel = true;
            break;
          }
          continue;
        }

        SPIRVValue *SPIRVFunc = SPIRVWriter->getTranslatedValue(&F);
        assert(SPIRVFunc && "All function must be already translated");
        Ops[FunctionIdIdx] = SPIRVFunc->getId();
        FuncDef = SPIRVFunc;
        if (!isNonSemanticDebugInfo())
          break;

        // Most likely unreachable because of Regularise LLVM pass
        if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
          IsEntryPointKernel = true;
          break;
        }
      }
    }
    // For NonSemantic.Shader.DebugInfo we store Function Id index as a
    // separate DebugFunctionDefinition instruction.
    if (isNonSemanticDebugInfo())
      Ops.pop_back();

    if (DISubprogram *FuncDecl = Func->getDeclaration())
      Ops.push_back(transDbgEntry(FuncDecl)->getId());
    else {
      Ops.push_back(getDebugInfoNoneId());
      if (BM->getDebugInfoEIS() == SPIRVEIS_NonSemantic_Shader_DebugInfo_200) {
        // Translate targetFuncName mostly for Fortran trampoline function if it
        // is the case
        StringRef TargetFunc = Func->getTargetFuncName();
        if (!TargetFunc.empty())
          Ops.push_back(BM->getString(TargetFunc.str())->getId());
      }
    }

    DebugFunc = BM->addDebugInfo(SPIRVDebug::Function, getVoidTy(), Ops);
    MDMap.insert(std::make_pair(Func, DebugFunc));
    // Functions local variable might be not refered to anywhere else, except
    // here.
    // Just translate them.
    for (const DINode *Var : Func->getRetainedNodes())
      transDbgEntry(Var);
  }
  // If the function has template parameters the function *is* a template.
  if (DITemplateParameterArray TPA = Func->getTemplateParams()) {
    DebugFunc = transDbgTemplateParams(TPA, DebugFunc);
  }

  if (isNonSemanticDebugInfo() &&
      (Func->isMainSubprogram() || IsEntryPointKernel)) [[maybe_unused]]
    SPIRVEntry *Inst = transDbgEntryPoint(Func, DebugFunc);

  if (isNonSemanticDebugInfo() && FuncDef) [[maybe_unused]]
    SPIRVEntry *Inst = transDbgFuncDefinition(FuncDef, DebugFunc);

  return DebugFunc;
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgFuncDefinition(SPIRVValue *FuncDef,
                                                       SPIRVEntry *DbgFunc) {
  using namespace SPIRVDebug::Operand::FunctionDefinition;
  SPIRVWordVec Ops(OperandCount);
  Ops[FunctionIdx] = DbgFunc->getId();
  Ops[DefinitionIdx] = FuncDef->getId();
  SPIRVFunction *F = static_cast<SPIRVFunction *>(FuncDef);
  SPIRVBasicBlock *BB = F->getNumBasicBlock() ? F->getBasicBlock(0) : nullptr;
  SPIRVId ExtSetId = BM->getExtInstSetId(BM->getDebugInfoEIS());

  return BM->addExtInst(getVoidTy(), ExtSetId, SPIRVDebug::FunctionDefinition,
                        Ops, BB, BB->getInst(0));
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgEntryPoint(const DISubprogram *Func,
                                                   SPIRVEntry *DbgFunc) {
  using namespace SPIRVDebug::Operand::EntryPoint;
  SPIRVWordVec Ops(OperandCount);
  Ops[EntryPointIdx] = DbgFunc->getId();

  DICompileUnit *CU = Func->getUnit();
  if (!CU) {
    Ops[CompilationUnitIdx] = SPIRVCUMap.begin()->second->getId();
    SPIRVWord EmptyStrIdx = BM->getString("")->getId();
    Ops[CompilerSignatureIdx] = EmptyStrIdx;
    Ops[CommandLineArgsIdx] = EmptyStrIdx;
    return BM->addDebugInfo(SPIRVDebug::EntryPoint, getVoidTy(), Ops);
  }

  StringRef Producer = CU->getProducer();
  StringRef Flags = CU->getFlags();
  SPIRVEntry *CUVal = SPIRVCUMap[CU] ? SPIRVCUMap[CU] : getDebugInfoNone();

  Ops[CompilationUnitIdx] = CUVal->getId();
  Ops[CompilerSignatureIdx] = BM->getString(Producer.str())->getId();
  Ops[CommandLineArgsIdx] = BM->getString(Flags.str())->getId();
  return BM->addDebugInfo(SPIRVDebug::EntryPoint, getVoidTy(), Ops);
}

// Location information

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgScope(const DIScope *S) {
  if (const DILexicalBlockFile *LBF = dyn_cast<DILexicalBlockFile>(S)) {
    using namespace SPIRVDebug::Operand::LexicalBlockDiscriminator;
    SPIRVWordVec Ops(OperandCount);
    Ops[SourceIdx] = getSource(S)->getId();
    Ops[DiscriminatorIdx] = LBF->getDiscriminator();
    Ops[ParentIdx] = getScope(S->getScope())->getId();
    if (isNonSemanticDebugInfo())
      transformToConstant(Ops, {DiscriminatorIdx});
    return BM->addDebugInfo(SPIRVDebug::LexicalBlockDiscriminator, getVoidTy(),
                            Ops);
  }
  using namespace SPIRVDebug::Operand::LexicalBlock;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[SourceIdx] = getSource(S)->getId();
  Ops[ParentIdx] = getScope(S->getScope())->getId();
  if (const DILexicalBlock *LB = dyn_cast<DILexicalBlock>(S)) {
    Ops[LineIdx] = LB->getLine();
    Ops[ColumnIdx] = LB->getColumn();
  } else if (const DINamespace *NS = dyn_cast<DINamespace>(S)) {
    Ops[LineIdx] = 0;   // This version of DINamespace has no line number
    Ops[ColumnIdx] = 0; // This version of DINamespace has no column number
    Ops.push_back(BM->getString(NS->getName().str())->getId());
    if (BM->getDebugInfoEIS() == SPIRVEIS_NonSemantic_Shader_DebugInfo_200) {
      SPIRVValue *ExpConst = BM->addConstant(
          SPIRVWriter->transType(Type::getInt1Ty(M->getContext())),
          NS->getExportSymbols() /*Is inlined namespace*/);
      Ops.push_back(ExpConst->getId());
    }
  }
  if (isNonSemanticDebugInfo())
    transformToConstant(Ops, {LineIdx, ColumnIdx});
  return BM->addDebugInfo(SPIRVDebug::LexicalBlock, getVoidTy(), Ops);
}

// Generating DebugScope and DebugNoScope instructions. They can interleave with
// core instructions.
SPIRVEntry *LLVMToSPIRVDbgTran::transDebugLoc(const DebugLoc &Loc,
                                              SPIRVBasicBlock *BB,
                                              SPIRVInstruction *InsertBefore) {
  SPIRVId ExtSetId = BM->getExtInstSetId(BM->getDebugInfoEIS());
  if (!Loc.get())
    return BM->addExtInst(getVoidTy(), ExtSetId, SPIRVDebug::NoScope,
                          std::vector<SPIRVWord>(), BB, InsertBefore);

  using namespace SPIRVDebug::Operand::Scope;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[ScopeIdx] = getScope(static_cast<DIScope *>(Loc.getScope()))->getId();
  if (DILocation *IA = Loc.getInlinedAt())
    Ops.push_back(transDbgEntry(IA)->getId());
  return BM->addExtInst(getVoidTy(), ExtSetId, SPIRVDebug::Scope, Ops, BB,
                        InsertBefore);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgInlinedAt(const DILocation *Loc) {
  // There is a Column operand in NonSemantic.Shader.200 spec
  if (BM->getDebugInfoEIS() == SPIRVEIS_NonSemantic_Shader_DebugInfo_200)
    return transDbgInlinedAtNonSemanticShader200(Loc);
  using namespace SPIRVDebug::Operand::InlinedAt::OpenCL;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[LineIdx] = Loc->getLine();
  Ops[ScopeIdx] = getScope(Loc->getScope())->getId();
  if (DILocation *IA = Loc->getInlinedAt())
    Ops.push_back(transDbgEntry(IA)->getId());
  if (isNonSemanticDebugInfo())
    transformToConstant(Ops, {LineIdx});
  return BM->addDebugInfo(SPIRVDebug::InlinedAt, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgInlinedAtNonSemanticShader200(
    const DILocation *Loc) {
  using namespace SPIRVDebug::Operand::InlinedAt::NonSemantic;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[LineIdx] = Loc->getLine();
  Ops[ColumnIdx] = Loc->getColumn();
  transformToConstant(Ops, {LineIdx, ColumnIdx});
  Ops[ScopeIdx] = getScope(Loc->getScope())->getId();
  if (DILocation *IA = Loc->getInlinedAt())
    Ops.push_back(transDbgEntry(IA)->getId());
  return BM->addDebugInfo(SPIRVDebug::InlinedAt, getVoidTy(), Ops);
}

template <class T>
SPIRVExtInst *LLVMToSPIRVDbgTran::getSource(const T *DIEntry) {
  const std::string FileName = getFullPath(DIEntry);
  auto It = FileMap.find(FileName);
  if (It != FileMap.end())
    return It->second;

  using namespace SPIRVDebug::Operand::Source;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[FileIdx] = BM->getString(FileName)->getId();
  DIFile *F = DIEntry ? DIEntry->getFile() : nullptr;

  if (F && F->getRawChecksum()) {
    auto CheckSum = F->getChecksum().value();

    if (!isNonSemanticDebugInfo())
      Ops.push_back(BM->getString("//__" + CheckSum.getKindAsString().str() +
                                  ":" + CheckSum.Value.str())
                        ->getId());
    else if (BM->getDebugInfoEIS() ==
             SPIRVEIS_NonSemantic_Shader_DebugInfo_200) {
      SPIRVDebug::FileChecksumKind ChecksumKind =
          SPIRV::DbgChecksumKindMap::map(CheckSum.Kind);

      Ops.push_back(
          BM->addIntegerConstant(static_cast<SPIRVTypeInt *>(getInt32Ty()),
                                 ChecksumKind)
              ->getId());
      Ops.push_back(BM->getString(CheckSum.Value.str())->getId());
    }
  }

  if (F && F->getRawSource() && isNonSemanticDebugInfo()) {
    std::string Str = F->getSource().value().str();
    constexpr size_t MaxNumWords =
        MaxWordCount - 2 /*Fixed WC for SPIRVString*/;
    constexpr size_t MaxStrSize = MaxNumWords * 4 - 1;
    const size_t NumWords = getSizeInWords(Str);

    if (BM->getDebugInfoEIS() == SPIRVEIS_NonSemantic_Shader_DebugInfo_200 &&
        Ops.size() == MinOperandCount) {
      Ops.push_back(getDebugInfoNoneId());
      Ops.push_back(getDebugInfoNoneId());
    }
    Ops.push_back(BM->getString(Str.substr(0, MaxStrSize))->getId());
    SPIRVExtInst *Source = static_cast<SPIRVExtInst *>(
        BM->addDebugInfo(SPIRVDebug::Source, getVoidTy(), Ops));
    FileMap[FileName] = Source;
    Str.erase(0, MaxStrSize);

    // No need to generate source continued instructions
    if (NumWords < MaxNumWords)
      return Source;

    uint64_t NumOfContinuedInstructions =
        NumWords / MaxNumWords - 1 + (NumWords % MaxNumWords ? 1 : 0);
    for (uint64_t J = 0; J < NumOfContinuedInstructions; J++) {
      SPIRVWord Op = BM->getString(Str.substr(0, MaxStrSize))->getId();
      BM->addDebugInfo(SPIRVDebug::SourceContinued, getVoidTy(), {Op});
      Str.erase(0, MaxStrSize);
    }
    return Source;
  }

  SPIRVExtInst *Source = static_cast<SPIRVExtInst *>(
      BM->addDebugInfo(SPIRVDebug::Source, getVoidTy(), Ops));
  FileMap[FileName] = Source;
  return Source;
}

void LLVMToSPIRVDbgTran::generateBuildIdentifierAndStoragePath(
    const DICompileUnit *DIEntry) {
  // get information from LLVM IR
  auto BuildIdentifier = DIEntry->getDWOId();
  const std::string BuildIdentifierString = std::to_string(BuildIdentifier);
  const std::string StoragePath = DIEntry->getSplitDebugFilename().str();

  using namespace SPIRVDebug::Operand;

  if (BuildIdentifierInsn || StoragePathInsn) {
#ifndef NDEBUG
    assert(BuildIdentifierInsn && StoragePathInsn &&
           "BuildIdentifier and StoragePath instructions must both be created");

    auto PreviousBuildIdentifierString =
        BM->get<SPIRVString>(
              BuildIdentifierInsn
                  ->getArguments()[BuildIdentifier::IdentifierIdx])
            ->getStr();
    assert(PreviousBuildIdentifierString == BuildIdentifierString &&
           "New BuildIdentifier should match previous BuildIdentifier");
    auto PreviousStoragePath =
        BM->get<SPIRVString>(
              StoragePathInsn->getArguments()[StoragePath::PathIdx])
            ->getStr();
    assert(PreviousStoragePath == StoragePath &&
           "New StoragePath should match previous StoragePath");
#endif
    return;
  }

  // generate BuildIdentifier inst
  SPIRVWordVec BuildIdentifierOps(BuildIdentifier::OperandCount);
  BuildIdentifierOps[BuildIdentifier::IdentifierIdx] =
      BM->getString(BuildIdentifierString)->getId();
  BuildIdentifierOps[BuildIdentifier::FlagsIdx] =
      BM->getLiteralAsConstant(1)->getId(); // Placeholder value for now
  BuildIdentifierInsn = static_cast<SPIRVExtInst *>(BM->addDebugInfo(
      SPIRVDebug::BuildIdentifier, getVoidTy(), BuildIdentifierOps));

  // generate StoragePath inst
  SPIRVWordVec StoragePathOps(StoragePath::OperandCount);
  StoragePathOps[StoragePath::PathIdx] = BM->getString(StoragePath)->getId();
  StoragePathInsn = static_cast<SPIRVExtInst *>(
      BM->addDebugInfo(SPIRVDebug::StoragePath, getVoidTy(), StoragePathOps));
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgFileType(const DIFile *F) {
  return BM->getString(getFullPath(F));
}

// Local variables

SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgLocalVariable(const DILocalVariable *Var) {
  using namespace SPIRVDebug::Operand::LocalVariable;
  SPIRVWordVec Ops(MinOperandCount);

  Ops[NameIdx] = BM->getString(Var->getName().str())->getId();
  Ops[TypeIdx] = transDbgEntry(Var->getType())->getId();
  Ops[SourceIdx] = getSource(Var->getFile())->getId();
  Ops[LineIdx] = Var->getLine();
  Ops[ColumnIdx] = 0; // This version of DILocalVariable has no column number
  Ops[ParentIdx] = getScope(Var->getScope())->getId();
  Ops[FlagsIdx] = transDebugFlags(Var);
  if (SPIRVWord ArgNumber = Var->getArg())
    Ops.push_back(ArgNumber);
  if (isNonSemanticDebugInfo())
    transformToConstant(Ops, {LineIdx, ColumnIdx, FlagsIdx});
  return BM->addDebugInfo(SPIRVDebug::LocalVariable, getVoidTy(), Ops);
}

// DWARF Operations and expressions

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgExpression(const DIExpression *Expr) {
  SPIRVWordVec Operations;
  for (unsigned I = 0, N = Expr->getNumElements(); I < N; ++I) {
    using namespace SPIRVDebug::Operand::Operation;
    auto DWARFOpCode = static_cast<dwarf::LocationAtom>(Expr->getElement(I));

    SPIRVDebug::ExpressionOpCode OC =
        SPIRV::DbgExpressionOpCodeMap::map(DWARFOpCode);
    if (OpCountMap.find(OC) == OpCountMap.end())
      report_fatal_error(llvm::Twine("unknown opcode found in DIExpression"));
    if (OC > SPIRVDebug::Fragment &&
        !(BM->allowExtraDIExpressions() ||
          BM->getDebugInfoEIS() == SPIRVEIS_NonSemantic_Shader_DebugInfo_200))
      report_fatal_error(
          llvm::Twine("unsupported opcode found in DIExpression"));

    unsigned OpCount = OpCountMap[OC];
    SPIRVWordVec Op(OpCount);
    Op[OpCodeIdx] = OC;
    if (isNonSemanticDebugInfo())
      transformToConstant(Op, {OpCodeIdx});
    for (unsigned J = 1; J < OpCount; ++J) {
      Op[J] = Expr->getElement(++I);
      if (isNonSemanticDebugInfo())
        transformToConstant(Op, {J});
    }
    auto *Operation = BM->addDebugInfo(SPIRVDebug::Operation, getVoidTy(), Op);
    Operations.push_back(Operation->getId());
  }
  return BM->addDebugInfo(SPIRVDebug::Expression, getVoidTy(), Operations);
}

// Imported entries (C++ using directive)

SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgImportedEntry(const DIImportedEntity *IE) {
  using namespace SPIRVDebug::Operand::ImportedEntity;
  auto Tag = static_cast<dwarf::Tag>(IE->getTag());
  // FIXME: 'OpenCL/bugged' version is kept because it's hard to remove it
  // It's W/A for missing 2nd index in OpenCL's implementation
  const SPIRVWord OffsetIdx = static_cast<int>(isNonSemanticDebugInfo());
  SPIRVWordVec Ops(OpenCL::OperandCount - OffsetIdx);
  Ops[OpenCL::NameIdx] = BM->getString(IE->getName().str())->getId();
  Ops[OpenCL::TagIdx] = SPIRV::DbgImportedEntityMap::map(Tag);
  Ops[OpenCL::SourceIdx - OffsetIdx] = getSource(IE->getFile())->getId();
  Ops[OpenCL::EntityIdx - OffsetIdx] = transDbgEntry(IE->getEntity())->getId();
  Ops[OpenCL::LineIdx - OffsetIdx] = IE->getLine();
  // This version of DIImportedEntity has no column number
  Ops[OpenCL::ColumnIdx - OffsetIdx] = 0;
  Ops[OpenCL::ParentIdx - OffsetIdx] = getScope(IE->getScope())->getId();
  if (isNonSemanticDebugInfo())
    transformToConstant(Ops, {OpenCL::TagIdx, OpenCL::LineIdx - OffsetIdx,
                              OpenCL::ColumnIdx - OffsetIdx});
  return BM->addDebugInfo(SPIRVDebug::ImportedEntity, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgModule(const DIModule *Module) {
  using namespace SPIRVDebug::Operand::ModuleINTEL;
  SPIRVWordVec Ops(OperandCount);
  Ops[NameIdx] = BM->getString(Module->getName().str())->getId();
  Ops[SourceIdx] = getSource(Module->getFile())->getId();
  Ops[LineIdx] = Module->getLineNo();
  Ops[ParentIdx] = getScope(Module->getScope())->getId();
  Ops[ConfigMacrosIdx] =
      BM->getString(Module->getConfigurationMacros().str())->getId();
  Ops[IncludePathIdx] = BM->getString(Module->getIncludePath().str())->getId();
  Ops[ApiNotesIdx] = BM->getString(Module->getAPINotesFile().str())->getId();
  Ops[IsDeclIdx] = Module->getIsDecl();
  if (BM->getDebugInfoEIS() == SPIRVEIS_NonSemantic_Shader_DebugInfo_200) {
    // The difference in translation of NonSemantic Debug Info and
    // SPV_INTEL_debug_module extension is that extension allows Line and IsDecl
    // operands to be Literals, when the non-OpenCL Debug Info allows only IDs
    // to the constant values.
    transformToConstant(Ops, {LineIdx, IsDeclIdx});
    return BM->addDebugInfo(SPIRVDebug::Module, getVoidTy(), Ops);
  }
  BM->addExtension(ExtensionID::SPV_INTEL_debug_module);
  BM->addCapability(spv::CapabilityDebugInfoModuleINTEL);
  return BM->addDebugInfo(SPIRVDebug::ModuleINTEL, getVoidTy(), Ops);
}
