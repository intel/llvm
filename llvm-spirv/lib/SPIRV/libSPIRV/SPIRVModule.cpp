//===- SPIRVModule.cpp - Class to represent SPIR-V module -------*- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
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
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
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
/// \file
///
/// This file implements Module class for SPIR-V.
///
//===----------------------------------------------------------------------===//

#include "SPIRVModule.h"
#include "SPIRVAsm.h"
#include "SPIRVDebug.h"
#include "SPIRVEntry.h"
#include "SPIRVExtInst.h"
#include "SPIRVFunction.h"
#include "SPIRVInstruction.h"
#include "SPIRVMemAliasingINTEL.h"
#include "SPIRVNameMapEnum.h"
#include "SPIRVStream.h"
#include "SPIRVType.h"
#include "SPIRVValue.h"

#include "llvm/ADT/APInt.h"

#include <set>
#include <unordered_map>
#include <unordered_set>

namespace SPIRV {

SPIRVModule::SPIRVModule()
    : AutoAddCapability(true), ValidateCapability(false), IsValid(true) {}

SPIRVModule::~SPIRVModule() {}

class SPIRVModuleImpl : public SPIRVModule {
public:
  SPIRVModuleImpl()
      : SPIRVModule(), NextId(1),
        SPIRVVersion(static_cast<SPIRVWord>(VersionNumber::SPIRV_1_0)),
        GeneratorId(SPIRVGEN_KhronosLLVMSPIRVTranslator), GeneratorVer(0),
        InstSchema(SPIRVISCH_Default), SrcLang(SourceLanguageOpenCL_C),
        SrcLangVer(102000), BoolTy(nullptr), VoidTy(nullptr) {
    AddrModel = sizeof(size_t) == 32 ? AddressingModelPhysical32
                                     : AddressingModelPhysical64;
    // OpenCL memory model requires Kernel capability
    setMemoryModel(MemoryModelOpenCL);
  }

  SPIRVModuleImpl(const SPIRV::TranslatorOpts &Opts) : SPIRVModuleImpl() {
    TranslationOpts = Opts;
    MaxVersion = Opts.getMaxVersion();
  }

  ~SPIRVModuleImpl() override;

  // Object query functions
  bool exist(SPIRVId) const override;
  bool exist(SPIRVId, SPIRVEntry **) const override;
  SPIRVId getId(SPIRVId Id = SPIRVID_INVALID, unsigned Increment = 1);
  SPIRVEntry *getEntry(SPIRVId Id) const override;
  // If we have at least on OpLine in the module the CurrentLine is non-empty
  bool hasDebugInfo() const override {
    return CurrentLine.get() || !DebugInstVec.empty();
  }

  // Error handling functions
  SPIRVErrorLog &getErrorLog() override { return ErrLog; }
  SPIRVErrorCode getError(std::string &ErrMsg) override {
    return ErrLog.getError(ErrMsg);
  }
  bool checkExtension(ExtensionID Ext, SPIRVErrorCode ErrCode,
                      const std::string &Msg) override {
    if (ErrLog.checkError(isAllowedToUseExtension(Ext), ErrCode, Msg))
      return true;
    setInvalid();
    return false;
  }

  // Module query functions
  SPIRVAddressingModelKind getAddressingModel() override { return AddrModel; }
  SPIRVExtInstSetKind getBuiltinSet(SPIRVId SetId) const override;
  const SPIRVCapMap &getCapability() const override { return CapMap; }
  bool hasCapability(SPIRVCapabilityKind Cap) const override {
    return CapMap.find(Cap) != CapMap.end();
  }
  std::set<std::string> &getExtension() override { return SPIRVExt; }
  SPIRVFunction *getFunction(unsigned I) const override { return FuncVec[I]; }
  SPIRVVariable *getVariable(unsigned I) const override {
    return VariableVec[I];
  }
  SPIRVValue *getValue(SPIRVId TheId) const override;
  std::vector<SPIRVValue *>
  getValues(const std::vector<SPIRVId> &) const override;
  std::vector<SPIRVId> getIds(const std::vector<SPIRVEntry *> &) const override;
  std::vector<SPIRVId> getIds(const std::vector<SPIRVValue *> &) const override;
  SPIRVType *getValueType(SPIRVId TheId) const override;
  std::vector<SPIRVType *>
  getValueTypes(const std::vector<SPIRVId> &) const override;
  SPIRVMemoryModelKind getMemoryModel() const override { return MemoryModel; }
  SPIRVConstant *getLiteralAsConstant(unsigned Literal) override;
  unsigned getNumFunctions() const override { return FuncVec.size(); }
  unsigned getNumVariables() const override { return VariableVec.size(); }
  SourceLanguage getSourceLanguage(SPIRVWord *Ver = nullptr) const override {
    if (Ver)
      *Ver = SrcLangVer;
    return SrcLang;
  }
  std::set<std::string> &getSourceExtension() override { return SrcExtension; }
  bool isEntryPoint(SPIRVExecutionModelKind, SPIRVId EP) const override;
  unsigned short getGeneratorId() const override { return GeneratorId; }
  unsigned short getGeneratorVer() const override { return GeneratorVer; }
  SPIRVWord getSPIRVVersion() const override { return SPIRVVersion; }
  const std::vector<SPIRVExtInst *> &getDebugInstVec() const override {
    return DebugInstVec;
  }
  const std::vector<SPIRVExtInst *> &getAuxDataInstVec() const override {
    return AuxDataInstVec;
  }
  const std::vector<SPIRVString *> &getStringVec() const override {
    return StringVec;
  }
  // Module changing functions
  bool importBuiltinSet(const std::string &, SPIRVId *) override;
  bool importBuiltinSetWithId(const std::string &, SPIRVId) override;
  void setAddressingModel(SPIRVAddressingModelKind AM) override {
    AddrModel = AM;
  }
  void setAlignment(SPIRVValue *, SPIRVWord) override;
  void setMemoryModel(SPIRVMemoryModelKind MM) override {
    MemoryModel = MM;
    if (MemoryModel == spv::MemoryModelOpenCL)
      addCapability(CapabilityKernel);
  }
  void setName(SPIRVEntry *E, const std::string &Name) override;
  void setSourceLanguage(SourceLanguage Lang, SPIRVWord Ver) override {
    SrcLang = Lang;
    SrcLangVer = Ver;
  }
  void setGeneratorId(unsigned short Id) override { GeneratorId = Id; }
  void setGeneratorVer(unsigned short Ver) override { GeneratorVer = Ver; }
  void resolveUnknownStructFields() override;
  void insertEntryNoId(SPIRVEntry *Entry) override { EntryNoId.insert(Entry); }

  void setSPIRVVersion(SPIRVWord Ver) override {
    assert(this->isAllowedToUseVersion(static_cast<VersionNumber>(Ver)));
    SPIRVVersion = Ver;
  }

  // Object creation functions
  template <class T> void addTo(std::vector<T *> &V, SPIRVEntry *E);
  SPIRVEntry *addEntry(SPIRVEntry *E) override;
  SPIRVBasicBlock *addBasicBlock(SPIRVFunction *, SPIRVId) override;
  SPIRVString *getString(const std::string &Str) override;
  SPIRVMemberName *addMemberName(SPIRVTypeStruct *ST, SPIRVWord MemberNumber,
                                 const std::string &Name) override;
  void addUnknownStructField(SPIRVTypeStruct *Struct, unsigned I,
                             SPIRVId ID) override;
  void addLine(SPIRVEntry *E, SPIRVId FileNameId, SPIRVWord Line,
               SPIRVWord Column) override;
  const std::shared_ptr<const SPIRVLine> &getCurrentLine() const override;
  void setCurrentLine(const std::shared_ptr<const SPIRVLine> &Line) override;
  void addDebugLine(SPIRVEntry *E, SPIRVType *TheType, SPIRVId FileNameId,
                    SPIRVWord LineStart, SPIRVWord LineEnd,
                    SPIRVWord ColumnStart, SPIRVWord ColumnEnd) override;
  const std::shared_ptr<const SPIRVExtInst> &
  getCurrentDebugLine() const override;
  void setCurrentDebugLine(
      const std::shared_ptr<const SPIRVExtInst> &DebugLine) override;
  void addCapability(SPIRVCapabilityKind) override;
  void addCapabilityInternal(SPIRVCapabilityKind) override;
  void addExtension(ExtensionID) override;
  const SPIRVDecorateGeneric *addDecorate(SPIRVDecorateGeneric *) override;
  SPIRVDecorationGroup *addDecorationGroup() override;
  SPIRVDecorationGroup *
  addDecorationGroup(SPIRVDecorationGroup *Group) override;
  SPIRVGroupDecorate *
  addGroupDecorate(SPIRVDecorationGroup *Group,
                   const std::vector<SPIRVEntry *> &Targets) override;
  SPIRVGroupDecorateGeneric *
  addGroupDecorateGeneric(SPIRVGroupDecorateGeneric *GDec) override;
  SPIRVGroupMemberDecorate *
  addGroupMemberDecorate(SPIRVDecorationGroup *Group,
                         const std::vector<SPIRVEntry *> &Targets) override;
  void addEntryPoint(SPIRVExecutionModelKind ExecModel, SPIRVId EntryPoint,
                     const std::string &Name,
                     const std::vector<SPIRVId> &Variables) override;
  SPIRVForward *addForward(SPIRVType *Ty) override;
  SPIRVForward *addForward(SPIRVId, SPIRVType *Ty) override;
  SPIRVFunction *addFunction(SPIRVFunction *) override;
  SPIRVFunction *addFunction(SPIRVTypeFunction *, SPIRVId) override;
  SPIRVEntry *replaceForward(SPIRVForward *, SPIRVEntry *) override;
  void eraseInstruction(SPIRVInstruction *, SPIRVBasicBlock *) override;

  // Type creation functions
  template <class T> T *addType(T *Ty);
  SPIRVTypeArray *addArrayType(SPIRVType *, SPIRVConstant *) override;
  SPIRVTypeBool *addBoolType() override;
  SPIRVTypeFloat *addFloatType(unsigned BitWidth) override;
  SPIRVTypeFunction *addFunctionType(SPIRVType *,
                                     const std::vector<SPIRVType *> &) override;
  SPIRVTypeInt *addIntegerType(unsigned BitWidth) override;
  SPIRVTypeOpaque *addOpaqueType(const std::string &) override;
  SPIRVTypePointer *addPointerType(SPIRVStorageClassKind, SPIRVType *) override;
  SPIRVTypeImage *addImageType(SPIRVType *,
                               const SPIRVTypeImageDescriptor &) override;
  SPIRVTypeImage *addImageType(SPIRVType *, const SPIRVTypeImageDescriptor &,
                               SPIRVAccessQualifierKind) override;
  SPIRVTypeSampler *addSamplerType() override;
  SPIRVTypePipeStorage *addPipeStorageType() override;
  SPIRVTypeSampledImage *addSampledImageType(SPIRVTypeImage *T) override;
  SPIRVTypeStruct *openStructType(unsigned, const std::string &) override;
  SPIRVEntry *addTypeStructContinuedINTEL(unsigned NumMembers) override;
  void closeStructType(SPIRVTypeStruct *T, bool) override;
  SPIRVTypeVector *addVectorType(SPIRVType *, SPIRVWord) override;
  SPIRVTypeJointMatrixINTEL *
  addJointMatrixINTELType(SPIRVType *, std::vector<SPIRVValue *>) override;
  SPIRVTypeCooperativeMatrixKHR *
  addCooperativeMatrixKHRType(SPIRVType *, std::vector<SPIRVValue *>) override;
  SPIRVType *addOpaqueGenericType(Op) override;
  SPIRVTypeDeviceEvent *addDeviceEventType() override;
  SPIRVTypeQueue *addQueueType() override;
  SPIRVTypePipe *addPipeType() override;
  SPIRVTypeVoid *addVoidType() override;
  SPIRVType *addSubgroupAvcINTELType(Op) override;
  SPIRVTypeVmeImageINTEL *addVmeImageINTELType(SPIRVTypeImage *T) override;
  SPIRVTypeBufferSurfaceINTEL *
  addBufferSurfaceINTELType(SPIRVAccessQualifierKind Access) override;
  SPIRVTypeTokenINTEL *addTokenTypeINTEL() override;

  // Constant creation functions
  SPIRVInstruction *addBranchInst(SPIRVLabel *, SPIRVBasicBlock *) override;
  SPIRVInstruction *addBranchConditionalInst(SPIRVValue *, SPIRVLabel *,
                                             SPIRVLabel *,
                                             SPIRVBasicBlock *) override;
  SPIRVValue *addCompositeConstant(SPIRVType *,
                                   const std::vector<SPIRVValue *> &) override;
  SPIRVEntry *addCompositeConstantContinuedINTEL(
      const std::vector<SPIRVValue *> &) override;
  SPIRVValue *
  addSpecConstantComposite(SPIRVType *Ty,
                           const std::vector<SPIRVValue *> &Elements) override;
  SPIRVEntry *addSpecConstantCompositeContinuedINTEL(
      const std::vector<SPIRVValue *> &) override;
  SPIRVValue *addConstantFunctionPointerINTEL(SPIRVType *Ty,
                                              SPIRVFunction *F) override;
  SPIRVValue *addConstant(SPIRVValue *) override;
  SPIRVValue *addConstant(SPIRVType *, uint64_t) override;
  SPIRVValue *addConstant(SPIRVType *, llvm::APInt) override;
  SPIRVValue *addSpecConstant(SPIRVType *, uint64_t) override;
  SPIRVValue *addDoubleConstant(SPIRVTypeFloat *, double) override;
  SPIRVValue *addFloatConstant(SPIRVTypeFloat *, float) override;
  SPIRVValue *addIntegerConstant(SPIRVTypeInt *, uint64_t) override;
  SPIRVValue *addNullConstant(SPIRVType *) override;
  SPIRVValue *addUndef(SPIRVType *TheType) override;
  SPIRVValue *addSamplerConstant(SPIRVType *TheType, SPIRVWord AddrMode,
                                 SPIRVWord ParametricMode,
                                 SPIRVWord FilterMode) override;
  SPIRVValue *addPipeStorageConstant(SPIRVType *TheType, SPIRVWord PacketSize,
                                     SPIRVWord PacketAlign,
                                     SPIRVWord Capacity) override;

  // Instruction creation functions
  SPIRVInstruction *addPtrAccessChainInst(SPIRVType *, SPIRVValue *,
                                          std::vector<SPIRVValue *>,
                                          SPIRVBasicBlock *, bool) override;
  SPIRVInstruction *addAsyncGroupCopy(SPIRVValue *Scope, SPIRVValue *Dest,
                                      SPIRVValue *Src, SPIRVValue *NumElems,
                                      SPIRVValue *Stride, SPIRVValue *Event,
                                      SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addExtInst(SPIRVType *, SPIRVWord, SPIRVWord,
                               const std::vector<SPIRVWord> &,
                               SPIRVBasicBlock *,
                               SPIRVInstruction * = nullptr) override;
  SPIRVInstruction *addExtInst(SPIRVType *, SPIRVWord, SPIRVWord,
                               const std::vector<SPIRVValue *> &,
                               SPIRVBasicBlock *,
                               SPIRVInstruction * = nullptr) override;
  SPIRVEntry *createDebugInfo(SPIRVWord, SPIRVType *TheType,
                              const std::vector<SPIRVWord> &) override;
  SPIRVEntry *addDebugInfo(SPIRVWord, SPIRVType *TheType,
                           const std::vector<SPIRVWord> &) override;
  SPIRVEntry *addAuxData(SPIRVWord, SPIRVType *TheType,
                         const std::vector<SPIRVWord> &) override;
  SPIRVEntry *addModuleProcessed(const std::string &) override;
  std::vector<SPIRVModuleProcessed *> getModuleProcessedVec() override;
  SPIRVInstruction *addBinaryInst(Op, SPIRVType *, SPIRVValue *, SPIRVValue *,
                                  SPIRVBasicBlock *) override;
  SPIRVInstruction *addCallInst(SPIRVFunction *, const std::vector<SPIRVWord> &,
                                SPIRVBasicBlock *) override;
  SPIRVInstruction *addIndirectCallInst(SPIRVValue *, SPIRVType *,
                                        const std::vector<SPIRVWord> &,
                                        SPIRVBasicBlock *) override;
  SPIRVEntry *getOrAddAsmTargetINTEL(const std::string &) override;
  SPIRVValue *addAsmINTEL(SPIRVTypeFunction *, SPIRVAsmTargetINTEL *,
                          const std::string &, const std::string &) override;
  SPIRVInstruction *addAsmCallINTELInst(SPIRVAsmINTEL *,
                                        const std::vector<SPIRVWord> &,
                                        SPIRVBasicBlock *) override;
  SPIRVInstruction *addCmpInst(Op, SPIRVType *, SPIRVValue *, SPIRVValue *,
                               SPIRVBasicBlock *) override;
  SPIRVInstruction *addLoadInst(SPIRVValue *, const std::vector<SPIRVWord> &,
                                SPIRVBasicBlock *) override;
  SPIRVInstruction *addPhiInst(SPIRVType *, std::vector<SPIRVValue *>,
                               SPIRVBasicBlock *) override;
  SPIRVInstruction *addCompositeConstructInst(SPIRVType *,
                                              const std::vector<SPIRVId> &,
                                              SPIRVBasicBlock *) override;
  SPIRVInstruction *addCompositeExtractInst(SPIRVType *, SPIRVValue *,
                                            const std::vector<SPIRVWord> &,
                                            SPIRVBasicBlock *) override;
  SPIRVInstruction *
  addCompositeInsertInst(SPIRVValue *Object, SPIRVValue *Composite,
                         const std::vector<SPIRVWord> &Indices,
                         SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addCopyObjectInst(SPIRVType *TheType, SPIRVValue *Operand,
                                      SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addCopyMemoryInst(SPIRVValue *, SPIRVValue *,
                                      const std::vector<SPIRVWord> &,
                                      SPIRVBasicBlock *) override;
  SPIRVInstruction *addCopyMemorySizedInst(SPIRVValue *, SPIRVValue *,
                                           SPIRVValue *,
                                           const std::vector<SPIRVWord> &,
                                           SPIRVBasicBlock *) override;
  SPIRVInstruction *addControlBarrierInst(SPIRVValue *ExecKind,
                                          SPIRVValue *MemKind,
                                          SPIRVValue *MemSema,
                                          SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addGroupInst(Op OpCode, SPIRVType *Type, Scope Scope,
                                 const std::vector<SPIRVValue *> &Ops,
                                 SPIRVBasicBlock *BB) override;
  virtual SPIRVInstruction *
  addInstruction(SPIRVInstruction *Inst, SPIRVBasicBlock *BB,
                 SPIRVInstruction *InsertBefore = nullptr);
  SPIRVInstTemplateBase *addInstTemplate(Op OC, SPIRVBasicBlock *BB,
                                         SPIRVType *Ty) override;
  SPIRVInstTemplateBase *addInstTemplate(Op OC,
                                         const std::vector<SPIRVWord> &Ops,
                                         SPIRVBasicBlock *BB,
                                         SPIRVType *Ty) override;
  void addInstTemplate(SPIRVInstTemplateBase *Ins,
                       const std::vector<SPIRVWord> &Ops, SPIRVBasicBlock *BB,
                       SPIRVType *Ty) override;
  SPIRVInstruction *addLifetimeInst(Op OC, SPIRVValue *Object, SPIRVWord Size,
                                    SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addMemoryBarrierInst(Scope ScopeKind, SPIRVWord MemFlag,
                                         SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addUnreachableInst(SPIRVBasicBlock *) override;
  SPIRVInstruction *addReturnInst(SPIRVBasicBlock *) override;
  SPIRVInstruction *addReturnValueInst(SPIRVValue *,
                                       SPIRVBasicBlock *) override;
  SPIRVInstruction *addSelectInst(SPIRVValue *, SPIRVValue *, SPIRVValue *,
                                  SPIRVBasicBlock *) override;
  SPIRVInstruction *
  addLoopMergeInst(SPIRVId MergeBlock, SPIRVId ContinueTarget,
                   SPIRVWord LoopControl,
                   std::vector<SPIRVWord> LoopControlParameters,
                   SPIRVBasicBlock *BB) override;
  SPIRVInstruction *
  addLoopControlINTELInst(SPIRVWord LoopControl,
                          std::vector<SPIRVWord> LoopControlParameters,
                          SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addFixedPointIntelInst(Op OC, SPIRVType *ResTy,
                                           SPIRVValue *Input,
                                           const std::vector<SPIRVWord> &Ops,
                                           SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addArbFloatPointIntelInst(Op OC, SPIRVType *ResTy,
                                              SPIRVValue *InA, SPIRVValue *InB,
                                              const std::vector<SPIRVWord> &Ops,
                                              SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addSelectionMergeInst(SPIRVId MergeBlock,
                                          SPIRVWord SelectionControl,
                                          SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addStoreInst(SPIRVValue *, SPIRVValue *,
                                 const std::vector<SPIRVWord> &,
                                 SPIRVBasicBlock *) override;
  SPIRVInstruction *addSwitchInst(
      SPIRVValue *, SPIRVBasicBlock *,
      const std::vector<std::pair<std::vector<SPIRVWord>, SPIRVBasicBlock *>> &,
      SPIRVBasicBlock *) override;
  SPIRVInstruction *addVectorTimesScalarInst(SPIRVType *TheType,
                                             SPIRVId TheVector,
                                             SPIRVId TheScalar,
                                             SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addVectorTimesMatrixInst(SPIRVType *TheType,
                                             SPIRVId TheVector,
                                             SPIRVId TheScalar,
                                             SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addMatrixTimesScalarInst(SPIRVType *TheType,
                                             SPIRVId TheMatrix,
                                             SPIRVId TheScalar,
                                             SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addMatrixTimesVectorInst(SPIRVType *TheType,
                                             SPIRVId TheMatrix,
                                             SPIRVId TheVector,
                                             SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addMatrixTimesMatrixInst(SPIRVType *TheType, SPIRVId M1,
                                             SPIRVId M2,
                                             SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addTransposeInst(SPIRVType *TheType, SPIRVId TheMatrix,
                                     SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addUnaryInst(Op, SPIRVType *, SPIRVValue *,
                                 SPIRVBasicBlock *) override;
  SPIRVInstruction *addVariable(SPIRVType *, bool, SPIRVLinkageTypeKind,
                                SPIRVValue *, const std::string &,
                                SPIRVStorageClassKind,
                                SPIRVBasicBlock *) override;
  SPIRVValue *addVectorShuffleInst(SPIRVType *Type, SPIRVValue *Vec1,
                                   SPIRVValue *Vec2,
                                   const std::vector<SPIRVWord> &Components,
                                   SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addVectorExtractDynamicInst(SPIRVValue *, SPIRVValue *,
                                                SPIRVBasicBlock *) override;
  SPIRVInstruction *addVectorInsertDynamicInst(SPIRVValue *, SPIRVValue *,
                                               SPIRVValue *,
                                               SPIRVBasicBlock *) override;
  SPIRVInstruction *addFPGARegINTELInst(SPIRVType *, SPIRVValue *,
                                        SPIRVBasicBlock *) override;
  SPIRVInstruction *addSampledImageInst(SPIRVType *, SPIRVValue *, SPIRVValue *,
                                        SPIRVBasicBlock *) override;
  template <typename AliasingInstType>
  SPIRVEntry *getOrAddMemAliasingINTELInst(std::vector<SPIRVId> Args,
                                           llvm::MDNode *MD);
  SPIRVEntry *getOrAddAliasDomainDeclINTELInst(std::vector<SPIRVId> Args,
                                               llvm::MDNode *MD) override;
  SPIRVEntry *getOrAddAliasScopeDeclINTELInst(std::vector<SPIRVId> Args,
                                              llvm::MDNode *MD) override;
  SPIRVEntry *getOrAddAliasScopeListDeclINTELInst(std::vector<SPIRVId> Args,
                                                  llvm::MDNode *MD) override;
  SPIRVInstruction *addAssumeTrueKHRInst(SPIRVValue *Condition,
                                         SPIRVBasicBlock *BB) override;
  SPIRVInstruction *addExpectKHRInst(SPIRVType *ResultTy, SPIRVValue *Value,
                                     SPIRVValue *ExpectedValue,
                                     SPIRVBasicBlock *BB) override;

  virtual SPIRVId getExtInstSetId(SPIRVExtInstSetKind Kind) const override;

  // I/O functions
  friend spv_ostream &operator<<(spv_ostream &O, SPIRVModule &M);
  friend std::istream &operator>>(std::istream &I, SPIRVModule &M);

private:
  SPIRVErrorLog ErrLog;
  SPIRVId NextId;
  SPIRVWord SPIRVVersion;
  unsigned short GeneratorId;
  unsigned short GeneratorVer;
  SPIRVInstructionSchemaKind InstSchema;
  SourceLanguage SrcLang;
  SPIRVWord SrcLangVer;
  std::set<std::string> SrcExtension;
  std::set<std::string> SPIRVExt;
  SPIRVAddressingModelKind AddrModel;
  SPIRVMemoryModelKind MemoryModel;

  typedef std::map<SPIRVId, SPIRVEntry *> SPIRVIdToEntryMap;
  typedef std::set<SPIRVEntry *> SPIRVEntrySet;
  typedef std::set<SPIRVId> SPIRVIdSet;
  typedef std::vector<SPIRVId> SPIRVIdVec;
  typedef std::vector<SPIRVFunction *> SPIRVFunctionVector;
  typedef std::vector<SPIRVTypeForwardPointer *> SPIRVForwardPointerVec;
  typedef std::vector<SPIRVType *> SPIRVTypeVec;
  typedef std::vector<SPIRVValue *> SPIRVConstantVector;
  typedef std::vector<SPIRVVariable *> SPIRVVariableVec;
  typedef std::vector<SPIRVString *> SPIRVStringVec;
  typedef std::vector<SPIRVMemberName *> SPIRVMemberNameVec;
  typedef std::vector<SPIRVDecorationGroup *> SPIRVDecGroupVec;
  typedef std::vector<SPIRVGroupDecorateGeneric *> SPIRVGroupDecVec;
  typedef std::vector<SPIRVAsmTargetINTEL *> SPIRVAsmTargetVector;
  typedef std::vector<SPIRVAsmINTEL *> SPIRVAsmVector;
  typedef std::vector<SPIRVEntryPoint *> SPIRVEntryPointVec;
  typedef std::map<SPIRVId, SPIRVExtInstSetKind> SPIRVIdToInstructionSetMap;
  std::map<SPIRVExtInstSetKind, SPIRVId> ExtInstSetIds;
  typedef std::map<SPIRVId, SPIRVExtInstSetKind> SPIRVIdToBuiltinSetMap;
  typedef std::map<SPIRVExecutionModelKind, SPIRVIdSet> SPIRVExecModelIdSetMap;
  typedef std::unordered_map<std::string, SPIRVString *> SPIRVStringMap;
  typedef std::map<SPIRVTypeStruct *, std::vector<std::pair<unsigned, SPIRVId>>>
      SPIRVUnknownStructFieldMap;
  typedef std::vector<SPIRVEntry *> SPIRVAliasInstMDVec;
  typedef std::unordered_map<llvm::MDNode *, SPIRVEntry *> SPIRVAliasInstMDMap;

  SPIRVForwardPointerVec ForwardPointerVec;
  SPIRVTypeVec TypeVec;
  SPIRVIdToEntryMap IdEntryMap;
  SPIRVIdToEntryMap IdTypeForwardMap; // Forward declared IDs
  SPIRVFunctionVector FuncVec;
  SPIRVConstantVector ConstVec;
  SPIRVVariableVec VariableVec;
  SPIRVEntrySet EntryNoId; // Entries without id
  SPIRVIdToInstructionSetMap IdToInstSetMap;
  SPIRVIdToBuiltinSetMap IdBuiltinMap;
  SPIRVIdSet NamedId;
  SPIRVStringVec StringVec;
  SPIRVMemberNameVec MemberNameVec;
  std::shared_ptr<const SPIRVLine> CurrentLine;
  std::shared_ptr<const SPIRVExtInst> CurrentDebugLine;
  SPIRVDecorateVec DecorateVec;
  SPIRVDecGroupVec DecGroupVec;
  SPIRVGroupDecVec GroupDecVec;
  SPIRVAsmTargetVector AsmTargetVec;
  SPIRVAsmVector AsmVec;
  SPIRVExecModelIdSetMap EntryPointSet;
  SPIRVEntryPointVec EntryPointVec;
  SPIRVStringMap StrMap;
  SPIRVCapMap CapMap;
  SPIRVUnknownStructFieldMap UnknownStructFieldMap;
  SPIRVTypeBool *BoolTy;
  SPIRVTypeVoid *VoidTy;
  SmallDenseMap<unsigned, SPIRVTypeInt *, 4> IntTypeMap;
  SmallDenseMap<unsigned, SPIRVTypeFloat *, 4> FloatTypeMap;
  std::map<unsigned, SPIRVConstant *> LiteralMap;
  std::vector<SPIRVExtInst *> DebugInstVec;
  std::vector<SPIRVExtInst *> AuxDataInstVec;
  std::vector<SPIRVModuleProcessed *> ModuleProcessedVec;
  SPIRVAliasInstMDVec AliasInstMDVec;
  SPIRVAliasInstMDMap AliasInstMDMap;

  void layoutEntry(SPIRVEntry *Entry);
};

SPIRVModuleImpl::~SPIRVModuleImpl() {
  for (auto *I : EntryNoId)
    delete I;

  for (auto I : IdEntryMap)
    delete I.second;

  for (auto C : CapMap)
    delete C.second;

  for (auto *M : ModuleProcessedVec)
    delete M;
}

const std::shared_ptr<const SPIRVLine> &
SPIRVModuleImpl::getCurrentLine() const {
  return CurrentLine;
}

void SPIRVModuleImpl::setCurrentLine(
    const std::shared_ptr<const SPIRVLine> &Line) {
  CurrentLine = Line;
}

void SPIRVModuleImpl::addLine(SPIRVEntry *E, SPIRVId FileNameId, SPIRVWord Line,
                              SPIRVWord Column) {
  if (!(CurrentLine && CurrentLine->equals(FileNameId, Line, Column)))
    CurrentLine.reset(new SPIRVLine(this, FileNameId, Line, Column));
  assert(E && "invalid entry");
  E->setLine(CurrentLine);
}

const std::shared_ptr<const SPIRVExtInst> &
SPIRVModuleImpl::getCurrentDebugLine() const {
  return CurrentDebugLine;
}

void SPIRVModuleImpl::setCurrentDebugLine(
    const std::shared_ptr<const SPIRVExtInst> &DebugLine) {
  CurrentDebugLine = DebugLine;
}

namespace {
bool isDebugLineEqual(const SPIRVExtInst &CurrentDebugLine, SPIRVId FileNameId,
                      SPIRVId LineStartId, SPIRVId LineEndId,
                      SPIRVId ColumnStartId, SPIRVId ColumnEndId) {
  assert(CurrentDebugLine.getExtOp() == SPIRVDebug::DebugLine);
  const std::vector<SPIRVWord> CurrentDebugLineArgs =
      CurrentDebugLine.getArguments();

  using namespace SPIRVDebug::Operand::DebugLine;
  return CurrentDebugLineArgs[SourceIdx] == FileNameId &&
         CurrentDebugLineArgs[StartIdx] == LineStartId &&
         CurrentDebugLineArgs[EndIdx] == LineEndId &&
         CurrentDebugLineArgs[ColumnStartIdx] == ColumnStartId &&
         CurrentDebugLineArgs[ColumnEndIdx] == ColumnEndId;
}
} // namespace

void SPIRVModuleImpl::addDebugLine(SPIRVEntry *E, SPIRVType *TheType,
                                   SPIRVId FileNameId, SPIRVWord LineStart,
                                   SPIRVWord LineEnd, SPIRVWord ColumnStart,
                                   SPIRVWord ColumnEnd) {
  if (!(CurrentDebugLine &&
        isDebugLineEqual(*CurrentDebugLine, FileNameId,
                         getLiteralAsConstant(LineStart)->getId(),
                         getLiteralAsConstant(LineEnd)->getId(),
                         getLiteralAsConstant(ColumnStart)->getId(),
                         getLiteralAsConstant(ColumnEnd)->getId()))) {
    using namespace SPIRVDebug::Operand::DebugLine;

    std::vector<SPIRVWord> DebugLineOps(OperandCount);
    DebugLineOps[SourceIdx] = FileNameId;
    DebugLineOps[StartIdx] = getLiteralAsConstant(LineStart)->getId();
    DebugLineOps[EndIdx] = getLiteralAsConstant(LineEnd)->getId();
    DebugLineOps[ColumnStartIdx] = getLiteralAsConstant(ColumnStart)->getId();
    DebugLineOps[ColumnEndIdx] = getLiteralAsConstant(ColumnEnd)->getId();

    CurrentDebugLine.reset(static_cast<SPIRVExtInst *>(
        createDebugInfo(SPIRVDebug::DebugLine, TheType, DebugLineOps)));
  }

  assert(E && "invalid entry");
  E->setDebugLine(CurrentDebugLine);
}

SPIRVValue *SPIRVModuleImpl::addSamplerConstant(SPIRVType *TheType,
                                                SPIRVWord AddrMode,
                                                SPIRVWord ParametricMode,
                                                SPIRVWord FilterMode) {
  return addConstant(new SPIRVConstantSampler(this, TheType, getId(), AddrMode,
                                              ParametricMode, FilterMode));
}

SPIRVValue *SPIRVModuleImpl::addPipeStorageConstant(SPIRVType *TheType,
                                                    SPIRVWord PacketSize,
                                                    SPIRVWord PacketAlign,
                                                    SPIRVWord Capacity) {
  return addConstant(new SPIRVConstantPipeStorage(
      this, TheType, getId(), PacketSize, PacketAlign, Capacity));
}

void SPIRVModuleImpl::addExtension(ExtensionID Ext) {
  std::string ExtName;
  SPIRVMap<ExtensionID, std::string>::find(Ext, &ExtName);
  if (!getErrorLog().checkError(isAllowedToUseExtension(Ext),
                                SPIRVEC_RequiresExtension, ExtName)) {
    setInvalid();
    return;
  }
  SPIRVExt.insert(ExtName);

  // SPV_EXT_shader_atomic_float16_add extends the
  // SPV_EXT_shader_atomic_float_add extension.
  // The specification requires both extensions to be added to use
  // AtomicFloat16AddEXT capability whereas getRequiredExtension()
  // is able to return a single extensionID.
  if (Ext == ExtensionID::SPV_EXT_shader_atomic_float16_add) {
    SPIRVMap<ExtensionID, std::string>::find(
        ExtensionID::SPV_EXT_shader_atomic_float_add, &ExtName);
    SPIRVExt.insert(ExtName);
  }
}

void SPIRVModuleImpl::addCapability(SPIRVCapabilityKind Cap) {
  addCapabilities(SPIRV::getCapability(Cap));
  SPIRVDBG(spvdbgs() << "addCapability: " << SPIRVCapabilityNameMap::map(Cap)
                     << '\n');
  if (hasCapability(Cap))
    return;

  auto *CapObj = new SPIRVCapability(this, Cap);
  if (AutoAddExtensions) {
    // While we are reading existing SPIR-V we need to read it as-is and don't
    // add required extensions for each entry automatically
    auto Ext = CapObj->getRequiredExtension();
    if (Ext.has_value())
      addExtension(Ext.value());
  }

  CapMap.insert(std::make_pair(Cap, CapObj));
}

void SPIRVModuleImpl::addCapabilityInternal(SPIRVCapabilityKind Cap) {
  if (AutoAddCapability) {
    if (hasCapability(Cap))
      return;

    CapMap.insert(std::make_pair(Cap, new SPIRVCapability(this, Cap)));
  }
}

SPIRVConstant *SPIRVModuleImpl::getLiteralAsConstant(unsigned Literal) {
  auto Loc = LiteralMap.find(Literal);
  if (Loc != LiteralMap.end())
    return Loc->second;
  auto *Ty = addIntegerType(32);
  auto *V =
      new SPIRVConstant(this, Ty, getId(), static_cast<uint64_t>(Literal));
  LiteralMap[Literal] = V;
  addConstant(V);
  return V;
}

void SPIRVModuleImpl::layoutEntry(SPIRVEntry *E) {
  auto OC = E->getOpCode();
  int IntOC = static_cast<int>(OC);
  switch (IntOC) {
  case OpString:
    addTo(StringVec, E);
    break;
  case OpMemberName:
    addTo(MemberNameVec, E);
    break;
  case OpVariable: {
    auto *BV = static_cast<SPIRVVariable *>(E);
    if (!BV->getParent())
      addTo(VariableVec, E);
  } break;
  case OpExtInst: {
    SPIRVExtInst *EI = static_cast<SPIRVExtInst *>(E);
    if ((EI->getExtSetKind() == SPIRVEIS_Debug ||
         EI->getExtSetKind() == SPIRVEIS_OpenCL_DebugInfo_100 ||
         EI->getExtSetKind() == SPIRVEIS_NonSemantic_Shader_DebugInfo_100 ||
         EI->getExtSetKind() == SPIRVEIS_NonSemantic_Shader_DebugInfo_200) &&
        EI->getExtOp() != SPIRVDebug::Declare &&
        EI->getExtOp() != SPIRVDebug::Value &&
        EI->getExtOp() != SPIRVDebug::Scope &&
        EI->getExtOp() != SPIRVDebug::NoScope) {
      DebugInstVec.push_back(EI);
    }
    if (EI->getExtSetKind() == SPIRVEIS_NonSemantic_AuxData)
      AuxDataInstVec.push_back(EI);
    break;
  }
  case OpAsmTargetINTEL: {
    addTo(AsmTargetVec, E);
    break;
  }
  case OpAliasDomainDeclINTEL:
  case OpAliasScopeDeclINTEL:
  case OpAliasScopeListDeclINTEL: {
    addTo(AliasInstMDVec, E);
    break;
  }
  case OpAsmINTEL: {
    addTo(AsmVec, E);
    break;
  }
  default:
    if (isTypeOpCode(OC))
      TypeVec.push_back(static_cast<SPIRVType *>(E));
    else if (isConstantOpCode(OC))
      ConstVec.push_back(static_cast<SPIRVConstant *>(E));
    break;
  }
}

// Add an entry to the id to entry map.
// Assert if the id is mapped to a different entry.
// Certain entries need to be add to specific collectors to maintain
// logic layout of SPIRV.
SPIRVEntry *SPIRVModuleImpl::addEntry(SPIRVEntry *Entry) {
  assert(Entry && "Invalid entry");
  if (Entry->hasId()) {
    SPIRVId Id = Entry->getId();
    assert(Entry->getId() != SPIRVID_INVALID && "Invalid id");
    SPIRVEntry *Mapped = nullptr;
    if (exist(Id, &Mapped)) {
      if (Mapped->getOpCode() == internal::OpForward) {
        replaceForward(static_cast<SPIRVForward *>(Mapped), Entry);
      } else {
        assert(Mapped == Entry && "Id used twice");
      }
    } else
      IdEntryMap[Id] = Entry;
  } else {
    // Collect entries with no ID to de-allocate them at the end.
    // Entry of OpLine will be deleted by std::shared_ptr automatically.
    if (Entry->getOpCode() != OpLine)
      EntryNoId.insert(Entry);

    // Store the known ID of pointer type that would be declared later.
    if (Entry->getOpCode() == OpTypeForwardPointer)
      IdTypeForwardMap[static_cast<SPIRVTypeForwardPointer *>(Entry)
                           ->getPointerId()] = Entry;
  }

  Entry->setModule(this);

  layoutEntry(Entry);
  if (AutoAddCapability) {
    for (auto &I : Entry->getRequiredCapability()) {
      addCapability(I);
    }
  }
  if (ValidateCapability) {
    assert(none_of(
        Entry->getRequiredCapability().begin(),
        Entry->getRequiredCapability().end(),
        [this](SPIRVCapabilityKind &val) { return !CapMap.count(val); }));
  }
  if (AutoAddExtensions) {
    // While we are reading existing SPIR-V we need to read it as-is and don't
    // add required extensions for each entry automatically
    auto Ext = Entry->getRequiredExtension();
    if (Ext.has_value())
      addExtension(Ext.value());
  }

  return Entry;
}

bool SPIRVModuleImpl::exist(SPIRVId Id) const { return exist(Id, nullptr); }

bool SPIRVModuleImpl::exist(SPIRVId Id, SPIRVEntry **Entry) const {
  assert(Id != SPIRVID_INVALID && "Invalid Id");
  SPIRVIdToEntryMap::const_iterator Loc = IdEntryMap.find(Id);
  if (Loc == IdEntryMap.end())
    return false;
  if (Entry)
    *Entry = Loc->second;
  return true;
}

// If Id is invalid, returns the next available id.
// Otherwise returns the given id and adjust the next available id by increment.
SPIRVId SPIRVModuleImpl::getId(SPIRVId Id, unsigned Increment) {
  if (!isValidId(Id))
    Id = NextId;
  else
    NextId = std::max(Id, NextId);
  NextId += Increment;
  return Id;
}

SPIRVEntry *SPIRVModuleImpl::getEntry(SPIRVId Id) const {
  assert(Id != SPIRVID_INVALID && "Invalid Id");
  SPIRVIdToEntryMap::const_iterator Loc = IdEntryMap.find(Id);
  if (Loc != IdEntryMap.end()) {
    return Loc->second;
  }
  SPIRVIdToEntryMap::const_iterator LocFwd = IdTypeForwardMap.find(Id);
  if (LocFwd != IdTypeForwardMap.end()) {
    return LocFwd->second;
  }
  assert(false && "Id is not in map");
  return nullptr;
}

SPIRVExtInstSetKind SPIRVModuleImpl::getBuiltinSet(SPIRVId SetId) const {
  auto Loc = IdToInstSetMap.find(SetId);
  assert(Loc != IdToInstSetMap.end() && "Invalid builtin set id");
  return Loc->second;
}

bool SPIRVModuleImpl::isEntryPoint(SPIRVExecutionModelKind ExecModel,
                                   SPIRVId EP) const {
  assert(isValid(ExecModel) && "Invalid execution model");
  assert(EP != SPIRVID_INVALID && "Invalid function id");
  auto Loc = EntryPointSet.find(ExecModel);
  if (Loc == EntryPointSet.end())
    return false;
  return Loc->second.count(EP);
}

// Module change functions
bool SPIRVModuleImpl::importBuiltinSet(const std::string &BuiltinSetName,
                                       SPIRVId *BuiltinSetId) {
  SPIRVId TmpBuiltinSetId = getId();
  if (!importBuiltinSetWithId(BuiltinSetName, TmpBuiltinSetId))
    return false;
  if (BuiltinSetId)
    *BuiltinSetId = TmpBuiltinSetId;
  return true;
}

bool SPIRVModuleImpl::importBuiltinSetWithId(const std::string &BuiltinSetName,
                                             SPIRVId BuiltinSetId) {
  SPIRVExtInstSetKind BuiltinSet = SPIRVEIS_Count;
  SPIRVCKRT(SPIRVBuiltinSetNameMap::rfind(BuiltinSetName, &BuiltinSet),
            InvalidBuiltinSetName, "Actual is " + BuiltinSetName);
  IdToInstSetMap[BuiltinSetId] = BuiltinSet;
  ExtInstSetIds[BuiltinSet] = BuiltinSetId;
  return true;
}

void SPIRVModuleImpl::setAlignment(SPIRVValue *V, SPIRVWord A) {
  V->setAlignment(A);
}

void SPIRVModuleImpl::setName(SPIRVEntry *E, const std::string &Name) {
  E->setName(Name);
  if (!E->hasId())
    return;
  if (!Name.empty())
    NamedId.insert(E->getId());
  else
    NamedId.erase(E->getId());
}

void SPIRVModuleImpl::resolveUnknownStructFields() {
  for (auto &KV : UnknownStructFieldMap) {
    auto *Struct = KV.first;
    for (auto &Indices : KV.second) {
      unsigned I = Indices.first;
      SPIRVId ID = Indices.second;

      auto *Ty = static_cast<SPIRVType *>(getEntry(ID));
      Struct->setMemberType(I, Ty);
    }
  }
}

// Type creation functions
template <class T> T *SPIRVModuleImpl::addType(T *Ty) {
  add(Ty);
  if (!Ty->getName().empty())
    setName(Ty, Ty->getName());
  return Ty;
}

SPIRVTypeVoid *SPIRVModuleImpl::addVoidType() {
  SPIRVTypeVoid *V = VoidTy;
  if (!V) {
    V = addType(new SPIRVTypeVoid(this, getId()));
    VoidTy = V;
  }
  return V;
}

SPIRVTypeArray *SPIRVModuleImpl::addArrayType(SPIRVType *ElementType,
                                              SPIRVConstant *Length) {
  return addType(new SPIRVTypeArray(this, getId(), ElementType, Length));
}

SPIRVTypeBool *SPIRVModuleImpl::addBoolType() {
  SPIRVTypeBool *B = BoolTy;
  if (!B) {
    B = addType(new SPIRVTypeBool(this, getId()));
    BoolTy = B;
  }
  return B;
}

SPIRVTypeInt *SPIRVModuleImpl::addIntegerType(unsigned BitWidth) {
  auto Loc = IntTypeMap.find(BitWidth);
  if (Loc != IntTypeMap.end())
    return Loc->second;
  auto *Ty = new SPIRVTypeInt(this, getId(), BitWidth, false);
  IntTypeMap[BitWidth] = Ty;
  return addType(Ty);
}

SPIRVTypeFloat *SPIRVModuleImpl::addFloatType(unsigned BitWidth) {
  auto Loc = FloatTypeMap.find(BitWidth);
  if (Loc != FloatTypeMap.end())
    return Loc->second;
  auto *Ty = new SPIRVTypeFloat(this, getId(), BitWidth);
  FloatTypeMap[BitWidth] = Ty;
  return addType(Ty);
}

SPIRVTypePointer *
SPIRVModuleImpl::addPointerType(SPIRVStorageClassKind StorageClass,
                                SPIRVType *ElementType) {
  return addType(
      new SPIRVTypePointer(this, getId(), StorageClass, ElementType));
}

SPIRVTypeFunction *SPIRVModuleImpl::addFunctionType(
    SPIRVType *ReturnType, const std::vector<SPIRVType *> &ParameterTypes) {
  return addType(
      new SPIRVTypeFunction(this, getId(), ReturnType, ParameterTypes));
}

SPIRVTypeOpaque *SPIRVModuleImpl::addOpaqueType(const std::string &Name) {
  return addType(new SPIRVTypeOpaque(this, getId(), Name));
}

SPIRVTypeStruct *SPIRVModuleImpl::openStructType(unsigned NumMembers,
                                                 const std::string &Name) {
  auto *T = new SPIRVTypeStruct(this, getId(), NumMembers, Name);
  return T;
}

SPIRVEntry *SPIRVModuleImpl::addTypeStructContinuedINTEL(unsigned NumMembers) {
  return add(new SPIRVTypeStructContinuedINTEL(this, NumMembers));
}

void SPIRVModuleImpl::closeStructType(SPIRVTypeStruct *T, bool Packed) {
  addType(T);
  T->setPacked(Packed);
}

SPIRVTypeVector *SPIRVModuleImpl::addVectorType(SPIRVType *CompType,
                                                SPIRVWord CompCount) {
  return addType(new SPIRVTypeVector(this, getId(), CompType, CompCount));
}

SPIRVTypeJointMatrixINTEL *
SPIRVModuleImpl::addJointMatrixINTELType(SPIRVType *CompType,
                                         std::vector<SPIRVValue *> Args) {
  return addType(new SPIRVTypeJointMatrixINTEL(this, getId(), CompType, Args));
}

SPIRVTypeCooperativeMatrixKHR *
SPIRVModuleImpl::addCooperativeMatrixKHRType(SPIRVType *CompType,
                                             std::vector<SPIRVValue *> Args) {
  return addType(
      new SPIRVTypeCooperativeMatrixKHR(this, getId(), CompType, Args));
}

SPIRVType *SPIRVModuleImpl::addOpaqueGenericType(Op TheOpCode) {
  return addType(new SPIRVTypeOpaqueGeneric(TheOpCode, this, getId()));
}

SPIRVTypeDeviceEvent *SPIRVModuleImpl::addDeviceEventType() {
  return addType(new SPIRVTypeDeviceEvent(this, getId()));
}

SPIRVTypeQueue *SPIRVModuleImpl::addQueueType() {
  return addType(new SPIRVTypeQueue(this, getId()));
}

SPIRVTypePipe *SPIRVModuleImpl::addPipeType() {
  return addType(new SPIRVTypePipe(this, getId()));
}

SPIRVTypeImage *
SPIRVModuleImpl::addImageType(SPIRVType *SampledType,
                              const SPIRVTypeImageDescriptor &Desc) {
  return addType(new SPIRVTypeImage(
      this, getId(), SampledType ? SampledType->getId() : 0, Desc));
}

SPIRVTypeImage *
SPIRVModuleImpl::addImageType(SPIRVType *SampledType,
                              const SPIRVTypeImageDescriptor &Desc,
                              SPIRVAccessQualifierKind Acc) {
  return addType(new SPIRVTypeImage(
      this, getId(), SampledType ? SampledType->getId() : 0, Desc, Acc));
}

SPIRVTypeSampler *SPIRVModuleImpl::addSamplerType() {
  return addType(new SPIRVTypeSampler(this, getId()));
}

SPIRVTypePipeStorage *SPIRVModuleImpl::addPipeStorageType() {
  return addType(new SPIRVTypePipeStorage(this, getId()));
}

SPIRVTypeSampledImage *SPIRVModuleImpl::addSampledImageType(SPIRVTypeImage *T) {
  return addType(new SPIRVTypeSampledImage(this, getId(), T));
}

SPIRVTypeVmeImageINTEL *
SPIRVModuleImpl::addVmeImageINTELType(SPIRVTypeImage *T) {
  return addType(new SPIRVTypeVmeImageINTEL(this, getId(), T));
}

SPIRVTypeBufferSurfaceINTEL *
SPIRVModuleImpl::addBufferSurfaceINTELType(SPIRVAccessQualifierKind Access) {
  return addType(new SPIRVTypeBufferSurfaceINTEL(this, getId(), Access));
}

SPIRVType *SPIRVModuleImpl::addSubgroupAvcINTELType(Op TheOpCode) {
  return addType(new SPIRVTypeSubgroupAvcINTEL(TheOpCode, this, getId()));
}

SPIRVTypeTokenINTEL *SPIRVModuleImpl::addTokenTypeINTEL() {
  return addType(new SPIRVTypeTokenINTEL(this, getId()));
}

SPIRVFunction *SPIRVModuleImpl::addFunction(SPIRVFunction *Func) {
  FuncVec.push_back(add(Func));
  return Func;
}

SPIRVFunction *SPIRVModuleImpl::addFunction(SPIRVTypeFunction *FuncType,
                                            SPIRVId Id) {
  return addFunction(new SPIRVFunction(
      this, FuncType, getId(Id, FuncType->getNumParameters() + 1)));
}

SPIRVBasicBlock *SPIRVModuleImpl::addBasicBlock(SPIRVFunction *Func,
                                                SPIRVId Id) {
  return Func->addBasicBlock(new SPIRVBasicBlock(getId(Id), Func));
}

const SPIRVDecorateGeneric *
SPIRVModuleImpl::addDecorate(SPIRVDecorateGeneric *Dec) {
  add(Dec);
  SPIRVId Id = Dec->getTargetId();
  bool Found = exist(Id);
  (void)Found;
  assert(Found && "Decorate target does not exist");
  if (!Dec->getOwner())
    DecorateVec.push_back(Dec);
  addCapabilities(Dec->getRequiredCapability());
  return Dec;
}

void SPIRVModuleImpl::addEntryPoint(SPIRVExecutionModelKind ExecModel,
                                    SPIRVId EntryPoint, const std::string &Name,
                                    const std::vector<SPIRVId> &Variables) {
  assert(isValid(ExecModel) && "Invalid execution model");
  assert(EntryPoint != SPIRVID_INVALID && "Invalid entry point");
  auto *EP =
      add(new SPIRVEntryPoint(this, ExecModel, EntryPoint, Name, Variables));
  EntryPointVec.push_back(EP);
  EntryPointSet[ExecModel].insert(EntryPoint);
  addCapabilities(SPIRV::getCapability(ExecModel));
}

SPIRVForward *SPIRVModuleImpl::addForward(SPIRVType *Ty) {
  return add(new SPIRVForward(this, Ty, getId()));
}

SPIRVForward *SPIRVModuleImpl::addForward(SPIRVId Id, SPIRVType *Ty) {
  return add(new SPIRVForward(this, Ty, Id));
}

SPIRVEntry *SPIRVModuleImpl::replaceForward(SPIRVForward *Forward,
                                            SPIRVEntry *Entry) {
  SPIRVId Id = Entry->getId();
  SPIRVId ForwardId = Forward->getId();
  if (ForwardId == Id)
    IdEntryMap[Id] = Entry;
  else {
    auto Loc = IdEntryMap.find(Id);
    assert(Loc != IdEntryMap.end());
    IdEntryMap.erase(Loc);
    Entry->setId(ForwardId);
    IdEntryMap[ForwardId] = Entry;
  }
  // Annotations include name, decorations, execution modes
  Entry->takeAnnotations(Forward);
  delete Forward;
  return Entry;
}

void SPIRVModuleImpl::eraseInstruction(SPIRVInstruction *I,
                                       SPIRVBasicBlock *BB) {
  SPIRVId Id = I->getId();
  BB->eraseInstruction(I);
  auto Loc = IdEntryMap.find(Id);
  assert(Loc != IdEntryMap.end());
  IdEntryMap.erase(Loc);
  delete I;
}

SPIRVValue *SPIRVModuleImpl::addConstant(SPIRVValue *C) { return add(C); }

SPIRVValue *SPIRVModuleImpl::addConstant(SPIRVType *Ty, uint64_t V) {
  if (Ty->isTypeBool()) {
    if (V)
      return addConstant(new SPIRVConstantTrue(this, Ty, getId()));
    else
      return addConstant(new SPIRVConstantFalse(this, Ty, getId()));
  }
  if (Ty->isTypeInt())
    return addIntegerConstant(static_cast<SPIRVTypeInt *>(Ty), V);
  return addConstant(new SPIRVConstant(this, Ty, getId(), V));
}

SPIRVValue *SPIRVModuleImpl::addConstant(SPIRVType *Ty, llvm::APInt V) {
  return addConstant(new SPIRVConstant(this, Ty, getId(), V));
}

SPIRVValue *SPIRVModuleImpl::addIntegerConstant(SPIRVTypeInt *Ty, uint64_t V) {
  if (Ty->getBitWidth() == 32) {
    unsigned I32 = static_cast<unsigned>(V);
    assert(I32 == V && "Integer value truncated");
    return getLiteralAsConstant(I32);
  }
  return addConstant(new SPIRVConstant(this, Ty, getId(), V));
}

SPIRVValue *SPIRVModuleImpl::addFloatConstant(SPIRVTypeFloat *Ty, float V) {
  return addConstant(new SPIRVConstant(this, Ty, getId(), V));
}

SPIRVValue *SPIRVModuleImpl::addDoubleConstant(SPIRVTypeFloat *Ty, double V) {
  return addConstant(new SPIRVConstant(this, Ty, getId(), V));
}

SPIRVValue *SPIRVModuleImpl::addNullConstant(SPIRVType *Ty) {
  return addConstant(new SPIRVConstantNull(this, Ty, getId()));
}

SPIRVValue *SPIRVModuleImpl::addCompositeConstant(
    SPIRVType *Ty, const std::vector<SPIRVValue *> &Elements) {
  constexpr int MaxNumElements = MaxWordCount - SPIRVConstantComposite::FixedWC;
  const int NumElements = Elements.size();

  // In case number of elements is greater than maximum WordCount and
  // SPV_INTEL_long_constant_composite is not enabled, the error will be emitted
  // by validate functionality of SPIRVCompositeConstant class.
  if (NumElements <= MaxNumElements ||
      !isAllowedToUseExtension(ExtensionID::SPV_INTEL_long_constant_composite))
    return addConstant(new SPIRVConstantComposite(this, Ty, getId(), Elements));

  auto Start = Elements.begin();
  auto End = Start + MaxNumElements;
  std::vector<SPIRVValue *> Slice(Start, End);
  auto *Res =
      static_cast<SPIRVConstantComposite *>(addCompositeConstant(Ty, Slice));
  for (; End != Elements.end();) {
    Start = End;
    End = ((Elements.end() - End) > MaxNumElements) ? End + MaxNumElements
                                                    : Elements.end();
    Slice.assign(Start, End);
    auto *Continued = static_cast<SPIRVConstantComposite::ContinuedInstType>(
        addCompositeConstantContinuedINTEL(Slice));
    Res->addContinuedInstruction(Continued);
  }
  return Res;
}

SPIRVEntry *SPIRVModuleImpl::addCompositeConstantContinuedINTEL(
    const std::vector<SPIRVValue *> &Elements) {
  return add(new SPIRVConstantCompositeContinuedINTEL(this, Elements));
}

SPIRVValue *SPIRVModuleImpl::addSpecConstantComposite(
    SPIRVType *Ty, const std::vector<SPIRVValue *> &Elements) {
  constexpr int MaxNumElements =
      MaxWordCount - SPIRVSpecConstantComposite::FixedWC;
  const int NumElements = Elements.size();

  // In case number of elements is greater than maximum WordCount and
  // SPV_INTEL_long_constant_composite is not enabled, the error will be emitted
  // by validate functionality of SPIRVSpecConstantComposite class.
  if (NumElements <= MaxNumElements ||
      !isAllowedToUseExtension(ExtensionID::SPV_INTEL_long_constant_composite))
    return addConstant(
        new SPIRVSpecConstantComposite(this, Ty, getId(), Elements));

  auto Start = Elements.begin();
  auto End = Start + MaxNumElements;
  std::vector<SPIRVValue *> Slice(Start, End);
  auto *Res = static_cast<SPIRVSpecConstantComposite *>(
      addSpecConstantComposite(Ty, Slice));
  for (; End != Elements.end();) {
    Start = End;
    End = ((Elements.end() - End) > MaxNumElements) ? End + MaxNumElements
                                                    : Elements.end();
    Slice.assign(Start, End);
    auto *Continued =
        static_cast<SPIRVSpecConstantComposite::ContinuedInstType>(
            addSpecConstantCompositeContinuedINTEL(Slice));
    Res->addContinuedInstruction(Continued);
  }
  return Res;
}

SPIRVEntry *SPIRVModuleImpl::addSpecConstantCompositeContinuedINTEL(
    const std::vector<SPIRVValue *> &Elements) {
  return add(new SPIRVSpecConstantCompositeContinuedINTEL(this, Elements));
}

SPIRVValue *SPIRVModuleImpl::addConstantFunctionPointerINTEL(SPIRVType *Ty,
                                                             SPIRVFunction *F) {
  return addConstant(
      new SPIRVConstantFunctionPointerINTEL(getId(), Ty, F, this));
}

SPIRVValue *SPIRVModuleImpl::addUndef(SPIRVType *TheType) {
  return addConstant(new SPIRVUndef(this, TheType, getId()));
}

SPIRVValue *SPIRVModuleImpl::addSpecConstant(SPIRVType *Ty, uint64_t V) {
  if (Ty->isTypeBool()) {
    if (V)
      return add(new SPIRVSpecConstantTrue(this, Ty, getId()));
    else
      return add(new SPIRVSpecConstantFalse(this, Ty, getId()));
  }
  return add(new SPIRVSpecConstant(this, Ty, getId(), V));
}

// Instruction creation functions

SPIRVInstruction *
SPIRVModuleImpl::addStoreInst(SPIRVValue *Target, SPIRVValue *Source,
                              const std::vector<SPIRVWord> &TheMemoryAccess,
                              SPIRVBasicBlock *BB) {
  return BB->addInstruction(
      new SPIRVStore(Target->getId(), Source->getId(), TheMemoryAccess, BB));
}

SPIRVInstruction *SPIRVModuleImpl::addSwitchInst(
    SPIRVValue *Select, SPIRVBasicBlock *Default,
    const std::vector<std::pair<std::vector<SPIRVWord>, SPIRVBasicBlock *>>
        &Pairs,
    SPIRVBasicBlock *BB) {
  return BB->addInstruction(new SPIRVSwitch(Select, Default, Pairs, BB));
}

SPIRVInstruction *
SPIRVModuleImpl::addVectorTimesScalarInst(SPIRVType *TheType, SPIRVId TheVector,
                                          SPIRVId TheScalar,
                                          SPIRVBasicBlock *BB) {
  return BB->addInstruction(
      new SPIRVVectorTimesScalar(TheType, getId(), TheVector, TheScalar, BB));
}

SPIRVInstruction *
SPIRVModuleImpl::addVectorTimesMatrixInst(SPIRVType *TheType, SPIRVId TheVector,
                                          SPIRVId TheMatrix,
                                          SPIRVBasicBlock *BB) {
  return BB->addInstruction(
      new SPIRVVectorTimesMatrix(TheType, getId(), TheVector, TheMatrix, BB));
}

SPIRVInstruction *
SPIRVModuleImpl::addMatrixTimesScalarInst(SPIRVType *TheType, SPIRVId TheMatrix,
                                          SPIRVId TheScalar,
                                          SPIRVBasicBlock *BB) {
  return BB->addInstruction(
      new SPIRVMatrixTimesScalar(TheType, getId(), TheMatrix, TheScalar, BB));
}

SPIRVInstruction *
SPIRVModuleImpl::addMatrixTimesVectorInst(SPIRVType *TheType, SPIRVId TheMatrix,
                                          SPIRVId TheVector,
                                          SPIRVBasicBlock *BB) {
  return BB->addInstruction(
      new SPIRVMatrixTimesVector(TheType, getId(), TheMatrix, TheVector, BB));
}

SPIRVInstruction *
SPIRVModuleImpl::addMatrixTimesMatrixInst(SPIRVType *TheType, SPIRVId M1,
                                          SPIRVId M2, SPIRVBasicBlock *BB) {
  return BB->addInstruction(
      new SPIRVMatrixTimesMatrix(TheType, getId(), M1, M2, BB));
}

SPIRVInstruction *SPIRVModuleImpl::addTransposeInst(SPIRVType *TheType,
                                                    SPIRVId TheMatrix,
                                                    SPIRVBasicBlock *BB) {
  return BB->addInstruction(
      new SPIRVTranspose(TheType, getId(), TheMatrix, BB));
}

SPIRVInstruction *
SPIRVModuleImpl::addGroupInst(Op OpCode, SPIRVType *Type, Scope Scope,
                              const std::vector<SPIRVValue *> &Ops,
                              SPIRVBasicBlock *BB) {
  assert(!Type || !Type->isTypeVoid());
  auto WordOps = getIds(Ops);
  WordOps.insert(WordOps.begin(), Scope);
  return addInstTemplate(OpCode, WordOps, BB, Type);
}

SPIRVInstruction *
SPIRVModuleImpl::addInstruction(SPIRVInstruction *Inst, SPIRVBasicBlock *BB,
                                SPIRVInstruction *InsertBefore) {
  if (BB)
    return BB->addInstruction(Inst, InsertBefore);
  if (Inst->getOpCode() != OpSpecConstantOp) {
    SPIRVInstruction *Res = createSpecConstantOpInst(Inst);
    delete Inst;
    Inst = Res;
  }
  return static_cast<SPIRVInstruction *>(addConstant(Inst));
}

SPIRVInstruction *
SPIRVModuleImpl::addLoadInst(SPIRVValue *Source,
                             const std::vector<SPIRVWord> &TheMemoryAccess,
                             SPIRVBasicBlock *BB) {
  return addInstruction(
      new SPIRVLoad(getId(), Source->getId(), TheMemoryAccess, BB), BB);
}

SPIRVInstruction *
SPIRVModuleImpl::addPhiInst(SPIRVType *Type,
                            std::vector<SPIRVValue *> IncomingPairs,
                            SPIRVBasicBlock *BB) {
  return addInstruction(new SPIRVPhi(Type, getId(), IncomingPairs, BB), BB);
}

SPIRVInstruction *SPIRVModuleImpl::addExtInst(
    SPIRVType *TheType, SPIRVWord BuiltinSet, SPIRVWord EntryPoint,
    const std::vector<SPIRVWord> &Args, SPIRVBasicBlock *BB,
    SPIRVInstruction *InsertBefore) {
  return addInstruction(
      new SPIRVExtInst(TheType, getId(), BuiltinSet, EntryPoint, Args, BB), BB,
      InsertBefore);
}

SPIRVInstruction *SPIRVModuleImpl::addExtInst(
    SPIRVType *TheType, SPIRVWord BuiltinSet, SPIRVWord EntryPoint,
    const std::vector<SPIRVValue *> &Args, SPIRVBasicBlock *BB,
    SPIRVInstruction *InsertBefore) {
  return addInstruction(
      new SPIRVExtInst(TheType, getId(), BuiltinSet, EntryPoint, Args, BB), BB,
      InsertBefore);
}

SPIRVEntry *
SPIRVModuleImpl::createDebugInfo(SPIRVWord InstId, SPIRVType *TheType,
                                 const std::vector<SPIRVWord> &Args) {
  return new SPIRVExtInst(this, getId(), TheType, SPIRVEIS_OpenCL_DebugInfo_100,
                          ExtInstSetIds[getDebugInfoEIS()], InstId, Args);
}

SPIRVEntry *SPIRVModuleImpl::addDebugInfo(SPIRVWord InstId, SPIRVType *TheType,
                                          const std::vector<SPIRVWord> &Args) {
  return addEntry(createDebugInfo(InstId, TheType, Args));
}

SPIRVEntry *SPIRVModuleImpl::addAuxData(SPIRVWord InstId, SPIRVType *TheType,
                                        const std::vector<SPIRVWord> &Args) {
  return addEntry(new SPIRVExtInst(
      this, getId(), TheType, SPIRVEIS_NonSemantic_AuxData,
      getExtInstSetId(SPIRVEIS_NonSemantic_AuxData), InstId, Args));
}

SPIRVEntry *SPIRVModuleImpl::addModuleProcessed(const std::string &Process) {
  ModuleProcessedVec.push_back(new SPIRVModuleProcessed(this, Process));
  return ModuleProcessedVec.back();
}

std::vector<SPIRVModuleProcessed *> SPIRVModuleImpl::getModuleProcessedVec() {
  return ModuleProcessedVec;
}

SPIRVInstruction *
SPIRVModuleImpl::addCallInst(SPIRVFunction *TheFunction,
                             const std::vector<SPIRVWord> &TheArguments,
                             SPIRVBasicBlock *BB) {
  return addInstruction(
      new SPIRVFunctionCall(getId(), TheFunction, TheArguments, BB), BB);
}

SPIRVInstruction *SPIRVModuleImpl::addIndirectCallInst(
    SPIRVValue *TheCalledValue, SPIRVType *TheReturnType,
    const std::vector<SPIRVWord> &TheArguments, SPIRVBasicBlock *BB) {
  return addInstruction(
      new SPIRVFunctionPointerCallINTEL(getId(), TheCalledValue, TheReturnType,
                                        TheArguments, BB),
      BB);
}

SPIRVEntry *
SPIRVModuleImpl::getOrAddAsmTargetINTEL(const std::string &TheTarget) {
  auto TargetIt = std::find_if(AsmTargetVec.begin(), AsmTargetVec.end(),
                               [&TheTarget](const SPIRVAsmTargetINTEL *Target) {
                                 return Target->getTarget() == TheTarget;
                               });
  if (TargetIt == AsmTargetVec.end())
    return add(new SPIRVAsmTargetINTEL(this, getId(), TheTarget));
  return *TargetIt;
}

SPIRVValue *SPIRVModuleImpl::addAsmINTEL(SPIRVTypeFunction *TheType,
                                         SPIRVAsmTargetINTEL *TheTarget,
                                         const std::string &TheInstructions,
                                         const std::string &TheConstraints) {
  auto *Asm = new SPIRVAsmINTEL(this, TheType, getId(), TheTarget,
                                TheInstructions, TheConstraints);
  return add(Asm);
}

SPIRVInstruction *
SPIRVModuleImpl::addAsmCallINTELInst(SPIRVAsmINTEL *TheAsm,
                                     const std::vector<SPIRVWord> &TheArguments,
                                     SPIRVBasicBlock *BB) {
  return addInstruction(
      new SPIRVAsmCallINTEL(getId(), TheAsm, TheArguments, BB), BB);
}

SPIRVInstruction *SPIRVModuleImpl::addBinaryInst(Op TheOpCode, SPIRVType *Type,
                                                 SPIRVValue *Op1,
                                                 SPIRVValue *Op2,
                                                 SPIRVBasicBlock *BB) {
  return addInstruction(SPIRVInstTemplateBase::create(
                            TheOpCode, Type, getId(),
                            getVec(Op1->getId(), Op2->getId()), BB, this),
                        BB);
}

SPIRVInstruction *SPIRVModuleImpl::addUnreachableInst(SPIRVBasicBlock *BB) {
  return addInstruction(new SPIRVUnreachable(BB), BB);
}

SPIRVInstruction *SPIRVModuleImpl::addReturnInst(SPIRVBasicBlock *BB) {
  return addInstruction(new SPIRVReturn(BB), BB);
}

SPIRVInstruction *SPIRVModuleImpl::addReturnValueInst(SPIRVValue *ReturnValue,
                                                      SPIRVBasicBlock *BB) {
  return addInstruction(new SPIRVReturnValue(ReturnValue, BB), BB);
}

SPIRVInstruction *SPIRVModuleImpl::addUnaryInst(Op TheOpCode,
                                                SPIRVType *TheType,
                                                SPIRVValue *Op,
                                                SPIRVBasicBlock *BB) {
  return addInstruction(
      SPIRVInstTemplateBase::create(TheOpCode, TheType, getId(),
                                    getVec(Op->getId()), BB, this),
      BB);
}

SPIRVInstruction *SPIRVModuleImpl::addVectorExtractDynamicInst(
    SPIRVValue *TheVector, SPIRVValue *Index, SPIRVBasicBlock *BB) {
  return addInstruction(
      new SPIRVVectorExtractDynamic(getId(), TheVector, Index, BB), BB);
}

SPIRVInstruction *SPIRVModuleImpl::addVectorInsertDynamicInst(
    SPIRVValue *TheVector, SPIRVValue *TheComponent, SPIRVValue *Index,
    SPIRVBasicBlock *BB) {
  return addInstruction(
      new SPIRVVectorInsertDynamic(getId(), TheVector, TheComponent, Index, BB),
      BB);
}

SPIRVValue *SPIRVModuleImpl::addVectorShuffleInst(
    SPIRVType *Type, SPIRVValue *Vec1, SPIRVValue *Vec2,
    const std::vector<SPIRVWord> &Components, SPIRVBasicBlock *BB) {
  std::vector<SPIRVId> Ops{Vec1->getId(), Vec2->getId()};
  Ops.insert(Ops.end(), Components.begin(), Components.end());

  return addInstruction(SPIRVInstTemplateBase::create(OpVectorShuffle, Type,
                                                      getId(), Ops, BB, this),
                        BB);
}

SPIRVInstruction *SPIRVModuleImpl::addBranchInst(SPIRVLabel *TargetLabel,
                                                 SPIRVBasicBlock *BB) {
  return addInstruction(new SPIRVBranch(TargetLabel, BB), BB);
}

SPIRVInstruction *SPIRVModuleImpl::addBranchConditionalInst(
    SPIRVValue *Condition, SPIRVLabel *TrueLabel, SPIRVLabel *FalseLabel,
    SPIRVBasicBlock *BB) {
  return addInstruction(
      new SPIRVBranchConditional(Condition, TrueLabel, FalseLabel, BB), BB);
}

SPIRVInstruction *SPIRVModuleImpl::addCmpInst(Op TheOpCode, SPIRVType *TheType,
                                              SPIRVValue *Op1, SPIRVValue *Op2,
                                              SPIRVBasicBlock *BB) {
  return addInstruction(SPIRVInstTemplateBase::create(
                            TheOpCode, TheType, getId(),
                            getVec(Op1->getId(), Op2->getId()), BB, this),
                        BB);
}

SPIRVInstruction *SPIRVModuleImpl::addControlBarrierInst(SPIRVValue *ExecKind,
                                                         SPIRVValue *MemKind,
                                                         SPIRVValue *MemSema,
                                                         SPIRVBasicBlock *BB) {
  return addInstruction(new SPIRVControlBarrier(ExecKind, MemKind, MemSema, BB),
                        BB);
}

SPIRVInstruction *SPIRVModuleImpl::addLifetimeInst(Op OC, SPIRVValue *Object,
                                                   SPIRVWord Size,
                                                   SPIRVBasicBlock *BB) {
  if (OC == OpLifetimeStart)
    return BB->addInstruction(
        new SPIRVLifetimeStart(Object->getId(), Size, BB));
  else
    return BB->addInstruction(new SPIRVLifetimeStop(Object->getId(), Size, BB));
}

SPIRVInstruction *SPIRVModuleImpl::addMemoryBarrierInst(Scope ScopeKind,
                                                        SPIRVWord MemFlag,
                                                        SPIRVBasicBlock *BB) {
  return addInstruction(SPIRVInstTemplateBase::create(
                            OpMemoryBarrier, nullptr, SPIRVID_INVALID,
                            getVec(static_cast<SPIRVWord>(ScopeKind), MemFlag),
                            BB, this),
                        BB);
}

SPIRVInstruction *SPIRVModuleImpl::addSelectInst(SPIRVValue *Condition,
                                                 SPIRVValue *Op1,
                                                 SPIRVValue *Op2,
                                                 SPIRVBasicBlock *BB) {
  return addInstruction(
      SPIRVInstTemplateBase::create(
          OpSelect, Op1->getType(), getId(),
          getVec(Condition->getId(), Op1->getId(), Op2->getId()), BB, this),
      BB);
}

SPIRVInstruction *SPIRVModuleImpl::addSelectionMergeInst(
    SPIRVId MergeBlock, SPIRVWord SelectionControl, SPIRVBasicBlock *BB) {
  return addInstruction(
      new SPIRVSelectionMerge(MergeBlock, SelectionControl, BB), BB);
}

SPIRVInstruction *SPIRVModuleImpl::addLoopMergeInst(
    SPIRVId MergeBlock, SPIRVId ContinueTarget, SPIRVWord LoopControl,
    std::vector<SPIRVWord> LoopControlParameters, SPIRVBasicBlock *BB) {
  return addInstruction(
      new SPIRVLoopMerge(MergeBlock, ContinueTarget, LoopControl,
                         LoopControlParameters, BB),
      BB, const_cast<SPIRVInstruction *>(BB->getTerminateInstr()));
}

SPIRVInstruction *SPIRVModuleImpl::addLoopControlINTELInst(
    SPIRVWord LoopControl, std::vector<SPIRVWord> LoopControlParameters,
    SPIRVBasicBlock *BB) {
  addCapability(CapabilityUnstructuredLoopControlsINTEL);
  addExtension(ExtensionID::SPV_INTEL_unstructured_loop_controls);
  return addInstruction(
      new SPIRVLoopControlINTEL(LoopControl, LoopControlParameters, BB), BB,
      const_cast<SPIRVInstruction *>(BB->getTerminateInstr()));
}

SPIRVInstruction *SPIRVModuleImpl::addFixedPointIntelInst(
    Op OC, SPIRVType *ResTy, SPIRVValue *Input,
    const std::vector<SPIRVWord> &Ops, SPIRVBasicBlock *BB) {
  std::vector<SPIRVWord> TheOps = getVec(Input->getId(), Ops);
  return addInstruction(
      SPIRVInstTemplateBase::create(OC, ResTy, getId(), TheOps, BB, this), BB);
}

SPIRVInstruction *SPIRVModuleImpl::addArbFloatPointIntelInst(
    Op OC, SPIRVType *ResTy, SPIRVValue *InA, SPIRVValue *InB,
    const std::vector<SPIRVWord> &Ops, SPIRVBasicBlock *BB) {
  // SPIR-V format:
  //   A<id> [Literal MA] [B<id>] [Literal MB] [Literal Mout] [Literal Sign]
  //   [Literal EnableSubnormals Literal RoundingMode Literal RoundingAccuracy]
  auto OpsItr = Ops.begin();
  std::vector<SPIRVWord> TheOps = getVec(InA->getId(), *OpsItr++);
  if (InB)
    TheOps.push_back(InB->getId());
  TheOps.insert(TheOps.end(), OpsItr, Ops.end());

  return addInstruction(
      SPIRVInstTemplateBase::create(OC, ResTy, getId(), TheOps, BB, this), BB);
}

SPIRVInstruction *
SPIRVModuleImpl::addPtrAccessChainInst(SPIRVType *Type, SPIRVValue *Base,
                                       std::vector<SPIRVValue *> Indices,
                                       SPIRVBasicBlock *BB, bool IsInBounds) {
  return addInstruction(
      SPIRVInstTemplateBase::create(
          IsInBounds ? OpInBoundsPtrAccessChain : OpPtrAccessChain, Type,
          getId(), getVec(Base->getId(), Base->getIds(Indices)), BB, this),
      BB);
}

SPIRVInstruction *SPIRVModuleImpl::addAsyncGroupCopy(
    SPIRVValue *Scope, SPIRVValue *Dest, SPIRVValue *Src, SPIRVValue *NumElems,
    SPIRVValue *Stride, SPIRVValue *Event, SPIRVBasicBlock *BB) {
  return addInstruction(new SPIRVGroupAsyncCopy(Scope, getId(), Dest, Src,
                                                NumElems, Stride, Event, BB),
                        BB);
}

SPIRVInstruction *SPIRVModuleImpl::addCompositeConstructInst(
    SPIRVType *Type, const std::vector<SPIRVId> &Constituents,
    SPIRVBasicBlock *BB) {
  return addInstruction(
      new SPIRVCompositeConstruct(Type, getId(), Constituents, BB), BB);
}

SPIRVInstruction *
SPIRVModuleImpl::addCompositeExtractInst(SPIRVType *Type, SPIRVValue *TheVector,
                                         const std::vector<SPIRVWord> &Indices,
                                         SPIRVBasicBlock *BB) {
  return addInstruction(SPIRVInstTemplateBase::create(
                            OpCompositeExtract, Type, getId(),
                            getVec(TheVector->getId(), Indices), BB, this),
                        BB);
}

SPIRVInstruction *SPIRVModuleImpl::addCompositeInsertInst(
    SPIRVValue *Object, SPIRVValue *Composite,
    const std::vector<SPIRVWord> &Indices, SPIRVBasicBlock *BB) {
  std::vector<SPIRVId> Ops{Object->getId(), Composite->getId()};
  Ops.insert(Ops.end(), Indices.begin(), Indices.end());
  return addInstruction(SPIRVInstTemplateBase::create(OpCompositeInsert,
                                                      Composite->getType(),
                                                      getId(), Ops, BB, this),
                        BB);
}

SPIRVInstruction *SPIRVModuleImpl::addCopyObjectInst(SPIRVType *TheType,
                                                     SPIRVValue *Operand,
                                                     SPIRVBasicBlock *BB) {
  return addInstruction(new SPIRVCopyObject(TheType, getId(), Operand, BB), BB);
}

SPIRVInstruction *SPIRVModuleImpl::addCopyMemoryInst(
    SPIRVValue *TheTarget, SPIRVValue *TheSource,
    const std::vector<SPIRVWord> &TheMemoryAccess, SPIRVBasicBlock *BB) {
  return addInstruction(
      new SPIRVCopyMemory(TheTarget, TheSource, TheMemoryAccess, BB), BB);
}

SPIRVInstruction *SPIRVModuleImpl::addCopyMemorySizedInst(
    SPIRVValue *TheTarget, SPIRVValue *TheSource, SPIRVValue *TheSize,
    const std::vector<SPIRVWord> &TheMemoryAccess, SPIRVBasicBlock *BB) {
  return addInstruction(new SPIRVCopyMemorySized(TheTarget, TheSource, TheSize,
                                                 TheMemoryAccess, BB),
                        BB);
}

SPIRVInstruction *SPIRVModuleImpl::addFPGARegINTELInst(SPIRVType *Type,
                                                       SPIRVValue *V,
                                                       SPIRVBasicBlock *BB) {
  return addInstruction(
      SPIRVInstTemplateBase::create(OpFPGARegINTEL, Type, getId(),
                                    getVec(V->getId()), BB, this),
      BB);
}

SPIRVInstruction *SPIRVModuleImpl::addSampledImageInst(SPIRVType *ResultTy,
                                                       SPIRVValue *Image,
                                                       SPIRVValue *Sampler,
                                                       SPIRVBasicBlock *BB) {
  return addInstruction(SPIRVInstTemplateBase::create(
                            OpSampledImage, ResultTy, getId(),
                            getVec(Image->getId(), Sampler->getId()), BB, this),
                        BB);
}

SPIRVInstruction *SPIRVModuleImpl::addAssumeTrueKHRInst(SPIRVValue *Condition,
                                                        SPIRVBasicBlock *BB) {
  return addInstruction(new SPIRVAssumeTrueKHR(Condition->getId(), BB), BB);
}

SPIRVInstruction *SPIRVModuleImpl::addExpectKHRInst(SPIRVType *ResultTy,
                                                    SPIRVValue *Value,
                                                    SPIRVValue *ExpectedValue,
                                                    SPIRVBasicBlock *BB) {
  return addInstruction(SPIRVInstTemplateBase::create(
                            OpExpectKHR, ResultTy, getId(),
                            getVec(Value->getId(), ExpectedValue->getId()), BB,
                            this),
                        BB);
}

// Create AliasDomainDeclINTEL/AliasScopeDeclINTEL/AliasScopeListDeclINTEL
// instructions
template <typename AliasingInstType>
SPIRVEntry *
SPIRVModuleImpl::getOrAddMemAliasingINTELInst(std::vector<SPIRVId> Args,
                                              llvm::MDNode *MD) {
  assert(MD && "noalias/alias.scope metadata can't be null");
  // Don't duplicate aliasing instruction. For that use a map with a MDNode key
  if (AliasInstMDMap.find(MD) != AliasInstMDMap.end())
    return AliasInstMDMap[MD];
  SPIRVEntry *AliasInst = add(new AliasingInstType(this, getId(), Args));
  AliasInstMDMap.emplace(std::make_pair(MD, AliasInst));
  return AliasInst;
}

// Create AliasDomainDeclINTEL instruction
SPIRVEntry *
SPIRVModuleImpl::getOrAddAliasDomainDeclINTELInst(std::vector<SPIRVId> Args,
                                                  llvm::MDNode *MD) {
  return getOrAddMemAliasingINTELInst<SPIRVAliasDomainDeclINTEL>(Args, MD);
}

// Create AliasScopeDeclINTEL instruction
SPIRVEntry *
SPIRVModuleImpl::getOrAddAliasScopeDeclINTELInst(std::vector<SPIRVId> Args,
                                                 llvm::MDNode *MD) {
  return getOrAddMemAliasingINTELInst<SPIRVAliasScopeDeclINTEL>(Args, MD);
}

// Create AliasScopeListDeclINTEL instruction
SPIRVEntry *
SPIRVModuleImpl::getOrAddAliasScopeListDeclINTELInst(std::vector<SPIRVId> Args,
                                                     llvm::MDNode *MD) {
  return getOrAddMemAliasingINTELInst<SPIRVAliasScopeListDeclINTEL>(Args, MD);
}

SPIRVInstruction *SPIRVModuleImpl::addVariable(
    SPIRVType *Type, bool IsConstant, SPIRVLinkageTypeKind LinkageTy,
    SPIRVValue *Initializer, const std::string &Name,
    SPIRVStorageClassKind StorageClass, SPIRVBasicBlock *BB) {
  SPIRVVariable *Variable = new SPIRVVariable(Type, getId(), Initializer, Name,
                                              StorageClass, BB, this);
  if (BB)
    return addInstruction(Variable, BB);

  add(Variable);
  if (LinkageTy != internal::LinkageTypeInternal)
    Variable->setLinkageType(LinkageTy);
  Variable->setIsConstant(IsConstant);
  return Variable;
}

template <class T>
spv_ostream &operator<<(spv_ostream &O, const std::vector<T *> &V) {
  for (auto &I : V)
    O << *I;
  return O;
}

template <class T, class B = std::less<T>>
spv_ostream &operator<<(spv_ostream &O, const std::unordered_set<T *, B> &V) {
  for (auto &I : V)
    O << *I;
  return O;
}

// To satisfy SPIR-V spec requirement:
// "All operands must be declared before being used",
// we do DFS based topological sort
// https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
class TopologicalSort {
  enum DFSState : char { Unvisited, Discovered, Visited };
  typedef std::vector<SPIRVType *> SPIRVTypeVec;
  typedef std::vector<SPIRVValue *> SPIRVConstantVector;
  typedef std::vector<SPIRVVariable *> SPIRVVariableVec;
  typedef std::vector<SPIRVEntry *> SPIRVConstAndVarVec;
  typedef std::vector<SPIRVTypeForwardPointer *> SPIRVForwardPointerVec;
  typedef std::function<bool(SPIRVEntry *, SPIRVEntry *)> Comp;
  typedef std::map<SPIRVEntry *, DFSState, Comp> EntryStateMapTy;
  typedef std::function<bool(const SPIRVTypeForwardPointer *,
                             const SPIRVTypeForwardPointer *)>
      Equal;
  typedef std::function<size_t(const SPIRVTypeForwardPointer *)> Hash;
  // We may create forward pointers as we go through the types. We use
  // unordered set to avoid duplicates.
  typedef std::unordered_set<SPIRVTypeForwardPointer *, Hash, Equal>
      SPIRVForwardPointerSet;

  SPIRVTypeVec TypeIntVec;
  SPIRVConstantVector ConstIntVec;
  SPIRVTypeVec TypeVec;
  SPIRVConstAndVarVec ConstAndVarVec;
  SPIRVForwardPointerSet ForwardPointerSet;
  EntryStateMapTy EntryStateMap;

  friend spv_ostream &operator<<(spv_ostream &O, const TopologicalSort &S);

  // This method implements recursive depth-first search among all Entries in
  // EntryStateMap. Traversing entries and adding them to corresponding
  // container after visiting all dependent entries(post-order traversal)
  // guarantees that the entry's operands will appear in the container before
  // the entry itslef.
  // Returns true if cyclic dependency detected.
  bool visit(SPIRVEntry *E) {
    DFSState &State = EntryStateMap[E];
    if (State == Visited)
      return false;
    if (State == Discovered) // Cyclic dependency detected
      return true;
    State = Discovered;
    for (SPIRVEntry *Op : E->getNonLiteralOperands()) {
      if (Op->getOpCode() == OpTypeForwardPointer) {
        SPIRVEntry *FP = E->getModule()->getEntry(
            static_cast<SPIRVTypeForwardPointer *>(Op)->getPointerId());
        Op = FP;
      }
      if (EntryStateMap[Op] == Visited)
        continue;
      if (visit(Op)) {
        // We've found a recursive data type, e.g. a structure having a member
        // which is a pointer to the same structure.
        State = Unvisited; // Forget about it
        if (E->getOpCode() == OpTypePointer) {
          // If we have a pointer in the recursive chain, we can break the
          // cyclic dependency by inserting a forward declaration of that
          // pointer.
          SPIRVTypePointer *Ptr = static_cast<SPIRVTypePointer *>(E);
          SPIRVModule *BM = E->getModule();
          ForwardPointerSet.insert(BM->add(new SPIRVTypeForwardPointer(
              BM, Ptr->getId(), Ptr->getPointerStorageClass())));
          return false;
        }
        return true;
      }
    }
    Op OC = E->getOpCode();
    if (OC == OpTypeInt)
      TypeIntVec.push_back(static_cast<SPIRVType *>(E));
    else if (isConstantOpCode(OC)) {
      SPIRVConstant *C = static_cast<SPIRVConstant *>(E);
      if (C->getType()->isTypeInt())
        ConstIntVec.push_back(C);
      else
        ConstAndVarVec.push_back(E);
    } else if (isTypeOpCode(OC))
      TypeVec.push_back(static_cast<SPIRVType *>(E));
    else
      ConstAndVarVec.push_back(E);
    State = Visited;
    return false;
  }

public:
  TopologicalSort(const SPIRVTypeVec &TypeVec,
                  const SPIRVConstantVector &ConstVec,
                  const SPIRVVariableVec &VariableVec,
                  SPIRVForwardPointerVec &ForwardPointerVec)
      : ForwardPointerSet(
            16, // bucket count
            [](const SPIRVTypeForwardPointer *Ptr) {
              return std::hash<SPIRVId>()(Ptr->getPointerId());
            },
            [](const SPIRVTypeForwardPointer *Ptr1,
               const SPIRVTypeForwardPointer *Ptr2) {
              return Ptr1->getPointerId() == Ptr2->getPointerId();
            }),
        EntryStateMap([](SPIRVEntry *A, SPIRVEntry *B) -> bool {
          return A->getId() < B->getId();
        }) {
    // Collect entries for sorting
    for (auto *T : TypeVec)
      EntryStateMap[T] = DFSState::Unvisited;
    for (auto *C : ConstVec)
      EntryStateMap[C] = DFSState::Unvisited;
    for (auto *V : VariableVec)
      EntryStateMap[V] = DFSState::Unvisited;
    // Run topoligical sort
    for (const auto &ES : EntryStateMap) {
      if (visit(ES.first))
        llvm_unreachable("Cyclic dependency for types detected");
    }
    // Append forward pointers vector
    ForwardPointerVec.insert(ForwardPointerVec.end(), ForwardPointerSet.begin(),
                             ForwardPointerSet.end());
  }
};

spv_ostream &operator<<(spv_ostream &O, const TopologicalSort &S) {
  O << S.TypeIntVec << S.ConstIntVec << S.TypeVec << S.ConstAndVarVec;
  return O;
}

spv_ostream &operator<<(spv_ostream &O, SPIRVModule &M) {
  SPIRVModuleImpl &MI = *static_cast<SPIRVModuleImpl *>(&M);
  // Start tracking of the current line with no line
  MI.CurrentLine.reset();
  MI.CurrentDebugLine.reset();

  SPIRVEncoder Encoder(O);
  Encoder << MagicNumber << MI.SPIRVVersion
          << (((SPIRVWord)MI.GeneratorId << 16) | MI.GeneratorVer)
          << MI.NextId /* Bound for Id */
          << MI.InstSchema;
  O << SPIRVNL();

  for (auto &I : MI.CapMap)
    O << *I.second;

  for (auto &I : M.getExtension()) {
    assert(!I.empty() && "Invalid extension");
    O << SPIRVExtension(&M, I);
  }

  for (auto &I : MI.IdToInstSetMap)
    O << SPIRVExtInstImport(&M, I.first, SPIRVBuiltinSetNameMap::map(I.second));

  O << SPIRVMemoryModel(&M);

  O << MI.EntryPointVec;

  for (auto &I : MI.EntryPointVec)
    MI.get<SPIRVFunction>(I->getTargetId())->encodeExecutionModes(O);

  O << MI.StringVec;

  for (auto &I : M.getSourceExtension()) {
    assert(!I.empty() && "Invalid source extension");
    O << SPIRVSourceExtension(&M, I);
  }

  O << SPIRVSource(&M);

  for (auto &I : MI.NamedId) {
    // Don't output name for entry point since it is redundant
    bool IsEntryPoint = false;
    for (auto &EPS : MI.EntryPointSet)
      if (EPS.second.count(I)) {
        IsEntryPoint = true;
        break;
      }
    if (!IsEntryPoint)
      M.getEntry(I)->encodeName(O);
  }

  if (M.isAllowedToUseExtension(
          ExtensionID::SPV_INTEL_memory_access_aliasing)) {
    O << SPIRVNL() << MI.AliasInstMDVec;
  }

  TopologicalSort TS(MI.TypeVec, MI.ConstVec, MI.VariableVec,
                     MI.ForwardPointerVec);

  O << MI.MemberNameVec << MI.ModuleProcessedVec << MI.DecGroupVec
    << MI.DecorateVec << MI.GroupDecVec << MI.ForwardPointerVec << TS;

  if (M.isAllowedToUseExtension(ExtensionID::SPV_INTEL_inline_assembly)) {
    O << SPIRVNL() << MI.AsmTargetVec << MI.AsmVec;
  }

  // At this point we know that FunctionDefinition could have been included both
  // into DebugInstVec and into basick block of function from FuncVec.
  // By spec we should only have this instruction to be present inside the
  // function body, so removing it from the DebugInstVec to avoid duplication.
  MI.DebugInstVec.erase(
      std::remove_if(MI.DebugInstVec.begin(), MI.DebugInstVec.end(),
                     [](SPIRVExtInst *I) {
                       return I->getExtOp() == SPIRVDebug::FunctionDefinition;
                     }),
      MI.DebugInstVec.end());

  O << SPIRVNL() << MI.DebugInstVec << MI.AuxDataInstVec << SPIRVNL()
    << MI.FuncVec;
  return O;
}

template <class T>
void SPIRVModuleImpl::addTo(std::vector<T *> &V, SPIRVEntry *E) {
  V.push_back(static_cast<T *>(E));
}

// The first decoration group includes all the previously defined decorates.
// The second decoration group includes all the decorates defined between the
// first and second decoration group. So long so forth.
SPIRVDecorationGroup *SPIRVModuleImpl::addDecorationGroup() {
  return addDecorationGroup(new SPIRVDecorationGroup(this, getId()));
}

SPIRVDecorationGroup *
SPIRVModuleImpl::addDecorationGroup(SPIRVDecorationGroup *Group) {
  add(Group);
  Group->takeDecorates(DecorateVec);
  DecGroupVec.push_back(Group);
  SPIRVDBG(spvdbgs() << "[addDecorationGroup] {" << *Group << "}\n";
           spvdbgs() << "  Remaining DecorateVec: {" << DecorateVec << "}\n");
  assert(DecorateVec.empty());
  return Group;
}

SPIRVGroupDecorateGeneric *
SPIRVModuleImpl::addGroupDecorateGeneric(SPIRVGroupDecorateGeneric *GDec) {
  add(GDec);
  GDec->decorateTargets();
  GroupDecVec.push_back(GDec);
  return GDec;
}
SPIRVGroupDecorate *
SPIRVModuleImpl::addGroupDecorate(SPIRVDecorationGroup *Group,
                                  const std::vector<SPIRVEntry *> &Targets) {
  auto *GD = new SPIRVGroupDecorate(Group, getIds(Targets));
  addGroupDecorateGeneric(GD);
  return GD;
}

SPIRVGroupMemberDecorate *SPIRVModuleImpl::addGroupMemberDecorate(
    SPIRVDecorationGroup *Group, const std::vector<SPIRVEntry *> &Targets) {
  auto *GMD = new SPIRVGroupMemberDecorate(Group, getIds(Targets));
  addGroupDecorateGeneric(GMD);
  return GMD;
}

SPIRVString *SPIRVModuleImpl::getString(const std::string &Str) {
  auto Loc = StrMap.find(Str);
  if (Loc != StrMap.end())
    return Loc->second;
  auto *S = add(new SPIRVString(this, getId(), Str));
  StrMap[Str] = S;
  return S;
}

SPIRVMemberName *SPIRVModuleImpl::addMemberName(SPIRVTypeStruct *ST,
                                                SPIRVWord MemberNumber,
                                                const std::string &Name) {
  return add(new SPIRVMemberName(ST, MemberNumber, Name));
}

void SPIRVModuleImpl::addUnknownStructField(SPIRVTypeStruct *Struct, unsigned I,
                                            SPIRVId ID) {
  UnknownStructFieldMap[Struct].push_back(std::make_pair(I, ID));
}

static std::string to_string(uint32_t Version) {
  std::string Res(formatVersionNumber(Version));
  Res += " (" + std::to_string(Version) + ")";
  return Res;
}

static std::string to_string(VersionNumber Version) {
  return to_string(static_cast<uint32_t>(Version));
}

std::istream &operator>>(std::istream &I, SPIRVModule &M) {
  SPIRVDecoder Decoder(I, M);
  SPIRVModuleImpl &MI = *static_cast<SPIRVModuleImpl *>(&M);
  // Disable automatic capability filling.
  MI.setAutoAddCapability(false);
  MI.setAutoAddExtensions(false);

  SPIRVWord Magic;
  Decoder >> Magic;
  if (!M.getErrorLog().checkError(Magic == MagicNumber, SPIRVEC_InvalidModule,
                                  "invalid magic number")) {
    M.setInvalid();
    return I;
  }

  Decoder >> MI.SPIRVVersion;
  bool SPIRVVersionIsKnown = isSPIRVVersionKnown(MI.SPIRVVersion);
  if (!M.getErrorLog().checkError(
          SPIRVVersionIsKnown, SPIRVEC_InvalidModule,
          "unsupported SPIR-V version number '" + to_string(MI.SPIRVVersion) +
              "'. Range of supported/known SPIR-V "
              "versions is " +
              to_string(VersionNumber::MinimumVersion) + " - " +
              to_string(VersionNumber::MaximumVersion))) {
    M.setInvalid();
    return I;
  }

  bool SPIRVVersionIsAllowed = M.isAllowedToUseVersion(MI.SPIRVVersion);
  if (!M.getErrorLog().checkError(
          SPIRVVersionIsAllowed, SPIRVEC_InvalidModule,
          "incorrect SPIR-V version number " + to_string(MI.SPIRVVersion) +
              " - it conflicts with maximum allowed version which is set to " +
              to_string(M.getMaximumAllowedSPIRVVersion()))) {
    M.setInvalid();
    return I;
  }

  SPIRVWord Generator = 0;
  Decoder >> Generator;
  MI.GeneratorId = Generator >> 16;
  MI.GeneratorVer = Generator & 0xFFFF;

  // Bound for Id
  Decoder >> MI.NextId;

  Decoder >> MI.InstSchema;
  if (!M.getErrorLog().checkError(MI.InstSchema == SPIRVISCH_Default,
                                  SPIRVEC_InvalidModule,
                                  "unsupported instruction schema")) {
    M.setInvalid();
    return I;
  }

  while (Decoder.getWordCountAndOpCode() && M.isModuleValid()) {
    SPIRVEntry *Entry = Decoder.getEntry();
    if (Entry != nullptr)
      M.add(Entry);
  }

  MI.resolveUnknownStructFields();
  return I;
}

SPIRVModule *SPIRVModule::createSPIRVModule() { return new SPIRVModuleImpl(); }

SPIRVModule *SPIRVModule::createSPIRVModule(const SPIRV::TranslatorOpts &Opts) {
  return new SPIRVModuleImpl(Opts);
}

SPIRVValue *SPIRVModuleImpl::getValue(SPIRVId TheId) const {
  return get<SPIRVValue>(TheId);
}

SPIRVType *SPIRVModuleImpl::getValueType(SPIRVId TheId) const {
  return get<SPIRVValue>(TheId)->getType();
}

std::vector<SPIRVValue *>
SPIRVModuleImpl::getValues(const std::vector<SPIRVId> &IdVec) const {
  std::vector<SPIRVValue *> ValueVec;
  for (auto I : IdVec)
    ValueVec.push_back(getValue(I));
  return ValueVec;
}

std::vector<SPIRVType *>
SPIRVModuleImpl::getValueTypes(const std::vector<SPIRVId> &IdVec) const {
  std::vector<SPIRVType *> TypeVec;
  for (auto I : IdVec)
    TypeVec.push_back(getValue(I)->getType());
  return TypeVec;
}

std::vector<SPIRVId>
SPIRVModuleImpl::getIds(const std::vector<SPIRVEntry *> &ValueVec) const {
  std::vector<SPIRVId> IdVec;
  for (auto *I : ValueVec)
    IdVec.push_back(I->getId());
  return IdVec;
}

std::vector<SPIRVId>
SPIRVModuleImpl::getIds(const std::vector<SPIRVValue *> &ValueVec) const {
  std::vector<SPIRVId> IdVec;
  for (auto *I : ValueVec)
    IdVec.push_back(I->getId());
  return IdVec;
}

SPIRVInstTemplateBase *
SPIRVModuleImpl::addInstTemplate(Op OC, SPIRVBasicBlock *BB, SPIRVType *Ty) {
  assert(!Ty || !Ty->isTypeVoid());
  SPIRVId Id = Ty ? getId() : SPIRVID_INVALID;
  auto *Ins = SPIRVInstTemplateBase::create(OC, Ty, Id, BB, this);
  BB->addInstruction(Ins);
  return Ins;
}

SPIRVInstTemplateBase *
SPIRVModuleImpl::addInstTemplate(Op OC, const std::vector<SPIRVWord> &Ops,
                                 SPIRVBasicBlock *BB, SPIRVType *Ty) {
  assert(!Ty || !Ty->isTypeVoid());
  SPIRVId Id = Ty ? getId() : SPIRVID_INVALID;
  auto *Ins = SPIRVInstTemplateBase::create(OC, Ty, Id, Ops, BB, this);
  BB->addInstruction(Ins);
  return Ins;
}

void SPIRVModuleImpl::addInstTemplate(SPIRVInstTemplateBase *Ins,
                                      const std::vector<SPIRVWord> &Ops,
                                      SPIRVBasicBlock *BB, SPIRVType *Ty) {
  assert(!Ty || !Ty->isTypeVoid());
  SPIRVId Id = Ty ? getId() : SPIRVID_INVALID;
  Ins->init(Ty, Id, BB, this);
  Ins->setOpWordsAndValidate(Ops);
  BB->addInstruction(Ins);
}

SPIRVId SPIRVModuleImpl::getExtInstSetId(SPIRVExtInstSetKind Kind) const {
  assert(Kind < SPIRVEIS_Count && "Unknown extended instruction set!");
  auto Res = ExtInstSetIds.find(Kind);
  assert(Res != ExtInstSetIds.end() && "extended instruction set not found!");
  return Res->second;
}

bool isSpirvBinary(const std::string &Img) {
  if (Img.size() < sizeof(unsigned))
    return false;
  const auto *Magic = reinterpret_cast<const unsigned *>(Img.data());
  return *Magic == MagicNumber;
}

#ifdef _SPIRV_SUPPORT_TEXT_FMT

bool convertSpirv(std::istream &IS, std::ostream &OS, std::string &ErrMsg,
                  bool FromText, bool ToText) {
  auto SaveOpt = SPIRVUseTextFormat;
  SPIRVUseTextFormat = FromText;
  // Conversion from/to SPIR-V text representation is a side feature of the
  // translator which is mostly intended for debug usage. So, this step cannot
  // be customized to enable/disable particular extensions or restrict/allow
  // particular SPIR-V versions: all known SPIR-V versions are allowed, all
  // known SPIR-V extensions are enabled during this conversion
  SPIRV::TranslatorOpts DefaultOpts;
  DefaultOpts.enableAllExtensions();
  SPIRVModuleImpl M(DefaultOpts);
  IS >> M;
  if (M.getError(ErrMsg) != SPIRVEC_Success) {
    SPIRVUseTextFormat = SaveOpt;
    return false;
  }
  SPIRVUseTextFormat = ToText;
  OS << M;
  if (M.getError(ErrMsg) != SPIRVEC_Success) {
    SPIRVUseTextFormat = SaveOpt;
    return false;
  }
  SPIRVUseTextFormat = SaveOpt;
  return true;
}

bool isSpirvText(const std::string &Img) {
  std::istringstream SS(Img);
  unsigned Magic = 0;
  SS >> Magic;
  if (SS.bad())
    return false;
  return Magic == MagicNumber;
}

bool convertSpirv(std::string &Input, std::string &Out, std::string &ErrMsg,
                  bool ToText) {
  auto FromText = isSpirvText(Input);
  if (ToText == FromText) {
    Out = Input;
    return true;
  }
  std::istringstream IS(Input);
  std::ostringstream OS;
  if (!convertSpirv(IS, OS, ErrMsg, FromText, ToText))
    return false;
  Out = OS.str();
  return true;
}

#endif // _SPIRV_SUPPORT_TEXT_FMT

} // namespace SPIRV
