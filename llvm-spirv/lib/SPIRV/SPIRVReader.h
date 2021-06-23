//===- SPIRVReader.h - Converts SPIR-V to LLVM ------------------*- C++ -*-===//
//
//                     The LLVM/SPIR-V Translator
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
/// This file contains declaration of SPIRVToLLVM class which implements
/// conversion of SPIR-V binary to LLVM IR.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRVREADER_H
#define SPIRVREADER_H

#include "SPIRVModule.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/GlobalValue.h" // llvm::GlobalValue::LinkageTypes

namespace llvm {
class Metadata;
class Module;
class Type;
class Instruction;
class CallInst;
class BasicBlock;
class Loop;
class Function;
class GlobalVariable;
class LLVMContext;
class MDString;
class IntrinsicInst;
class LoadInst;
class BranchInst;
class BinaryOperator;
class Value;
} // namespace llvm
using namespace llvm;

namespace SPIRV {
class SPIRVFunctionParameter;
class SPIRVConstantSampler;
class SPIRVConstantPipeStorage;
class SPIRVLoopMerge;
class SPIRVToLLVMDbgTran;
class SPIRVToLLVM {
public:
  SPIRVToLLVM(Module *LLVMModule, SPIRVModule *TheSPIRVModule);

  static const StringSet<> BuiltInConstFunc;
  std::string getOCLBuiltinName(SPIRVInstruction *BI);

  Type *transType(SPIRVType *BT, bool IsClassMember = false);
  std::string transTypeToOCLTypeName(SPIRVType *BT, bool IsSigned = true);
  std::vector<Type *> transTypeVector(const std::vector<SPIRVType *> &);
  bool translate();
  bool transAddressingModel();

  Value *transValue(SPIRVValue *, Function *F, BasicBlock *,
                    bool CreatePlaceHolder = true);
  Value *transValueWithoutDecoration(SPIRVValue *, Function *F, BasicBlock *,
                                     bool CreatePlaceHolder = true);
  Value *transDeviceEvent(SPIRVValue *BV, Function *F, BasicBlock *BB);
  bool transDecoration(SPIRVValue *, Value *);
  bool transAlign(SPIRVValue *, Value *);
  Instruction *transOCLBuiltinFromExtInst(SPIRVExtInst *BC, BasicBlock *BB);
  std::vector<Value *> transValue(const std::vector<SPIRVValue *> &,
                                  Function *F, BasicBlock *);
  Function *transFunction(SPIRVFunction *F);
  Value *transBlockInvoke(SPIRVValue *Invoke, BasicBlock *BB);
  Instruction *transEnqueueKernelBI(SPIRVInstruction *BI, BasicBlock *BB);
  Instruction *transWGSizeQueryBI(SPIRVInstruction *BI, BasicBlock *BB);
  Instruction *transSGSizeQueryBI(SPIRVInstruction *BI, BasicBlock *BB);
  bool transFPContractMetadata();
  bool transMetadata();
  bool transOCLMetadata(SPIRVFunction *BF);
  bool transVectorComputeMetadata(SPIRVFunction *BF);
  bool transFPGAFunctionMetadata(SPIRVFunction *BF, Function *F);
  Value *transAsmINTEL(SPIRVAsmINTEL *BA);
  CallInst *transAsmCallINTEL(SPIRVAsmCallINTEL *BI, Function *F,
                              BasicBlock *BB);
  CallInst *transFixedPointInst(SPIRVInstruction *BI, BasicBlock *BB);
  CallInst *transArbFloatInst(SPIRVInstruction *BI, BasicBlock *BB,
                              bool IsBinaryInst = false);
  bool transNonTemporalMetadata(Instruction *I);
  template <typename SPIRVInstType>
  void transAliasingMemAccess(SPIRVInstType *BI, Instruction *I);
  void addMemAliasMetadata(Instruction *I, SPIRVId AliasListId,
                           uint32_t AliasMDKind);
  void transSourceLanguage();
  bool transSourceExtension();
  void transGeneratorMD();
  Value *transConvertInst(SPIRVValue *BV, Function *F, BasicBlock *BB);
  Instruction *transBuiltinFromInst(const std::string &FuncName,
                                    SPIRVInstruction *BI, BasicBlock *BB);
  Instruction *transOCLBuiltinFromInst(SPIRVInstruction *BI, BasicBlock *BB);
  Instruction *transSPIRVBuiltinFromInst(SPIRVInstruction *BI, BasicBlock *BB);

  /// Post-process translated LLVM module for OpenCL.
  bool postProcessOCL();

  /// \brief Post-process OpenCL builtin functions returning struct type.
  ///
  /// Some OpenCL builtin functions are translated to SPIR-V instructions with
  /// struct type result, e.g. NDRange creation functions. Such functions
  /// need to be post-processed to return the struct through sret argument.
  bool postProcessOCLBuiltinReturnStruct(Function *F);

  /// \brief Post-process OpenCL builtin functions having array argument.
  ///
  /// These functions are translated to functions with array type argument
  /// first, then post-processed to have pointer arguments.
  bool postProcessOCLBuiltinWithArrayArguments(Function *F,
                                               StringRef DemangledName);

  /// \brief Post-process OpBuildNDRange.
  ///   OpBuildNDRange GlobalWorkSize, LocalWorkSize, GlobalWorkOffset
  /// =>
  ///   call ndrange_XD(GlobalWorkOffset, GlobalWorkSize, LocalWorkSize)
  /// \return transformed call instruction.
  CallInst *postProcessOCLBuildNDRange(SPIRVInstruction *BI, CallInst *CI,
                                       const std::string &DemangledName);

  /// \brief Expand OCL builtin functions with scalar argument, e.g.
  /// step, smoothstep.
  /// gentype func (fp edge, gentype x)
  /// =>
  /// gentype func (gentype edge, gentype x)
  /// \return transformed call instruction.
  CallInst *expandOCLBuiltinWithScalarArg(CallInst *CI,
                                          const std::string &FuncName);

  typedef DenseMap<SPIRVType *, Type *> SPIRVToLLVMTypeMap;
  typedef DenseMap<SPIRVValue *, Value *> SPIRVToLLVMValueMap;
  typedef DenseMap<SPIRVValue *, Value *> SPIRVBlockToLLVMStructMap;
  typedef DenseMap<SPIRVFunction *, Function *> SPIRVToLLVMFunctionMap;
  typedef DenseMap<GlobalVariable *, SPIRVBuiltinVariableKind> BuiltinVarMap;
  typedef std::unordered_map<SPIRVId, MDNode *> SPIRVToLLVMMDAliasInstMap;

  // A SPIRV value may be translated to a load instruction of a placeholder
  // global variable. This map records load instruction of these placeholders
  // which are supposed to be replaced by the real values later.
  typedef std::map<SPIRVValue *, LoadInst *> SPIRVToLLVMPlaceholderMap;

  typedef std::map<const BasicBlock *, const SPIRVValue *>
      SPIRVToLLVMLoopMetadataMap;

private:
  Module *M;
  BuiltinVarMap BuiltinGVMap;
  LLVMContext *Context;
  SPIRVModule *BM;
  SPIRVToLLVMTypeMap TypeMap;
  SPIRVToLLVMValueMap ValueMap;
  SPIRVToLLVMFunctionMap FuncMap;
  SPIRVBlockToLLVMStructMap BlockMap;
  SPIRVToLLVMPlaceholderMap PlaceholderMap;
  std::unique_ptr<SPIRVToLLVMDbgTran> DbgTran;
  std::vector<Constant *> GlobalAnnotations;

  // Loops metadata is translated in the end of a function translation.
  // This storage contains pairs of translated loop header basic block and loop
  // metadata SPIR-V instruction in SPIR-V representation of this basic block.
  SPIRVToLLVMLoopMetadataMap FuncLoopMetadataMap;

  // These storages are used to prevent duplication of alias.scope/noalias
  // metadata
  SPIRVToLLVMMDAliasInstMap MDAliasDomainMap;
  SPIRVToLLVMMDAliasInstMap MDAliasScopeMap;
  SPIRVToLLVMMDAliasInstMap MDAliasListMap;

  Type *mapType(SPIRVType *BT, Type *T);

  // If a value is mapped twice, the existing mapped value is a placeholder,
  // which must be a load instruction of a global variable whose name starts
  // with kPlaceholderPrefix.
  Value *mapValue(SPIRVValue *BV, Value *V);

  bool isSPIRVBuiltinVariable(GlobalVariable *GV,
                              SPIRVBuiltinVariableKind *Kind = nullptr);

  // OpenCL function always has NoUnwind attribute.
  // Change this if it is no longer true.
  bool isFuncNoUnwind() const { return true; }

  bool isFuncReadNone(const std::string &Name) const {
    return BuiltInConstFunc.count(Name);
  }

  bool isSPIRVCmpInstTransToLLVMInst(SPIRVInstruction *BI) const;
  bool isDirectlyTranslatedToOCL(Op OpCode) const;
  bool transOCLBuiltinsFromVariables();
  bool transOCLBuiltinFromVariable(GlobalVariable *GV,
                                   SPIRVBuiltinVariableKind Kind);
  MDString *transOCLKernelArgTypeName(SPIRVFunctionParameter *);
  Value *mapFunction(SPIRVFunction *BF, Function *F);
  Value *getTranslatedValue(SPIRVValue *BV);
  IntrinsicInst *getLifetimeStartIntrinsic(Instruction *I);
  SPIRVErrorLog &getErrorLog();
  void setCallingConv(CallInst *Call);
  void setAttrByCalledFunc(CallInst *Call);
  Type *transFPType(SPIRVType *T);
  Value *transShiftLogicalBitwiseInst(SPIRVValue *BV, BasicBlock *BB,
                                      Function *F);
  Value *transCmpInst(SPIRVValue *BV, BasicBlock *BB, Function *F);
  void transOCLBuiltinFromInstPreproc(SPIRVInstruction *BI, Type *&RetTy,
                                      std::vector<SPIRVValue *> &Args);
  Instruction *transOCLBuiltinPostproc(SPIRVInstruction *BI, CallInst *CI,
                                       BasicBlock *BB,
                                       const std::string &DemangledName);
  std::string transOCLImageTypeName(SPIRV::SPIRVTypeImage *ST);
  std::string transOCLSampledImageTypeName(SPIRV::SPIRVTypeSampledImage *ST);
  std::string transOCLPipeTypeName(
      SPIRV::SPIRVTypePipe *ST, bool UseSPIRVFriendlyFormat = false,
      SPIRVAccessQualifierKind PipeAccess = AccessQualifierReadOnly);
  std::string transOCLPipeStorageTypeName(SPIRV::SPIRVTypePipeStorage *PST);
  std::string transOCLImageTypeAccessQualifier(SPIRV::SPIRVTypeImage *ST);
  std::string transOCLPipeTypeAccessQualifier(SPIRV::SPIRVTypePipe *ST);
  std::string transVCTypeName(SPIRVTypeBufferSurfaceINTEL *PST);

  Value *oclTransConstantSampler(SPIRV::SPIRVConstantSampler *BCS,
                                 BasicBlock *BB);
  Value *oclTransConstantPipeStorage(SPIRV::SPIRVConstantPipeStorage *BCPS);
  void setName(llvm::Value *V, SPIRVValue *BV);
  template <typename LoopInstType>
  void setLLVMLoopMetadata(const LoopInstType *LM, const Loop *LoopObj);
  void transLLVMLoopMetadata(const Function *F);
  inline llvm::Metadata *getMetadataFromName(std::string Name);
  inline std::vector<llvm::Metadata *>
  getMetadataFromNameAndParameter(std::string Name, SPIRVWord Parameter);
  void insertImageNameAccessQualifier(SPIRV::SPIRVTypeImage *ST,
                                      std::string &Name);
  template <class Source, class Func> bool foreachFuncCtlMask(Source, Func);
  llvm::GlobalValue::LinkageTypes transLinkageType(const SPIRVValue *V);
  Instruction *transOCLAllAny(SPIRVInstruction *BI, BasicBlock *BB);
  Instruction *transOCLRelational(SPIRVInstruction *BI, BasicBlock *BB);

  void transUserSemantic(SPIRV::SPIRVFunction *Fun);
  void transGlobalAnnotations();
  void transGlobalCtorDtors(SPIRVVariable *BV);
  void createCXXStructor(const char *ListName,
                         SmallVectorImpl<Function *> &Funcs);
  void transIntelFPGADecorations(SPIRVValue *BV, Value *V);
  void transMemAliasingINTELDecorations(SPIRVValue *BV, Value *V);
}; // class SPIRVToLLVM

} // namespace SPIRV

#endif // SPIRVREADER_H
