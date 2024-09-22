//===- SPIRVEntry.cpp - Base Class for SPIR-V Entities ----------*- C++ -*-===//
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
/// This file implements base class for SPIR-V entities.
///
//===----------------------------------------------------------------------===//

#include "SPIRVEntry.h"
#include "SPIRVAsm.h"
#include "SPIRVBasicBlock.h"
#include "SPIRVDebug.h"
#include "SPIRVDecorate.h"
#include "SPIRVFunction.h"
#include "SPIRVInstruction.h"
#include "SPIRVMemAliasingINTEL.h"
#include "SPIRVNameMapEnum.h"
#include "SPIRVStream.h"
#include "SPIRVType.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>

using namespace SPIRV;

namespace SPIRV {

template <typename T> SPIRVEntry *create() { return new T(); }

SPIRVEntry *SPIRVEntry::create(Op OpCode) {
  typedef SPIRVEntry *(*SPIRVFactoryTy)();
  struct TableEntry {
    Op Opn;
    SPIRVFactoryTy Factory;
    operator std::pair<const Op, SPIRVFactoryTy>() {
      return std::make_pair(Opn, Factory);
    }
  };

  static TableEntry Table[] = {
#define _SPIRV_OP(x, ...) {Op##x, &SPIRV::create<SPIRV##x>},
#define _SPIRV_OP_INTERNAL(x, ...) {internal::Op##x, &SPIRV::create<SPIRV##x>},
#include "SPIRVOpCodeEnum.h"
#include "SPIRVOpCodeEnumInternal.h"
#undef _SPIRV_OP_INTERNAL
#undef _SPIRV_OP
  };

  typedef std::unordered_map<Op, SPIRVFactoryTy> OpToFactoryMapTy;
  static const OpToFactoryMapTy OpToFactoryMap(std::begin(Table),
                                               std::end(Table));

  // TODO: To remove this when we make a switch to new version
  if (OpCode == internal::OpTypeJointMatrixINTELv2)
    OpCode = internal::OpTypeJointMatrixINTEL;

  OpToFactoryMapTy::const_iterator Loc = OpToFactoryMap.find(OpCode);
  if (Loc != OpToFactoryMap.end())
    return Loc->second();

  SPIRVDBG(spvdbgs() << "No factory for OpCode " << (unsigned)OpCode << '\n';)
  assert(0 && "Not implemented");
  return 0;
}

std::unique_ptr<SPIRV::SPIRVEntry> SPIRVEntry::createUnique(Op OC) {
  return std::unique_ptr<SPIRVEntry>(create(OC));
}

std::unique_ptr<SPIRV::SPIRVExtInst>
SPIRVEntry::createUnique(SPIRVExtInstSetKind Set, unsigned ExtOp) {
  return std::unique_ptr<SPIRVExtInst>(new SPIRVExtInst(Set, ExtOp));
}

SPIRVErrorLog &SPIRVEntry::getErrorLog() const { return Module->getErrorLog(); }

bool SPIRVEntry::exist(SPIRVId TheId) const { return Module->exist(TheId); }

SPIRVEntry *SPIRVEntry::getOrCreate(SPIRVId TheId) const {
  SPIRVEntry *Entry = nullptr;
  bool Found = Module->exist(TheId, &Entry);
  if (!Found)
    return Module->addForward(TheId, nullptr);
  return Entry;
}

SPIRVValue *SPIRVEntry::getValue(SPIRVId TheId) const {
  return get<SPIRVValue>(TheId);
}

SPIRVType *SPIRVEntry::getValueType(SPIRVId TheId) const {
  return get<SPIRVValue>(TheId)->getType();
}

SPIRVEncoder SPIRVEntry::getEncoder(spv_ostream &O) const {
  return SPIRVEncoder(O);
}

SPIRVDecoder SPIRVEntry::getDecoder(std::istream &I) {
  return SPIRVDecoder(I, *Module);
}

void SPIRVEntry::setWordCount(SPIRVWord TheWordCount) {
  WordCount = TheWordCount;
}

void SPIRVEntry::setName(const std::string &TheName) {
  Name = TheName;
  SPIRVDBG(spvdbgs() << "Set name for obj " << Id << " " << Name << '\n');
}

void SPIRVEntry::setModule(SPIRVModule *TheModule) {
  assert(TheModule && "Invalid module");
  if (TheModule == Module)
    return;
  assert(Module == NULL && "Cannot change owner of entry");
  Module = TheModule;
}

void SPIRVEntry::encode(spv_ostream &O) const {
  assert(0 && "Not implemented");
}

void SPIRVEntry::encodeName(spv_ostream &O) const {
  if (!Name.empty())
    O << SPIRVName(this, Name);
}

bool SPIRVEntry::isEndOfBlock() const {
  switch (OpCode) {
  case OpBranch:
  case OpBranchConditional:
  case OpSwitch:
  case OpKill:
  case OpReturn:
  case OpReturnValue:
  case OpUnreachable:
    return true;
  default:
    return false;
  }
}

void SPIRVEntry::encodeLine(spv_ostream &O) const {
  if (!Module)
    return;
  const std::shared_ptr<const SPIRVLine> &CurrLine = Module->getCurrentLine();
  if (Line && (!CurrLine || *Line != *CurrLine)) {
    O << *Line;
    Module->setCurrentLine(Line);
  }
  if (isEndOfBlock() || OpCode == OpNoLine)
    Module->setCurrentLine(nullptr);
}

namespace {
bool isDebugLineEqual(const SPIRVExtInst &DL1, const SPIRVExtInst &DL2) {
  std::vector<SPIRVWord> DL1Args = DL1.getArguments();
  std::vector<SPIRVWord> DL2Args = DL2.getArguments();

  using namespace SPIRVDebug::Operand::DebugLine;
  assert(DL1Args.size() == OperandCount && DL2Args.size() == OperandCount &&
         "Invalid number of operands");
  return DL1Args[SourceIdx] == DL2Args[SourceIdx] &&
         DL1Args[StartIdx] == DL2Args[StartIdx] &&
         DL1Args[EndIdx] == DL2Args[EndIdx] &&
         DL1Args[ColumnStartIdx] == DL2Args[ColumnStartIdx] &&
         DL1Args[ColumnEndIdx] == DL2Args[ColumnEndIdx];
}
} // namespace

void SPIRVEntry::encodeDebugLine(spv_ostream &O) const {
  if (!Module)
    return;
  const std::shared_ptr<const SPIRVExtInst> &CurrDebugLine =
      Module->getCurrentDebugLine();
  if (DebugLine &&
      (!CurrDebugLine || !isDebugLineEqual(*DebugLine, *CurrDebugLine))) {
    O << *DebugLine;
    Module->setCurrentDebugLine(DebugLine);
  }
  if (isEndOfBlock() ||
      isExtInst(SPIRVEIS_NonSemantic_Shader_DebugInfo_100,
                SPIRVDebug::DebugNoLine) ||
      isExtInst(SPIRVEIS_NonSemantic_Shader_DebugInfo_200,
                SPIRVDebug::DebugNoLine))
    Module->setCurrentDebugLine(nullptr);
}

void SPIRVEntry::encodeAll(spv_ostream &O) const {
  encodeLine(O);
  encodeDebugLine(O);
  encodeWordCountOpCode(O);
  encode(O);
  encodeChildren(O);
}

void SPIRVEntry::encodeChildren(spv_ostream &O) const {}

void SPIRVEntry::encodeWordCountOpCode(spv_ostream &O) const {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
  if (SPIRVUseTextFormat) {
    getEncoder(O) << WordCount << OpCode;
    return;
  }
#endif
  assert(WordCount < 65536 && "WordCount must fit into 16-bit value");
  SPIRVWord WordCountOpCode = (WordCount << WordCountShift) | OpCode;
  getEncoder(O) << WordCountOpCode;
}
// Read words from SPIRV binary and create members for SPIRVEntry.
// The word count and op code has already been read before calling this
// function for creating the SPIRVEntry. Therefore the input stream only
// contains the remaining part of the words for the SPIRVEntry.
void SPIRVEntry::decode(std::istream &I) { assert(0 && "Not implemented"); }

std::vector<SPIRVValue *>
SPIRVEntry::getValues(const std::vector<SPIRVId> &IdVec) const {
  std::vector<SPIRVValue *> ValueVec;
  for (auto I : IdVec)
    ValueVec.push_back(getValue(I));
  return ValueVec;
}

std::vector<SPIRVType *>
SPIRVEntry::getValueTypes(const std::vector<SPIRVId> &IdVec) const {
  std::vector<SPIRVType *> TypeVec;
  for (auto I : IdVec)
    TypeVec.push_back(getValue(I)->getType());
  return TypeVec;
}

std::vector<SPIRVId>
SPIRVEntry::getIds(const std::vector<SPIRVValue *> ValueVec) const {
  std::vector<SPIRVId> IdVec;
  for (auto *I : ValueVec)
    IdVec.push_back(I->getId());
  return IdVec;
}

SPIRVEntry *SPIRVEntry::getEntry(SPIRVId TheId) const {
  return Module->getEntry(TheId);
}

void SPIRVEntry::validateFunctionControlMask(SPIRVWord TheFCtlMask) const {
  SPIRVCK(isValidFunctionControlMask(TheFCtlMask), InvalidFunctionControlMask,
          "");
}

void SPIRVEntry::validateValues(const std::vector<SPIRVId> &Ids) const {
  for (auto I : Ids)
    getValue(I)->validate();
}

void SPIRVEntry::validateBuiltin(SPIRVWord TheSet, SPIRVWord Index) const {
  assert(TheSet != SPIRVWORD_MAX && Index != SPIRVWORD_MAX &&
         "Invalid builtin");
}

void SPIRVEntry::addDecorate(SPIRVDecorate *Dec) {
  auto Kind = Dec->getDecorateKind();
  Decorates.insert(std::make_pair(Kind, Dec));
  Module->addDecorate(Dec);
  if (Kind == spv::DecorationLinkageAttributes) {
    auto *LinkageAttr = static_cast<const SPIRVDecorateLinkageAttr *>(Dec);
    setName(LinkageAttr->getLinkageName());
  }
  SPIRVDBG(spvdbgs() << "[addDecorate] Add "
                     << SPIRVDecorationNameMap::map(Kind) << " to Id " << Id
                     << '\n';)
}

void SPIRVEntry::addDecorate(SPIRVDecorateId *Dec) {
  auto Kind = Dec->getDecorateKind();
  DecorateIds.insert(std::make_pair(Kind, Dec));
  Module->addDecorate(Dec);
  SPIRVDBG(spvdbgs() << "[addDecorateId] Add"
                     << SPIRVDecorationNameMap::map(Kind) << " to Id " << Id
                     << '\n';)
}

void SPIRVEntry::addDecorate(Decoration Kind) {
  addDecorate(new SPIRVDecorate(Kind, this));
}

void SPIRVEntry::addDecorate(Decoration Kind, SPIRVWord Literal) {
  switch (static_cast<int>(Kind)) {
  case DecorationAliasScopeINTEL:
  case DecorationNoAliasINTEL:
    addDecorate(new SPIRVDecorateId(Kind, this, Literal));
    return;
  default:
    addDecorate(new SPIRVDecorate(Kind, this, Literal));
  }
}

void SPIRVEntry::eraseDecorate(Decoration Dec) { Decorates.erase(Dec); }

void SPIRVEntry::takeDecorates(SPIRVEntry *E) {
  Decorates = std::move(E->Decorates);
  SPIRVDBG(spvdbgs() << "[takeDecorates] " << Id << '\n';)
}

void SPIRVEntry::takeDecorateIds(SPIRVEntry *E) {
  DecorateIds = std::move(E->DecorateIds);
  SPIRVDBG(spvdbgs() << "[takeDecorateIds] " << Id << '\n';)
}

void SPIRVEntry::setLine(const std::shared_ptr<const SPIRVLine> &L) {
  Line = L;
  SPIRVDBG(if (L) spvdbgs() << "[setLine] " << *L << '\n';)
}

void SPIRVEntry::setDebugLine(const std::shared_ptr<const SPIRVExtInst> &DL) {
  DebugLine = DL;
  SPIRVDBG(if (DL) spvdbgs() << "[setDebugLine] " << *DL << '\n';)
}

void SPIRVEntry::addMemberDecorate(SPIRVMemberDecorate *Dec) {
  assert(canHaveMemberDecorates());
  MemberDecorates.insert(std::make_pair(
      std::make_pair(Dec->getMemberNumber(), Dec->getDecorateKind()), Dec));
  Module->addDecorate(Dec);
  SPIRVDBG(spvdbgs() << "[addMemberDecorate] " << *Dec << '\n';)
}

void SPIRVEntry::addMemberDecorate(SPIRVWord MemberNumber, Decoration Kind) {
  addMemberDecorate(new SPIRVMemberDecorate(Kind, MemberNumber, this));
}

void SPIRVEntry::addMemberDecorate(SPIRVWord MemberNumber, Decoration Kind,
                                   SPIRVWord Literal) {
  addMemberDecorate(new SPIRVMemberDecorate(Kind, MemberNumber, this, Literal));
}

void SPIRVEntry::eraseMemberDecorate(SPIRVWord MemberNumber, Decoration Dec) {
  MemberDecorates.erase(std::make_pair(MemberNumber, Dec));
}

void SPIRVEntry::takeMemberDecorates(SPIRVEntry *E) {
  MemberDecorates = std::move(E->MemberDecorates);
  SPIRVDBG(spvdbgs() << "[takeMemberDecorates] " << Id << '\n';)
}

void SPIRVEntry::takeAnnotations(SPIRVForward *E) {
  Module->setName(this, E->getName());
  takeDecorates(E);
  takeDecorateIds(E);
  takeMemberDecorates(E);
  if (OpCode == OpFunction)
    static_cast<SPIRVFunction *>(this)->takeExecutionModes(E);
}

void SPIRVEntry::replaceTargetIdInDecorates(SPIRVId Id) {
  for (auto It = Decorates.begin(), E = Decorates.end(); It != E; ++It)
    const_cast<SPIRVDecorate *>(It->second)->setTargetId(Id);
  for (auto It = DecorateIds.begin(), E = DecorateIds.end(); It != E; ++It)
    const_cast<SPIRVDecorateId *>(It->second)->setTargetId(Id);
  for (auto It = MemberDecorates.begin(), E = MemberDecorates.end(); It != E;
       ++It)
    const_cast<SPIRVMemberDecorate *>(It->second)->setTargetId(Id);
}

// Check if an entry has Kind of decoration and get the literal of the
// first decoration of such kind at Index.
bool SPIRVEntry::hasDecorate(Decoration Kind, size_t Index,
                             SPIRVWord *Result) const {
  auto Loc = Decorates.find(Kind);
  if (Loc == Decorates.end())
    return false;
  if (Result)
    *Result = Loc->second->getLiteral(Index);
  return true;
}

bool SPIRVEntry::hasDecorateId(Decoration Kind, size_t Index,
                               SPIRVId *Result) const {
  auto Loc = DecorateIds.find(Kind);
  if (Loc == DecorateIds.end())
    return false;
  if (Result)
    *Result = Loc->second->getLiteral(Index);
  return true;
}

// Check if an entry member has Kind of decoration and get the literal of the
// first decoration of such kind at Index.
bool SPIRVEntry::hasMemberDecorate(Decoration Kind, size_t Index,
                                   SPIRVWord MemberNumber,
                                   SPIRVWord *Result) const {
  auto Loc = MemberDecorates.find({MemberNumber, Kind});
  if (Loc == MemberDecorates.end())
    return false;
  if (Result)
    *Result = Loc->second->getLiteral(Index);
  return true;
}

std::vector<std::string>
SPIRVEntry::getDecorationStringLiteral(Decoration Kind) const {
  auto Loc = Decorates.find(Kind);
  if (Loc == Decorates.end())
    return {};

  return getVecString(Loc->second->getVecLiteral());
}

std::vector<std::string>
SPIRVEntry::getMemberDecorationStringLiteral(Decoration Kind,
                                             SPIRVWord MemberNumber) const {
  auto Loc = MemberDecorates.find({MemberNumber, Kind});
  if (Loc == MemberDecorates.end())
    return {};

  return getVecString(Loc->second->getVecLiteral());
}

std::vector<std::vector<std::string>>
SPIRVEntry::getAllDecorationStringLiterals(Decoration Kind) const {
  auto Loc = Decorates.find(Kind);
  if (Loc == Decorates.end())
    return {};

  std::vector<std::vector<std::string>> Literals;
  auto It = Decorates.equal_range(Kind);
  for (auto Itr = It.first; Itr != It.second; ++Itr)
    Literals.push_back(getVecString(Itr->second->getVecLiteral()));
  return Literals;
}

std::vector<std::vector<std::string>>
SPIRVEntry::getAllMemberDecorationStringLiterals(Decoration Kind,
                                                 SPIRVWord MemberNumber) const {
  auto Loc = MemberDecorates.find({MemberNumber, Kind});
  if (Loc == MemberDecorates.end())
    return {};

  std::vector<std::vector<std::string>> Literals;
  auto It = MemberDecorates.equal_range({MemberNumber, Kind});
  for (auto Itr = It.first; Itr != It.second; ++Itr)
    Literals.push_back(getVecString(Itr->second->getVecLiteral()));
  return Literals;
}

std::vector<SPIRVWord>
SPIRVEntry::getDecorationLiterals(Decoration Kind) const {
  auto Loc = Decorates.find(Kind);
  if (Loc == Decorates.end())
    return {};

  return (Loc->second->getVecLiteral());
}

std::vector<SPIRVId>
SPIRVEntry::getDecorationIdLiterals(Decoration Kind) const {
  auto Loc = DecorateIds.find(Kind);
  if (Loc == DecorateIds.end())
    return {};

  return (Loc->second->getVecLiteral());
}

std::vector<SPIRVWord>
SPIRVEntry::getMemberDecorationLiterals(Decoration Kind,
                                        SPIRVWord MemberNumber) const {
  auto Loc = MemberDecorates.find({MemberNumber, Kind});
  if (Loc == MemberDecorates.end())
    return {};

  return (Loc->second->getVecLiteral());
}

// Get literals of all decorations of Kind at Index.
std::set<SPIRVWord> SPIRVEntry::getDecorate(Decoration Kind,
                                            size_t Index) const {
  auto Range = Decorates.equal_range(Kind);
  std::set<SPIRVWord> Value;
  for (auto I = Range.first, E = Range.second; I != E; ++I) {
    assert(Index < I->second->getLiteralCount() && "Invalid index");
    Value.insert(I->second->getLiteral(Index));
  }
  return Value;
}

std::vector<SPIRVDecorate const *>
SPIRVEntry::getDecorations(Decoration Kind) const {
  auto Range = Decorates.equal_range(Kind);
  std::vector<SPIRVDecorate const *> Decors;
  Decors.reserve(Decorates.count(Kind));
  for (auto I = Range.first, E = Range.second; I != E; ++I) {
    Decors.push_back(I->second);
  }
  return Decors;
}

std::vector<SPIRVDecorate const *> SPIRVEntry::getDecorations() const {
  std::vector<SPIRVDecorate const *> Decors;
  Decors.reserve(Decorates.size());
  for (auto &DecoPair : Decorates)
    Decors.push_back(DecoPair.second);
  return Decors;
}

std::set<SPIRVId> SPIRVEntry::getDecorateId(Decoration Kind,
                                            size_t Index) const {
  auto Range = DecorateIds.equal_range(Kind);
  std::set<SPIRVId> Value;
  for (auto I = Range.first, E = Range.second; I != E; ++I) {
    assert(Index < I->second->getLiteralCount() && "Invalid index");
    Value.insert(I->second->getLiteral(Index));
  }
  return Value;
}

std::vector<SPIRVDecorateId const *>
SPIRVEntry::getDecorationIds(Decoration Kind) const {
  auto Range = DecorateIds.equal_range(Kind);
  std::vector<SPIRVDecorateId const *> Decors;
  Decors.reserve(DecorateIds.count(Kind));
  for (auto I = Range.first, E = Range.second; I != E; ++I) {
    Decors.push_back(I->second);
  }
  return Decors;
}

bool SPIRVEntry::hasLinkageType() const {
  return OpCode == OpFunction || OpCode == OpVariable;
}

bool SPIRVEntry::isExtInst(const SPIRVExtInstSetKind InstSet) const {
  if (isExtInst()) {
    const SPIRVExtInst *EI = static_cast<const SPIRVExtInst *>(this);
    return EI->getExtSetKind() == InstSet;
  }
  return false;
}

bool SPIRVEntry::isExtInst(const SPIRVExtInstSetKind InstSet,
                           const SPIRVWord ExtOp) const {
  if (isExtInst()) {
    const SPIRVExtInst *EI = static_cast<const SPIRVExtInst *>(this);
    if (EI->getExtSetKind() == InstSet) {
      return EI->getExtOp() == ExtOp;
    }
  }
  return false;
}

void SPIRVEntry::encodeDecorate(spv_ostream &O) const {
  for (auto &I : Decorates)
    O << *I.second;
  for (auto &I : DecorateIds)
    O << *I.second;
}

SPIRVLinkageTypeKind SPIRVEntry::getLinkageType() const {
  assert(hasLinkageType());
  DecorateMapType::const_iterator Loc =
      Decorates.find(DecorationLinkageAttributes);
  if (Loc == Decorates.end())
    return internal::LinkageTypeInternal;
  return static_cast<const SPIRVDecorateLinkageAttr *>(Loc->second)
      ->getLinkageType();
}

void SPIRVEntry::setLinkageType(SPIRVLinkageTypeKind LT) {
  assert(isValid(LT));
  assert(hasLinkageType());
  addDecorate(new SPIRVDecorateLinkageAttr(this, Name, LT));
}

void SPIRVEntry::updateModuleVersion() const {
  if (!Module)
    return;

  Module->setMinSPIRVVersion(getRequiredSPIRVVersion());
}

spv_ostream &operator<<(spv_ostream &O, const SPIRVEntry &E) {
  E.validate();
  E.encodeAll(O);
  O << SPIRVNL();
  return O;
}

std::istream &operator>>(std::istream &I, SPIRVEntry &E) {
  E.decode(I);
  return I;
}

SPIRVEntryPoint::SPIRVEntryPoint(SPIRVModule *TheModule,
                                 SPIRVExecutionModelKind TheExecModel,
                                 SPIRVId TheId, const std::string &TheName,
                                 std::vector<SPIRVId> Variables)
    : SPIRVAnnotation(OpEntryPoint, TheModule->get<SPIRVFunction>(TheId),
                      getSizeInWords(TheName) + Variables.size() + 3),
      ExecModel(TheExecModel), Name(TheName), Variables(Variables) {}

void SPIRVEntryPoint::encode(spv_ostream &O) const {
  getEncoder(O) << ExecModel << Target << Name << Variables;
}

void SPIRVEntryPoint::decode(std::istream &I) {
  getDecoder(I) >> ExecModel >> Target >> Name;
  Variables.resize(WordCount - FixedWC - getSizeInWords(Name) + 1);
  getDecoder(I) >> Variables;
  Module->setName(getOrCreateTarget(), Name);
  Module->addEntryPoint(ExecModel, Target, Name, Variables);
}

void SPIRVExecutionMode::encode(spv_ostream &O) const {
  getEncoder(O) << Target << ExecMode << WordLiterals;
}

void SPIRVExecutionMode::decode(std::istream &I) {
  getDecoder(I) >> Target >> ExecMode;
  switch (static_cast<uint32_t>(ExecMode)) {
  case ExecutionModeLocalSize:
  case ExecutionModeLocalSizeHint:
  case ExecutionModeMaxWorkgroupSizeINTEL:
    WordLiterals.resize(3);
    break;
  case ExecutionModeInvocations:
  case ExecutionModeOutputVertices:
  case ExecutionModeVecTypeHint:
  case ExecutionModeDenormPreserve:
  case ExecutionModeDenormFlushToZero:
  case ExecutionModeSignedZeroInfNanPreserve:
  case ExecutionModeRoundingModeRTE:
  case ExecutionModeRoundingModeRTZ:
  case ExecutionModeRoundingModeRTPINTEL:
  case ExecutionModeRoundingModeRTNINTEL:
  case ExecutionModeFloatingPointModeALTINTEL:
  case ExecutionModeFloatingPointModeIEEEINTEL:
  case ExecutionModeSharedLocalMemorySizeINTEL:
  case ExecutionModeNamedBarrierCountINTEL:
  case ExecutionModeSubgroupSize:
  case ExecutionModeMaxWorkDimINTEL:
  case ExecutionModeNumSIMDWorkitemsINTEL:
  case ExecutionModeSchedulerTargetFmaxMhzINTEL:
  case ExecutionModeRegisterMapInterfaceINTEL:
  case ExecutionModeStreamingInterfaceINTEL:
  case spv::internal::ExecutionModeNamedSubgroupSizeINTEL:
  case ExecutionModeMaximumRegistersINTEL:
  case ExecutionModeMaximumRegistersIdINTEL:
  case ExecutionModeNamedMaximumRegistersINTEL:
    WordLiterals.resize(1);
    break;
  default:
    // Do nothing. Keep this to avoid VS2013 warning.
    break;
  }
  getDecoder(I) >> WordLiterals;
  getOrCreateTarget()->addExecutionMode(Module->add(this));
}

SPIRVForward *SPIRVAnnotationGeneric::getOrCreateTarget() const {
  SPIRVEntry *Entry = nullptr;
  bool Found = Module->exist(Target, &Entry);
  assert((!Found || Entry->getOpCode() == internal::OpForward) &&
         "Annotations only allowed on forward");
  if (!Found)
    Entry = Module->addForward(Target, nullptr);
  return static_cast<SPIRVForward *>(Entry);
}

SPIRVName::SPIRVName(const SPIRVEntry *TheTarget, const std::string &TheStr)
    : SPIRVAnnotation(OpName, TheTarget, getSizeInWords(TheStr) + 2),
      Str(TheStr) {}

void SPIRVName::encode(spv_ostream &O) const { getEncoder(O) << Target << Str; }

void SPIRVName::decode(std::istream &I) {
  getDecoder(I) >> Target >> Str;
  Module->setName(getOrCreateTarget(), Str);
}

void SPIRVName::validate() const {
  assert(WordCount == getSizeInWords(Str) + 2 && "Incorrect word count");
}

_SPIRV_IMP_ENCDEC2(SPIRVString, Id, Str)
_SPIRV_IMP_ENCDEC3(SPIRVMemberName, Target, MemberNumber, Str)

void SPIRVLine::encode(spv_ostream &O) const {
  getEncoder(O) << FileName << Line << Column;
}

void SPIRVLine::decode(std::istream &I) {
  getDecoder(I) >> FileName >> Line >> Column;
}

void SPIRVLine::validate() const {
  assert(OpCode == OpLine);
  assert(WordCount == 4);
  assert(get<SPIRVEntry>(FileName)->getOpCode() == OpString);
  assert(Line != SPIRVWORD_MAX);
  assert(Column != SPIRVWORD_MAX);
  assert(!hasId());
}

void SPIRVMemberName::validate() const {
  assert(OpCode == OpMemberName);
  assert(WordCount == getSizeInWords(Str) + FixedWC);
  assert(get<SPIRVEntry>(Target)->getOpCode() == OpTypeStruct);
  assert(MemberNumber < get<SPIRVTypeStruct>(Target)->getStructMemberCount());
}

SPIRVExtInstImport::SPIRVExtInstImport(SPIRVModule *TheModule, SPIRVId TheId,
                                       const std::string &TheStr)
    : SPIRVEntry(TheModule, 2 + getSizeInWords(TheStr), OC, TheId),
      Str(TheStr) {
  validate();
}

void SPIRVExtInstImport::encode(spv_ostream &O) const {
  getEncoder(O) << Id << Str;
}

void SPIRVExtInstImport::decode(std::istream &I) {
  getDecoder(I) >> Id >> Str;
  Module->importBuiltinSetWithId(Str, Id);
}

void SPIRVExtInstImport::validate() const {
  SPIRVEntry::validate();
  assert(!Str.empty() && "Invalid builtin set");
}

void SPIRVMemoryModel::encode(spv_ostream &O) const {
  getEncoder(O) << Module->getAddressingModel() << Module->getMemoryModel();
}

void SPIRVMemoryModel::decode(std::istream &I) {
  SPIRVAddressingModelKind AddrModel;
  SPIRVMemoryModelKind MemModel;
  getDecoder(I) >> AddrModel >> MemModel;
  Module->setAddressingModel(AddrModel);
  Module->setMemoryModel(MemModel);
}

void SPIRVMemoryModel::validate() const {
  auto AM = Module->getAddressingModel();
  auto MM = Module->getMemoryModel();
  SPIRVCK(isValid(AM), InvalidAddressingModel,
          "Actual is " + std::to_string(AM));
  SPIRVCK(isValid(MM), InvalidMemoryModel, "Actual is " + std::to_string(MM));
}

void SPIRVSource::encode(spv_ostream &O) const {
  SPIRVWord Ver = SPIRVWORD_MAX;
  auto Language = Module->getSourceLanguage(&Ver);
  getEncoder(O) << Language << Ver;
}

void SPIRVSource::decode(std::istream &I) {
  SourceLanguage Lang = SourceLanguageUnknown;
  SPIRVWord Ver = SPIRVWORD_MAX;
  getDecoder(I) >> Lang >> Ver;
  Module->setSourceLanguage(Lang, Ver);
}

SPIRVSourceExtension::SPIRVSourceExtension(SPIRVModule *M,
                                           const std::string &SS)
    : SPIRVEntryNoId(M, 1 + getSizeInWords(SS)), S(SS) {}

void SPIRVSourceExtension::encode(spv_ostream &O) const { getEncoder(O) << S; }

void SPIRVSourceExtension::decode(std::istream &I) {
  getDecoder(I) >> S;
  Module->getSourceExtension().insert(S);
}

SPIRVExtension::SPIRVExtension(SPIRVModule *M, const std::string &SS)
    : SPIRVEntryNoId(M, 1 + getSizeInWords(SS)), S(SS) {}

void SPIRVExtension::encode(spv_ostream &O) const { getEncoder(O) << S; }

void SPIRVExtension::decode(std::istream &I) {
  getDecoder(I) >> S;
  Module->getExtension().insert(S);
}

SPIRVCapability::SPIRVCapability(SPIRVModule *M, SPIRVCapabilityKind K)
    : SPIRVEntryNoId(M, 2), Kind(K) {
  updateModuleVersion();
}

void SPIRVCapability::encode(spv_ostream &O) const { getEncoder(O) << Kind; }

void SPIRVCapability::decode(std::istream &I) {
  getDecoder(I) >> Kind;
  Module->addCapability(Kind);
}

template <spv::Op OC> void SPIRVContinuedInstINTELBase<OC>::validate() const {
  SPIRVEntry::validate();
}

template <spv::Op OC>
void SPIRVContinuedInstINTELBase<OC>::encode(spv_ostream &O) const {
  SPIRVEntry::getEncoder(O) << (Elements);
}
template <spv::Op OC>
void SPIRVContinuedInstINTELBase<OC>::decode(std::istream &I) {
  SPIRVEntry::getDecoder(I) >> (Elements);
}

SPIRVType *SPIRVTypeStructContinuedINTEL::getMemberType(size_t I) const {
  return static_cast<SPIRVType *>(SPIRVEntry::getEntry(Elements[I]));
}

void SPIRVModuleProcessed::validate() const {
  assert(WordCount == FixedWC + getSizeInWords(ProcessStr) &&
         "Incorrect word count in OpModuleProcessed");
}

void SPIRVModuleProcessed::encode(spv_ostream &O) const {
  getEncoder(O) << ProcessStr;
}

void SPIRVModuleProcessed::decode(std::istream &I) {
  getDecoder(I) >> ProcessStr;
  Module->addModuleProcessed(ProcessStr);
}

std::string SPIRVModuleProcessed::getProcessStr() { return ProcessStr; }

} // namespace SPIRV
