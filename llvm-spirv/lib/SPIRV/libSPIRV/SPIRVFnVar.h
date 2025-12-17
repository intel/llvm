//===- SPIRVFnVar.cpp                                                     -===//
//
//                     The LLVM/SPIRV Translator
//
// Copyright (c) 2025 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines entries from the SPV_INTEL_function_variants extension. It
/// also provides a function for specializing a multi-target module into a
/// targeted one.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVFNVAR_H
#define SPIRV_LIBSPIRV_SPIRVFNVAR_H

#include "SPIRVEntry.h"
#include "SPIRVInstruction.h"
#include "SPIRVValue.h"

#if _SPIRVDBG
#include <iomanip>
#endif

namespace SPIRV {

// Specialize multitarget module into a targeted one according to
// SPV_INTEL_function_variants.
//
// This is a SPIR-V to SPIR-V transform. After the transform, the module should
// not contain any notion of SPV_INTEL_function_variants and can be processed
// even by consumers that do not support this extension.
bool specializeFnVariants(SPIRVModule *BM, std::string &ErrMsg);

// Below are entries defined by the extension:

class SPIRVConditionalEntryPointINTEL : public SPIRVAnnotation {
public:
  static const SPIRVWord FixedWC = 5;
  SPIRVConditionalEntryPointINTEL(SPIRVModule *TheModule, SPIRVId Condition,
                                  SPIRVExecutionModelKind TheExecModel,
                                  SPIRVId TheId, const std::string &TheName,
                                  std::vector<SPIRVId> Variables)
      : SPIRVAnnotation(OpConditionalEntryPointINTEL,
                        TheModule->get<SPIRVFunction>(TheId),
                        getSizeInWords(TheName) + Variables.size() + 4),
        Condition(Condition), ExecModel(TheExecModel), Name(TheName),
        Variables(std::move(Variables)) {}
  SPIRVConditionalEntryPointINTEL()
      : SPIRVAnnotation(OpConditionalEntryPointINTEL), Condition() {}

  SPIRVId getCondition() const { return Condition; }
  SPIRVExecutionModelKind getExecModel() const { return ExecModel; }
  std::string getName() const { return Name; }
  std::vector<SPIRVId> getVariables() const { return Variables; }

protected:
  void encode(spv_ostream &O) const override {
    getEncoder(O) << Condition << ExecModel << Target << Name << Variables;
  }

  void decode(std::istream &I) override {
    getDecoder(I) >> Condition >> ExecModel >> Target >> Name;
    Variables.resize(WordCount - FixedWC - getSizeInWords(Name) + 1);
    getDecoder(I) >> Variables;
    Module->setName(getOrCreateTarget(), Name);
    Module->addConditionalEntryPoint(Condition, ExecModel, Target, Name,
                                     Variables);
  }

private:
  SPIRVId Condition;
  SPIRVExecutionModelKind ExecModel = ExecutionModelMax;
  std::string Name;
  std::vector<SPIRVId> Variables;
};

class SPIRVConditionalExtensionINTEL
    : public SPIRVEntryNoId<OpConditionalExtensionINTEL> {
public:
  SPIRVConditionalExtensionINTEL(SPIRVModule *M, SPIRVId C,
                                 const std::string &SS)
      : SPIRVEntryNoId(M, 2 + getSizeInWords(SS)), Condition(C), S(SS) {}
  SPIRVConditionalExtensionINTEL() : Condition() {}

  std::string getExtensionName() const { return S; }
  SPIRVId getCondition() const { return Condition; }

protected:
  void encode(spv_ostream &O) const override {
    getEncoder(O) << Condition << S;
  }

  void decode(std::istream &I) override {
    getDecoder(I) >> Condition >> S;
    Module->getConditionalExtensions().insert(std::make_pair(Condition, S));
  }

private:
  SPIRVId Condition;
  std::string S;
};

class SPIRVConditionalCapabilityINTEL
    : public SPIRVEntryNoId<OpConditionalCapabilityINTEL> {
public:
  static const SPIRVWord FixedWC = 3;
  SPIRVConditionalCapabilityINTEL(SPIRVModule *M, SPIRVId C,
                                  SPIRVCapabilityKind K)
      : SPIRVEntryNoId(M, 3), Condition(C), Kind(K) {
    updateModuleVersion();
  }
  SPIRVConditionalCapabilityINTEL() : Condition(), Kind() {}

  SPIRVId getCondition() const { return Condition; }

protected:
  void encode(spv_ostream &O) const override {
    getEncoder(O) << Condition << Kind;
  }

  void decode(std::istream &I) override {
    getDecoder(I) >> Condition >> Kind;
    Module->addConditionalCapability(Condition, Kind);
  }

private:
  SPIRVId Condition;
  SPIRVCapabilityKind Kind;
};

class SPIRVConditionalCopyObjectINTEL : public SPIRVInstruction {
public:
  const static Op OC = OpConditionalCopyObjectINTEL;
  const static SPIRVWord FixedWordCount = 3;

  // Complete constructor
  SPIRVConditionalCopyObjectINTEL(SPIRVType *TheType, SPIRVId TheId,
                                  const std::vector<SPIRVId> &TheConstituents,
                                  SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(FixedWordCount + TheConstituents.size(), OC, TheType,
                         TheId, TheBB),
        Constituents(TheConstituents) {
    validate();
    assert(TheBB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVConditionalCopyObjectINTEL() : SPIRVInstruction(OC) {}

  std::vector<SPIRVId> getOperandIds() { return Constituents; }

  std::vector<SPIRVValue *> getOperands() override {
    return getValues(Constituents);
  }

protected:
  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    Constituents.resize(TheWordCount - FixedWordCount);
  }
  _SPIRV_DEF_ENCDEC3(Type, Id, Constituents)
  void validate() const override {
    SPIRVInstruction::validate();
    size_t TypeOpCode = this->getType()->getOpCode();
    assert(TypeOpCode != OpTypeVoid && "Conditional copy type cannot be void");
    (void)(TypeOpCode);
    assert(Constituents.size() % 2 == 0 &&
           "Conditional copy requires condition-operand pairs");
    assert(Constituents.size() >= 2 &&
           "Conditional copy requires at least one condition-operand pair");
  }
  std::vector<SPIRVId> Constituents;
};

class SPIRVSpecConstantTargetINTEL : public SPIRVValue {
public:
  constexpr static SPIRVWord FixedWC = 4;
  constexpr static spv::Op OC = OpSpecConstantTargetINTEL;

  // Complete constructor
  SPIRVSpecConstantTargetINTEL(SPIRVModule *M, SPIRVType *TheType,
                               SPIRVId TheId, SPIRVWord TheTarget,
                               const std::vector<SPIRVWord> TheFeatures)
      : SPIRVValue(M, TheFeatures.size() + FixedWC, OC, TheType, TheId) {
    Features = TheFeatures;
    Target = TheTarget;
    NumWords = TheFeatures.size() + FixedWC;
    validate();
  }
  // Incomplete constructor
  SPIRVSpecConstantTargetINTEL() : SPIRVValue(OC), NumWords(), Target() {}

  SPIRVWord getTarget() const { return Target; }
  bool matchesDevice() {
    std::optional<SPIRVWord> DeviceTarget = getModule()->getFnVarTarget();
    std::vector<SPIRVWord> DeviceFeatures = getModule()->getFnVarFeatures();
    bool Res = true;
    if (DeviceTarget != std::nullopt && DeviceTarget.value() != Target) {
      Res = false;
    }
    if (!DeviceFeatures.empty()) {
      for (const auto &Feature : Features) {
        if (std::find(DeviceFeatures.cbegin(), DeviceFeatures.cend(),
                      Feature) == DeviceFeatures.cend()) {
          Res = false;
        }
      }
    }

    SPIRVDBG(
        spvdbgs() << "[FnVar] match instr  Target: ";
        spvdbgs() << std::setw(4) << Target;
        spvdbgs() << ", Features:"; if (Features.empty()) {
          spvdbgs() << " none";
        } else {
          for (const auto &Feat : Features) {
            spvdbgs() << " " << Feat;
          }
        } spvdbgs() << " | ID: %"
                    << getId() << std::endl;
        spvdbgs() << "[FnVar]       device Target: "; if (DeviceTarget ==
                                                          std::nullopt) {
          spvdbgs() << "none";
        } else {
          spvdbgs() << std::setw(4) << DeviceTarget.value();
        } spvdbgs() << ", Features:";
        for (const auto &Feat : DeviceFeatures) {
          spvdbgs() << " " << Feat;
        } spvdbgs()
        << std::endl;
        spvdbgs() << "[FnVar]       result: " << Res << std::endl;);

    return Res;
  }

protected:
  _SPIRV_DEF_ENCDEC4(Type, Id, Target, Features);
  void setWordCount(SPIRVWord WordCount) override {
    SPIRVEntry::setWordCount(WordCount);
    Features.resize(WordCount - FixedWC);
    NumWords = WordCount - FixedWC;
  }

private:
  unsigned NumWords;
  std::vector<SPIRVWord> Features;
  SPIRVWord Target;
};

class SPIRVSpecConstantArchitectureINTEL : public SPIRVValue {
public:
  constexpr static SPIRVWord FixedWC = 7;
  constexpr static spv::Op OC = OpSpecConstantArchitectureINTEL;

  // Complete constructor
  SPIRVSpecConstantArchitectureINTEL(SPIRVModule *M, SPIRVType *TheType,
                                     SPIRVId TheId, SPIRVWord TheCategory,
                                     SPIRVWord TheFamily, spv::Op TheCmpOp,
                                     SPIRVWord TheArchitecture)
      : SPIRVValue(M, FixedWC, OC, TheType, TheId) {
    Category = TheCategory;
    Family = TheFamily;
    CmpOp = TheCmpOp;
    Architecture = TheArchitecture;
    validate();
  }
  // Incomplete constructor
  SPIRVSpecConstantArchitectureINTEL()
      : SPIRVValue(OC), Category(), Family(), CmpOp(), Architecture() {}

  SPIRVWord getCategory() { return Category; }
  SPIRVWord getFamily() { return Family; }
  spv::Op getCmpOp() { return CmpOp; }
  SPIRVWord getArchitecture() { return Architecture; }
  bool matchesDevice() {
    std::optional<SPIRVWord> DeviceCategory = getModule()->getFnVarCategory();
    std::optional<SPIRVWord> DeviceFamily = getModule()->getFnVarFamily();
    std::optional<SPIRVWord> DeviceArchitecture = getModule()->getFnVarArch();
    bool Res = true;

    if (DeviceCategory != std::nullopt && DeviceCategory.value() != Category) {
      Res = false;
    }
    if (DeviceFamily != std::nullopt && DeviceFamily.value() != Family) {
      Res = false;
    }
    if (DeviceArchitecture != std::nullopt) {
      switch (CmpOp) {
      case OpIEqual:
        Res = DeviceArchitecture == Architecture;
        break;
      case OpINotEqual:
        Res = DeviceArchitecture != Architecture;
        break;
      case OpULessThan:
        Res = DeviceArchitecture < Architecture;
        break;
      case OpULessThanEqual:
        Res = DeviceArchitecture <= Architecture;
        break;
      case OpUGreaterThan:
        Res = DeviceArchitecture > Architecture;
        break;
      case OpUGreaterThanEqual:
        Res = DeviceArchitecture >= Architecture;
        break;
      default:
        assert(false && "Invalid checked CmpOp");
        Res = false;
        break;
      }
    }

    SPIRVDBG(
        spvdbgs() << "[FnVar] match instr  Category: " << std::setw(4)
                  << Category

                  << ", Family: " << std::setw(4) << Family << ", CmpOp: "
                  << std::setw(4) << CmpOp << ", Architecture: " << std::setw(4)
                  << Architecture << " | ID: %" << getId() << std::endl;
        spvdbgs() << "[FnVar]       device Category: "; if (DeviceCategory ==
                                                            std::nullopt) {
          spvdbgs() << "none";
        } else {
          spvdbgs() << std::setw(4) << DeviceCategory.value();
        } spvdbgs() << ", Family: ";
        if (DeviceFamily == std::nullopt) { spvdbgs() << "none"; } else {
          spvdbgs() << std::setw(4) << DeviceFamily.value();
        } spvdbgs()
        << ",              Architecture: ";
        if (DeviceArchitecture == std::nullopt) { spvdbgs() << "none"; } else {
          spvdbgs() << std::setw(4) << DeviceArchitecture.value();
        } spvdbgs()
        << std::endl;
        spvdbgs() << "[FnVar]       result: " << Res << std::endl;);

    return Res;
  }

protected:
  _SPIRV_DEF_ENCDEC6(Type, Id, Category, Family, CmpOp, Architecture);

private:
  SPIRVWord Category;
  SPIRVWord Family;
  spv::Op CmpOp;
  SPIRVWord Architecture;
};

class SPIRVSpecConstantCapabilitiesINTEL : public SPIRVValue {
public:
  constexpr static SPIRVWord FixedWC = 3;
  constexpr static spv::Op OC = OpSpecConstantCapabilitiesINTEL;

  // Complete constructor
  SPIRVSpecConstantCapabilitiesINTEL(
      SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId,
      const std::vector<SPIRVWord> TheCapabilities)
      : SPIRVValue(M, TheCapabilities.size() + FixedWC, OC, TheType, TheId) {
    Capabilities = TheCapabilities;
    NumWords = TheCapabilities.size() + FixedWC;
    validate();
  }
  // Incomplete constructor
  SPIRVSpecConstantCapabilitiesINTEL() : SPIRVValue(OC), NumWords() {}

  std::vector<SPIRVWord> getCapabilities() const { return Capabilities; }
  bool matchesDevice() {
    std::vector<SPIRVWord> DeviceCapabilities =
        getModule()->getFnVarCapabilities();
    bool Res = true;

    if (!DeviceCapabilities.empty()) {
      for (const auto &Capability : Capabilities) {
        if (std::find(DeviceCapabilities.cbegin(), DeviceCapabilities.cend(),
                      Capability) == DeviceCapabilities.cend()) {
          Res = false;
        }
      }
    }

    SPIRVDBG(
        spvdbgs() << "[FnVar] match instr  Capabilities: ";
        if (Capabilities.empty()) { spvdbgs() << "none"; } else {
          for (const auto &Cap : Capabilities) {
            spvdbgs() << " " << Cap;
          }
        } spvdbgs()
        << " | ID: %" << getId() << std::endl;
        spvdbgs() << "[FnVar]       device Capabilities: ";
        for (const auto &Cap : DeviceCapabilities) {
          spvdbgs() << " " << Cap;
        } spvdbgs()
        << std::endl;
        spvdbgs() << "[FnVar]       result: " << Res << std::endl;);

    return Res;
  }

protected:
  _SPIRV_DEF_ENCDEC3(Type, Id, Capabilities);

  void validate() const override {
    SPIRVValue::validate();
    if (Capabilities.size() < 1) {
      std::stringstream SS;
      SS << "Id: " << Id << ", OpCode: " << OpCodeNameMap::map(OpCode)
         << ", Name: \"" << Name << "\". Expected at least one capability.\n";
      getErrorLog().checkError(false, SPIRVEC_InvalidNumberOfOperands,
                               SS.str());
    }
  }

  void setWordCount(SPIRVWord WordCount) override {
    SPIRVEntry::setWordCount(WordCount);
    Capabilities.resize(WordCount - FixedWC);
    NumWords = WordCount - FixedWC;
  }

private:
  unsigned NumWords;
  std::vector<SPIRVWord> Capabilities;
};

} // namespace SPIRV
#endif // SPIRV_LIBSPIRV_SPIRVFNVAR_H
