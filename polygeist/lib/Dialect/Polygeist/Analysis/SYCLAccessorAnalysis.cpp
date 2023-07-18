//===- SYCLAccessorAnalysis.h - Analysis for sycl::buffer
//-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Analysis/SYCLAccessorAnalysis.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"

#define DEBUG_TYPE "sycl-accessor-analysis"

namespace mlir {
namespace polygeist {

//===----------------------------------------------------------------------===//
// AccessorInformation
//===----------------------------------------------------------------------===//

raw_ostream &operator<<(raw_ostream &os, const AccessorInformation &info) {

  os.indent(4) << "Needs range: " << ((info.needsRange()) ? "Yes" : "No")
               << "\n";
  os.indent(4) << "Range: ";
  if (info.hasConstantRange()) {
    os << "range{";
    llvm::interleaveComma(info.getConstantRange(), os);
    os << "}\n";
  } else {
    os << "<unknown>\n";
  }

  os.indent(4) << "Needs offset: " << ((info.needsOffset()) ? "Yes" : "No")
               << "\n";
  os.indent(4) << "Offset: ";
  if (info.hasConstantOffset()) {
    os << "id{";
    llvm::interleaveComma(info.getConstantOffset(), os);
    os << "}\n";
  } else {
    os << "<unknown>\n";
  }

  os.indent(4) << "Buffer: ";
  if (info.hasKnownBuffer())
    os << info.buffer;
  else
    os << "<unknown>";
  os << "\n";

  os.indent(4) << "Buffer information: ";
  if (info.hasBufferInformation())
    os << info.getBufferInfo();
  else
    os << "<unknown>";
  os << "\n";

  return os;
}

const AccessorInformation
AccessorInformation::join(const AccessorInformation &other,
                          AliasAnalysis &aliasAnalysis) const {
  Value jointBuf = nullptr;
  if (hasKnownBuffer() && other.hasKnownBuffer() &&
      (getBuffer() == other.getBuffer() ||
       aliasAnalysis.alias(getBuffer(), other.getBuffer()) ==
           AliasResult::MustAlias))
    jointBuf = getBuffer();

  std::optional<BufferInformation> jointBufInfo = std::nullopt;
  if (hasBufferInformation() && other.hasBufferInformation())
    jointBufInfo = getBufferInfo().join(other.getBufferInfo(), aliasAnalysis);

  // The range can only omitted if this and the incoming agree info agree it is
  // not needed, otherwise the conservative default is to assume the range is
  // needed.
  bool jointNeedsRange =
      (needsRange() == other.needsRange()) ? needsRange() : true;
  SmallVector<size_t, 3> jointRange = (constantRange == other.constantRange)
                                          ? constantRange
                                          : SmallVector<size_t, 3>{};

  // The offset can only omitted if this and the incoming agree info agree it is
  // not needed, otherwise the conservative default is to assume the offset is
  // needed.
  bool jointNeedsOffset =
      (needsOffset() == other.needsOffset()) ? needsOffset() : true;
  SmallVector<size_t, 3> jointOffset = (constantOffset == other.constantOffset)
                                           ? constantOffset
                                           : SmallVector<size_t, 3>{};

  return AccessorInformation(jointBuf, jointBufInfo, jointNeedsRange,
                             jointRange, jointNeedsOffset, jointOffset);
}

//===----------------------------------------------------------------------===//
// SYCLAccessorAnalysis
//===----------------------------------------------------------------------===//

SYCLAccessorAnalysis::SYCLAccessorAnalysis(Operation *op, AnalysisManager &mgr)
    : operation(op), am(mgr), aliasAnalysis(nullptr), idRangeAnalysis(op, mgr),
      bufferAnalysis(op, mgr) {}

SYCLAccessorAnalysis &
SYCLAccessorAnalysis::initialize(bool useRelaxedAliasing) {

  // Initialize the dataflow solver
  aliasAnalysis = &am.getAnalysis<mlir::AliasAnalysis>();
  aliasAnalysis->addAnalysisImplementation(
      sycl::AliasAnalysis(useRelaxedAliasing));

  solver = std::make_unique<DataFlowSolverWrapper>(*aliasAnalysis);

  // Populate the solver and run the analyses needed by this analysis.
  solver->loadWithRequiredAnalysis<ReachingDefinitionAnalysis>(*aliasAnalysis);

  if (failed(solver->initializeAndRun(operation))) {
    operation->emitError("Failed to run required dataflow analyses");
    return *this;
  }

  idRangeAnalysis.initialize(useRelaxedAliasing);
  bufferAnalysis.initialize(useRelaxedAliasing);

  initialized = true;

  return *this;
}

std::optional<AccessorInformation>
SYCLAccessorAnalysis::getAccessorInformationFromConstruction(Operation *op,
                                                             Value operand) {
  assert(initialized &&
         "Analysis only available after successful initialization");
  assert(isa<LLVM::LLVMPointerType>(operand.getType()) &&
         "Expecting an LLVM pointer");
  assert(aliasAnalysis != nullptr && "Alias analysis not initialized");

  const polygeist::ReachingDefinition *reachingDef =
      solver->lookupState<polygeist::ReachingDefinition>(op);
  assert(reachingDef && "expected a reaching definition");

  auto mods = reachingDef->getModifiers(operand, *solver);
  if (!mods || mods->empty())
    return std::nullopt;

  if (!llvm::all_of(*mods,
                    [&](const Definition &def) { return isConstructor(def); }))
    return std::nullopt;

  auto pMods = reachingDef->getPotentialModifiers(operand, *solver);
  if (pMods) {
    if (!llvm::all_of(
            *pMods, [&](const Definition &def) { return isConstructor(def); }))
      return std::nullopt;
  }

  bool first = true;
  AccessorInformation info;
  for (const Definition &def : *mods) {
    if (first) {
      info = getInformation(def);
      first = false;
    } else {
      info = info.join(getInformation(def), *aliasAnalysis);
      if (false /* TODO */)
        // Early return: As soon as joining of the different information has led
        // to an info with no fixed size and no definitive sub-buffer
        // information, we can end the processing.
        return info;
    }
  }

  if (pMods) {
    for (const Definition &def : *pMods) {
      info = info.join(getInformation(def), *aliasAnalysis);
      if (false /* TODO */)
        // Early return: As soon as joining of the different information has led
        // to an info with no fixed size and no definitive sub-buffer
        // information, we can end the processing.
        return info;
    }
  }

  return info;
}

bool SYCLAccessorAnalysis::isConstructor(const Definition &def) {
  if (!def.isOperation())
    return false;

  auto constructor = dyn_cast<sycl::SYCLHostConstructorOp>(def.getOperation());
  if (!constructor)
    return false;

  return isa<sycl::AccessorType>(constructor.getType().getValue());
}

AccessorInformation
SYCLAccessorAnalysis::getInformation(const Definition &def) {
  assert(def.isOperation() && "Expecting operation");

  auto constructor = cast<sycl::SYCLHostConstructorOp>(def.getOperation());

  Value buffer = nullptr;
  std::optional<BufferInformation> bufferInfo = std::nullopt;
  if (constructor->getNumOperands() > 2) {
    buffer = constructor->getOperand(1);
    bufferInfo = bufferAnalysis.getBufferInformationFromConstruction(
        constructor, constructor->getOperand(1));
  }

  auto accessorTy = cast<sycl::AccessorType>(constructor.getType().getValue());
  // Whether the range and offset parameter are required by the accessor
  // constructed is encoded in the list of body types.
  bool needsRange = llvm::any_of(accessorTy.getBody(), [](const Type &ty) {
    return isa<sycl::RangeType>(ty);
  });
  bool needsOffset = llvm::any_of(accessorTy.getBody(), [](const Type &ty) {
    return isa<sycl::IDType>(ty);
  });
  SmallVector<size_t, 3> constRange;
  SmallVector<size_t, 3> constOffset;
  if (needsRange) {
    // In the C++ API, the range can either be the second or third parameter to
    // the constructor.
    std::optional<IDRangeInformation> optRangeInfo =
        getOperandInfo<sycl::RangeType>(constructor, 2, 3);
    if (optRangeInfo.has_value() && optRangeInfo->isConstant())
      constRange = optRangeInfo->getConstantValues();
  }
  if (needsOffset) {
    // In the C++ API, the range can either be the third or fourth parameter to
    // the constructor.
    std::optional<IDRangeInformation> optIDInfo =
        getOperandInfo<sycl::IDType>(constructor, 3, 4);
    if (optIDInfo.has_value() && optIDInfo->isConstant())
      constOffset = optIDInfo->getConstantValues();
  }

  return AccessorInformation(buffer, bufferInfo, needsRange, constRange,
                             needsOffset, constOffset);
}

template <typename OperandType>
std::optional<IDRangeInformation>
SYCLAccessorAnalysis::getOperandInfo(sycl::SYCLHostConstructorOp constructor,
                                     size_t possibleIndex1,
                                     size_t possibleIndex2) {
  std::optional<IDRangeInformation> firstInfo;
  if (possibleIndex1 < constructor->getNumOperands())
    firstInfo =
        idRangeAnalysis.getIDRangeInformationFromConstruction<OperandType>(
            constructor, constructor->getOperand(possibleIndex1));

  std::optional<IDRangeInformation> secondInfo;
  if (possibleIndex2 < constructor->getNumOperands())
    secondInfo =
        idRangeAnalysis.getIDRangeInformationFromConstruction<OperandType>(
            constructor, constructor->getOperand(possibleIndex2));

  assert(!(firstInfo.has_value() && secondInfo.has_value()) &&
         "Only one argument of this type expected");

  return (firstInfo.has_value()) ? firstInfo : secondInfo;
}

} // namespace polygeist
} // namespace mlir
