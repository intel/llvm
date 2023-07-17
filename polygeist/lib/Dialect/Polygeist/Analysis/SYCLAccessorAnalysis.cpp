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
  // TODO
  return os;
}

const AccessorInformation
AccessorInformation::join(const AccessorInformation &other,
                          AliasAnalysis &aliasAnalysis) const {
  // TODO
}

//===----------------------------------------------------------------------===//
// SYCLAccessorAnalysis
//===----------------------------------------------------------------------===//

SYCLAccessorAnalysis::SYCLAccessorAnalysis(Operation *op, AnalysisManager &mgr)
    : operation(op), am(mgr), idRangeAnalysis(op, mgr) {}

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

  // The default is true because we conservatively need to assume that the
  // accessor uses those two fields, unless we can infer otherwise.
  bool needsRange = true;
  bool needsOffset = true;
  SmallVector<size_t, 3> constRange;
  SmallVector<size_t, 3> constOffset;
  Value buffer = (constructor->getNumOperands() > 2)
                     ? constructor->getOperand(1)
                     : nullptr;

  // Deduct two for the this pointer and the DPC++-specific code location
  // argument.
  switch (constructor->getNumOperands() - 2) {
  case 0:
  case 2:
    needsRange = false;
    needsOffset = false;
    break;
  case 3: {
    needsOffset = false;
    // The second parameter can either be a handler, a tag or a range.
    Value secondArg = constructor->getOperand(2);
    needsRange = !(isHandler(secondArg) || isAccessTag(secondArg));
    if (needsRange) {
      auto rangeInfo =
          idRangeAnalysis
              .getIDRangeInformationFromConstruction<sycl::RangeType>(
                  constructor, secondArg);
      if (rangeInfo && rangeInfo->isConstant())
        constRange = rangeInfo->getConstantValues();
    }
    break;
  }
  case 4: {
    // The second parameter can either be a handler or a range.
    Value secondArg = constructor->getOperand(2);
    // The third parameter can either be a tag, a range or an id.
    Value thirdArg = constructor.getOperand(3);
    if (isHandler(secondArg)) {
      needsOffset = false;
      if (isAccessTag(thirdArg)) {
        needsRange = false;
        break;
      }
      auto rangeInfo =
          idRangeAnalysis
              .getIDRangeInformationFromConstruction<sycl::RangeType>(
                  constructor, thirdArg);
      if (rangeInfo && rangeInfo->isConstant())
        constRange = rangeInfo->getConstantValues();
      break;
    }
  }
  }
  // TODO
}

bool SYCLAccessorAnalysis::isHandler(Value value) const { return false; }

bool SYCLAccessorAnalysis::isAccessTag(Value value) const { return false; }

} // namespace polygeist
} // namespace mlir
