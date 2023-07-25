//===- SYCLNDRangeAnalysis.cpp - Analysis for sycl::nd_range --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Analysis/SYCLNDRangeAnalysis.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"

using namespace mlir;
using namespace mlir::polygeist;

//===----------------------------------------------------------------------===//
// NDRangeInformation
//===----------------------------------------------------------------------===//

static LogicalResult joinDimsInfo(IDRangeInformation &globalSizeInfo,
                                  IDRangeInformation &localSizeInfo,
                                  IDRangeInformation &offsetInfo) {
  // Split between top and non-top components
  SmallVector<IDRangeInformation *, 3> tops;
  SmallVector<const IDRangeInformation *, 3> nonTops;
  const auto addToVec = [&](IDRangeInformation &info) {
    if (info.hasFixedDimensions())
      nonTops.emplace_back(&info);
    else
      tops.emplace_back(&info);
  };
  addToVec(globalSizeInfo);
  addToVec(localSizeInfo);
  addToVec(offsetInfo);

  if (nonTops.empty())
    return success();

  // Get number of dimensions
  size_t dimensions = nonTops.front()->getNumDimensions();
  if (!std::all_of(nonTops.begin() + 1, nonTops.end(),
                   [=](const IDRangeInformation *info) {
                     return info->getNumDimensions() == dimensions;
                   }))
    return failure();

  // Set all top values to this number of dimensions
  for (IDRangeInformation *info : tops)
    *info = IDRangeInformation(dimensions);

  return success();
}

static bool isConstructor(const Definition &def) {
  if (!def.isOperation())
    return false;

  // NOTE: This could be extended to also handle `SYCLConstructorOp`.
  auto constructor = dyn_cast<sycl::SYCLHostConstructorOp>(def.getOperation());
  if (!constructor)
    return false;

  return isa<sycl::NdRangeType>(constructor.getType().getValue());
}

namespace mlir {
namespace polygeist {
NDRangeInformation::NDRangeInformation(size_t dimensions)
    : globalSizeInfo(dimensions), localSizeInfo(dimensions),
      offsetInfo(dimensions) {
  assert(1 <= dimensions && dimensions < 4 && "Invalid number of dimensions");
}

NDRangeInformation::NDRangeInformation(const IDRangeInformation &globalSizeInfo,
                                       const IDRangeInformation &localSizeInfo,
                                       const IDRangeInformation &offsetInfo)
    : globalSizeInfo(globalSizeInfo), localSizeInfo(localSizeInfo),
      offsetInfo(offsetInfo) {
  if (failed(joinDimsInfo(this->globalSizeInfo, this->localSizeInfo,
                          this->offsetInfo))) {
    IDRangeInformation top;
    this->globalSizeInfo = top;
    this->localSizeInfo = top;
    this->offsetInfo = top;
  }
}

const IDRangeInformation &NDRangeInformation::getGlobalSizeInfo() const {
  return globalSizeInfo;
}

const IDRangeInformation &NDRangeInformation::getLocalSizeInfo() const {
  return localSizeInfo;
}

const IDRangeInformation &NDRangeInformation::getOffsetInfo() const {
  return offsetInfo;
}

NDRangeInformation NDRangeInformation::join(const NDRangeInformation &lhs,
                                            const NDRangeInformation &rhs) {
  return {lhs.globalSizeInfo.join(rhs.globalSizeInfo),
          lhs.localSizeInfo.join(rhs.localSizeInfo),
          lhs.offsetInfo.join(rhs.offsetInfo)};
}

raw_ostream &operator<<(raw_ostream &os, const NDRangeInformation &ndr) {
  if (ndr.isTop())
    return os << "<unknown>";
  return os << "<global_size: " << ndr.globalSizeInfo
            << ", local_size: " << ndr.localSizeInfo
            << ", offset: " << ndr.offsetInfo << ">";
}

bool NDRangeInformation::isTop() const {
  return !globalSizeInfo.hasFixedDimensions();
}

//===----------------------------------------------------------------------===//
// SYCLNDRangeAnalysis
//===----------------------------------------------------------------------===//

SYCLNDRangeAnalysis::SYCLNDRangeAnalysis(Operation *op, AnalysisManager &am)
    : operation(op), am(am), idRangeAnalysis(op, am) {}

SYCLNDRangeAnalysis &SYCLNDRangeAnalysis::initialize(bool useRelaxedAliasing) {
  // Initialize the dataflow solver
  AliasAnalysis &aliasAnalysis = am.getAnalysis<mlir::AliasAnalysis>();
  aliasAnalysis.addAnalysisImplementation(
      sycl::AliasAnalysis(useRelaxedAliasing));
  solver = std::make_unique<DataFlowSolverWrapper>(aliasAnalysis);

  // Populate the solver and run the analyses needed by this analysis.
  solver->loadWithRequiredAnalysis<ReachingDefinitionAnalysis>(aliasAnalysis);

  if (failed(solver->initializeAndRun(operation))) {
    operation->emitError("Failed to run required dataflow analyses");
    return *this;
  }

  idRangeAnalysis.initialize(useRelaxedAliasing);

  initialized = true;

  return *this;
}

std::optional<NDRangeInformation>
SYCLNDRangeAnalysis::getNDRangeInformationFromConstruction(Operation *op,
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

  if (!llvm::all_of(*mods, isConstructor))
    return std::nullopt;

  auto pMods = reachingDef->getPotentialModifiers(operand, *solver);
  if (pMods) {
    if (!llvm::all_of(*pMods, isConstructor))
      return std::nullopt;
  }

  bool first = true;
  NDRangeInformation info;
  for (const Definition &def : *mods) {
    if (first) {
      info = getInformation(def);
      first = false;
      continue;
    }

    info = NDRangeInformation::join(info, getInformation(def));
    // Early return: As soon as joining of the different information has led to
    // top, we can end the processing.
    if (info.isTop())
      return info;
  }

  if (pMods) {
    for (const Definition &def : *pMods) {
      info = NDRangeInformation::join(info, getInformation(def));
      // Early return: As soon as joining of the different information has led
      // to top, we can end the processing.
      if (info.isTop())
        return info;
    }
  }

  return info;
}

NDRangeInformation SYCLNDRangeAnalysis::getInformation(const Definition &def) {
  assert(def.isOperation() && "Expecting operation");

  auto constructor = cast<sycl::SYCLHostConstructorOp>(def.getOperation());

  OperandRange args = constructor.getArgs();

  auto type = cast<sycl::NdRangeType>(constructor.getType().getValue());

  switch (args.size()) {
  default:
    // Unknown constructor
    return NDRangeInformation();
  case 1: {
    Value other = args[0];
    if (isa<LLVM::LLVMPointerType>(other.getType())) {
      // Copy constructor
      if (std::optional<NDRangeInformation> info =
              getNDRangeInformationFromConstruction(constructor, other))
        return *info;
      return NDRangeInformation(type.getDimension());
    }
    return NDRangeInformation();
  }
  case 2:
  case 3: {
    std::optional<IDRangeInformation> globalSizeInfo =
        idRangeAnalysis.getIDRangeInformationFromConstruction<sycl::RangeType>(
            constructor, args[0]);
    std::optional<IDRangeInformation> localSizeInfo =
        idRangeAnalysis.getIDRangeInformationFromConstruction<sycl::RangeType>(
            constructor, args[1]);
    std::optional<IDRangeInformation> offsetInfo =
        args.size() == 2
            // Offset is all zeroes by default
            ? std::make_optional<IDRangeInformation>(
                  SmallVector<size_t, 3>(type.getDimension()))
            : idRangeAnalysis
                  .getIDRangeInformationFromConstruction<sycl::IDType>(
                      constructor, args[2]);
    IDRangeInformation top;
    const auto getInfo = [&](const std::optional<IDRangeInformation> &info) {
      return info.value_or(top);
    };
    NDRangeInformation result(getInfo(globalSizeInfo), getInfo(localSizeInfo),
                              getInfo(offsetInfo));
    return result.isTop() ? NDRangeInformation(type.getDimension()) : result;
  }
  }
}
} // namespace polygeist
} // namespace mlir
