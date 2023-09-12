//===- SYCLNDRangeAnalysis.cpp - Analysis for sycl::nd_range --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Analysis/SYCLNDRangeAnalysis.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/ConstructorBaseAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"

using namespace mlir;
using namespace mlir::sycl;

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

namespace mlir {
namespace sycl {
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

const NDRangeInformation
NDRangeInformation::join(const NDRangeInformation &other,
                         mlir::AliasAnalysis &alias) {
  return {globalSizeInfo.join(other.globalSizeInfo, alias),
          localSizeInfo.join(other.localSizeInfo, alias),
          offsetInfo.join(other.offsetInfo, alias)};
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
    : ConstructorBaseAnalysis<SYCLNDRangeAnalysis, NDRangeInformation>(op, am),
      idRangeAnalysis(op, am) {}

void SYCLNDRangeAnalysis::finalizeInitialization(bool useRelaxedAliasing) {
  idRangeAnalysis.initialize(useRelaxedAliasing);
}

std::optional<NDRangeInformation>
SYCLNDRangeAnalysis::getNDRangeInformationFromConstruction(Operation *op,
                                                           Value operand) {
  return getInformationFromConstruction<sycl::NdRangeType>(op, operand);
}

template <>
NDRangeInformation SYCLNDRangeAnalysis::getInformationImpl<sycl::NdRangeType>(
    const polygeist::Definition &def) {
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
} // namespace sycl
} // namespace mlir
