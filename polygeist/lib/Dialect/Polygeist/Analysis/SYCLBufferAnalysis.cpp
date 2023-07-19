//===- SYCLBufferAnalysis.cpp - Analysis for sycl::buffer -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Analysis/SYCLBufferAnalysis.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"
#include <memory>

#define DEBUG_TYPE "sycl-buffer-analysis"

namespace mlir {
namespace polygeist {

//===----------------------------------------------------------------------===//
// BufferInformation
//===----------------------------------------------------------------------===//

raw_ostream &operator<<(raw_ostream &os, const BufferInformation &info) {
  os.indent(4) << "Sub-Buffer: ";
  switch (info.getSubBuffer()) {
  case SubBufferLattice::YES:
    os << "Yes";
    break;
  case SubBufferLattice::NO:
    os << "No";
    break;
  case SubBufferLattice::MAYBE:
    os << "Maybe";
  };
  os << "\n";
  os.indent(4) << "Size: ";
  if (info.hasConstantSize()) {
    os << "range{";
    llvm::interleaveComma(info.getConstantSize(), os);
    os << "}\n";
  } else {
    os << "<unknown>\n";
  }
  if (info.getSubBuffer() == SubBufferLattice::YES) {
    os.indent(4) << "Base buffer: ";
    if (info.hasKnownBaseBuffer())
      os << info.getKnownBaseBuffer();
    else
      os << "<unknown>";

    os << "\n";
    os.indent(4) << "Base buffer size: ";
    if (info.hasKnownBaseBufferSize()) {
      os << "range{";
      llvm::interleaveComma(info.getKnownBaseBufferSize(), os);
      os << "}\n";
    } else {
      os << "<unknown>\n";
    }
    os.indent(4) << "Sub-buffer offset: ";
    if (info.hasConstantOffset()) {
      os << "id{";
      llvm::interleaveComma(info.getConstantOffset(), os);
      os << "}\n";
    } else {
      os << "<unknown>\n";
    }
  }
  return os;
}

BufferInformation::BufferInformation()
    : constantSize{}, subBuf{SubBufferLattice::MAYBE}, baseBuffer{nullptr},
      baseBufferSize{}, subBufOffset{} {}

BufferInformation::BufferInformation(
    llvm::ArrayRef<size_t> constRange, SubBufferLattice IsSubBuffer,
    Value baseBuffer, llvm::ArrayRef<size_t> constBaseBufferSize,
    llvm::ArrayRef<size_t> constSubBufferOffset)
    : constantSize{constRange}, subBuf{IsSubBuffer}, baseBuffer{baseBuffer},
      baseBufferSize{constBaseBufferSize}, subBufOffset{constSubBufferOffset} {}

const BufferInformation
BufferInformation::join(const BufferInformation &other,
                        AliasAnalysis &aliasAnalysis) const {
  auto jointSize = (constantSize == other.constantSize)
                       ? constantSize
                       : SmallVector<size_t, 3>{};

  if (subBuf == SubBufferLattice::YES && subBuf == other.subBuf) {
    Value jointBaseBuf = nullptr;
    if (baseBuffer == other.baseBuffer ||
        aliasAnalysis.alias(baseBuffer, other.baseBuffer) ==
            AliasResult::MustAlias)
      jointBaseBuf = baseBuffer;

    auto jointBaseBufSize = (baseBufferSize == other.baseBufferSize)
                                ? baseBufferSize
                                : SmallVector<size_t, 3>{};

    auto jointOffset = (subBufOffset == other.subBufOffset)
                           ? subBufOffset
                           : SmallVector<size_t, 3>{};

    return BufferInformation(jointSize, SubBufferLattice::YES, jointBaseBuf,
                             jointBaseBufSize, jointOffset);
  }

  // When control flow reaches this point, either this or the incoming
  // information are not definitely a sub-buffer. If they are both definitely
  // *not* a sub-buffer, preserve that fact, otherwise be conservative and
  // assume the resulting information might be a sub-buffer.
  auto jointSubBuf = (subBuf == other.subBuf && subBuf == SubBufferLattice::NO)
                         ? SubBufferLattice::NO
                         : SubBufferLattice::MAYBE;

  return BufferInformation(jointSize, jointSubBuf, nullptr, {}, {});
}

//===----------------------------------------------------------------------===//
// SYCLBufferAnalysis
//===----------------------------------------------------------------------===//

SYCLBufferAnalysis::SYCLBufferAnalysis(Operation *op, AnalysisManager &mgr)
    : operation(op), am(mgr), aliasAnalysis(nullptr), idRangeAnalysis(op, mgr) {
}

SYCLBufferAnalysis &SYCLBufferAnalysis::initialize(bool useRelaxedAliasing) {

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

std::optional<BufferInformation>
SYCLBufferAnalysis::getBufferInformationFromConstruction(Operation *op,
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
  BufferInformation info;
  for (const Definition &def : *mods) {
    if (first) {
      info = getInformation(def);
      first = false;
    } else {
      info = info.join(getInformation(def), *aliasAnalysis);
      if (!info.hasConstantSize() &&
          info.getSubBuffer() == SubBufferLattice::MAYBE)
        // Early return: As soon as joining of the different information has led
        // to an info with no fixed size and no definitive sub-buffer
        // information, we can end the processing.
        return info;
    }
  }

  if (pMods) {
    for (const Definition &def : *pMods) {
      info = info.join(getInformation(def), *aliasAnalysis);
      if (!info.hasConstantSize() &&
          info.getSubBuffer() == SubBufferLattice::MAYBE)
        // Early return: As soon as joining of the different information has led
        // to an info with no fixed size and no definitive sub-buffer
        // information, we can end the processing.
        return info;
    }
  }

  return info;
}

bool SYCLBufferAnalysis::isConstructor(const Definition &def) {
  if (!def.isOperation())
    return false;

  auto constructor = dyn_cast<sycl::SYCLHostConstructorOp>(def.getOperation());
  if (!constructor)
    return false;

  return isa<sycl::BufferType>(constructor.getType().getValue());
}

BufferInformation SYCLBufferAnalysis::getInformation(const Definition &def) {
  assert(def.isOperation() && "Expecting operation");

  auto constructor = cast<sycl::SYCLHostConstructorOp>(def.getOperation());

  bool isSubBuffer =
      cast<sycl::BufferType>(constructor.getType().getValue()).getSubbuffer();

  if (isSubBuffer) {
    Value baseBuffer = constructor->getOperand(1);

    // Try to collect information about the base buffer
    auto baseBufferInfo =
        getBufferInformationFromConstruction(constructor, baseBuffer);

    auto baseBufferSize =
        (baseBufferInfo && baseBufferInfo->hasConstantSize())
            ? SmallVector<size_t, 3>(baseBufferInfo->getConstantSize())
            : SmallVector<size_t, 3>{};

    SmallVector<size_t, 3> offset;
    if (constructor->getNumOperands() > 2) {
      // Try to determine if the offset into the base buffer is constant.
      auto offsetInfo =
          idRangeAnalysis.getIDRangeInformationFromConstruction<sycl::IDType>(
              constructor, constructor->getOperand(2));
      if (offsetInfo && offsetInfo->isConstant())
        offset = offsetInfo->getConstantValues();
    }

    SmallVector<size_t, 3> subRange;
    if (constructor.getNumOperands() > 3) {
      // Try to determine if the sub-range of this sub-buffer is constant.
      auto rangeInfo =
          idRangeAnalysis
              .getIDRangeInformationFromConstruction<sycl::RangeType>(
                  constructor, constructor->getOperand(3));
      if (rangeInfo && rangeInfo->isConstant())
        subRange = rangeInfo->getConstantValues();
    }

    return BufferInformation(subRange, SubBufferLattice::YES, baseBuffer,
                             baseBufferSize, offset);
  }

  // Try to determine if the size of this buffer is constant. If the buffer is
  // not a sub-buffer, the `sycl::range` argument will either be the first or
  // second argument in C++, aka the second or third argument in IR.

  SmallVector<size_t, 3> constantSize;
  if (constructor->getNumOperands() > 1) {
    auto rangeInfo =
        idRangeAnalysis.getIDRangeInformationFromConstruction<sycl::RangeType>(
            constructor, constructor->getOperand(1));
    if (rangeInfo && rangeInfo->isConstant())
      constantSize = rangeInfo->getConstantValues();
  }

  if (constructor.getNumOperands() > 2) {
    auto rangeInfo =
        idRangeAnalysis.getIDRangeInformationFromConstruction<sycl::RangeType>(
            constructor, constructor->getOperand(2));
    assert(!(!constantSize.empty() && rangeInfo) &&
           "Expecting only one of the second and third argument to be a range");

    if (rangeInfo && rangeInfo->isConstant())
      constantSize = rangeInfo->getConstantValues();
  }

  return BufferInformation(constantSize, SubBufferLattice::NO, nullptr, {}, {});
}

} // namespace polygeist
} // namespace mlir
