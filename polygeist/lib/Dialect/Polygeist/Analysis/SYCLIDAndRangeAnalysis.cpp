//===- SYCLIDAndRangeAnalysis.cpp - Analysis for sycl::id/range -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Analysis/SYCLIDAndRangeAnalysis.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"
#include <memory>

#define DEBUG_TYPE "sycl-id-range-analysis"

namespace mlir {
namespace polygeist {

//===----------------------------------------------------------------------===//
// IDRangeInformation
//===----------------------------------------------------------------------===//

raw_ostream &operator<<(raw_ostream &os, const IDRangeInformation &info) {
  if (!info.hasFixedDimensions()) {
    os << "<unknown>";
    return os;
  }

  if (info.isConstant()) {
    os << "constant<";
    llvm::interleaveComma(info.getConstantValues(), os);
    os << ">";
    return os;
  }

  os << "fixed<" << info.getNumDimensions() << ">";
  return os;
}

IDRangeInformation::IDRangeInformation()
    : dimensions{std::nullopt}, constantValues{std::nullopt} {}

IDRangeInformation::IDRangeInformation(size_t dim)
    : dimensions{dim}, constantValues{std::nullopt} {}

IDRangeInformation::IDRangeInformation(llvm::ArrayRef<size_t> constVals)
    : dimensions{constVals.size()}, constantValues{constVals} {}

bool IDRangeInformation::hasFixedDimensions() const {
  return dimensions.has_value();
}

size_t IDRangeInformation::getNumDimensions() const {
  assert(hasFixedDimensions() &&
         "Requesting fixed dimensions from non-fixed id/range");
  return *dimensions;
}

bool IDRangeInformation::isConstant() const {
  return constantValues.has_value();
}

const llvm::SmallVector<size_t, 3> &
IDRangeInformation::getConstantValues() const {
  assert(isConstant() &&
         "Requesting constant values from non-constant id/range");
  return *constantValues;
}

const IDRangeInformation
IDRangeInformation::join(const IDRangeInformation &other) const {
  if (isConstant() && other.isConstant()) {
    if (getConstantValues() == other.getConstantValues())
      return *this;
  }
  if (hasFixedDimensions() && other.hasFixedDimensions()) {
    if (getNumDimensions() == other.getNumDimensions())
      return IDRangeInformation{getNumDimensions()};
  }
  return IDRangeInformation{};
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

template <typename T>
struct has_default_constructor : public std::is_same<T, mlir::sycl::IDType> {};

template <typename T>
constexpr bool has_default_constructor_v = has_default_constructor<T>::value;

static std::optional<int> getConstantUInt(Value v) {
  Operation *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;

  if (!op->hasTrait<OpTrait::ConstantLike>())
    return std::nullopt;

  llvm::SmallVector<OpFoldResult> folded;
  if (failed(op->fold({}, folded)) || folded.size() != 1)
    return std::nullopt;

  if (!folded.front().is<Attribute>() ||
      !isa<IntegerAttr>(folded.front().get<Attribute>()))
    return std::nullopt;

  return cast<IntegerAttr>(folded.front().get<Attribute>()).getInt();
}

template <typename Type> bool isConstructor(const Definition &def) {
  if (!def.isOperation())
    return false;

  // NOTE: This could be extended to also handle `SYCLConstructorOp`.
  auto constructor = dyn_cast<sycl::SYCLHostConstructorOp>(def.getOperation());
  if (!constructor)
    return false;

  return isa<Type>(constructor.getType().getValue());
}

} // namespace

//===----------------------------------------------------------------------===//
// SYCLIDAndRangeAnalysis
//===----------------------------------------------------------------------===//

SYCLIDAndRangeAnalysis::SYCLIDAndRangeAnalysis(Operation *op,
                                               AnalysisManager &mgr)
    : operation(op), am(mgr) {}

SYCLIDAndRangeAnalysis &
SYCLIDAndRangeAnalysis::initialize(bool useRelaxedAliasing) {
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

  initialized = true;

  return *this;
}

template <typename Type>
std::optional<IDRangeInformation>
SYCLIDAndRangeAnalysis::getIDRangeInformationFromConstruction(Operation *op,
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

  if (!llvm::all_of(*mods, isConstructor<Type>))
    return std::nullopt;

  auto pMods = reachingDef->getPotentialModifiers(operand, *solver);
  if (pMods) {
    if (!llvm::all_of(*pMods, isConstructor<Type>))
      return std::nullopt;
  }

  bool first = true;
  IDRangeInformation info;
  for (const Definition &def : *mods) {
    if (first) {
      info = getInformation<Type>(def);
      first = false;
    } else {
      info = info.join(getInformation<Type>(def));
      if (!info.hasFixedDimensions())
        // Early return: As soon as joining of the different information has led
        // to an info with no fixed dimension (and therefore also non-constant
        // values), we can end the processing.
        return info;
    }
  }

  if (pMods) {
    for (const Definition &def : *pMods) {
      info = info.join(getInformation<Type>(def));
      if (!info.hasFixedDimensions())
        // Early return: As soon as joining of the different information has led
        // to an info with no fixed dimension (and therefore also non-constant
        // values), we can end the processing.
        return info;
    }
  }

  return info;
}

template <typename IDRange>
IDRangeInformation
SYCLIDAndRangeAnalysis::getInformation(const Definition &def) {
  assert(def.isOperation() && "Expecting operation");

  auto constructor = cast<sycl::SYCLHostConstructorOp>(def.getOperation());

  auto type = cast<IDRange>(constructor.getType().getValue());

  OperandRange args = constructor.getArgs();

  if constexpr (has_default_constructor_v<IDRange>) {
    if (args.empty()) {
      // Default constructor
      std::array<size_t, 3> values{0, 0, 0};
      return IDRangeInformation(
          ArrayRef<size_t>(values.data(), type.getDimension()));
    }
  }

  assert(!args.empty() && "Expecting at least one argument");

  if (args.size() == 1) {
    Value other = args[0];
    if (isa<LLVM::LLVMPointerType>(other.getType())) {
      // Copy constructor
      if (std::optional<IDRangeInformation> info =
              getIDRangeInformationFromConstruction<IDRange>(constructor,
                                                             other))
        return *info;
      return IDRangeInformation(type.getDimension());
    }
  }

  // (std::size_t+) constructor

  assert(llvm::all_of(args.getTypes(),
                      [](Type ty) { return isa<IntegerType>(ty); }));

  SmallVector<std::optional<int>> constValues;
  llvm::transform(args, std::back_inserter(constValues), getConstantUInt);

  if (llvm::all_of(constValues, [](auto &opt) { return opt.has_value(); })) {
    SmallVector<size_t, 3> constInt;
    llvm::transform(constValues, std::back_inserter(constInt),
                    [](auto &opt) { return *opt; });
    return IDRangeInformation(constInt);
  }

  return IDRangeInformation(type.getDimension());
}

template std::optional<IDRangeInformation>
SYCLIDAndRangeAnalysis::getIDRangeInformationFromConstruction<sycl::IDType>(
    Operation *, Value);

template std::optional<IDRangeInformation>
SYCLIDAndRangeAnalysis::getIDRangeInformationFromConstruction<sycl::RangeType>(
    Operation *, Value);

} // namespace polygeist
} // namespace mlir
