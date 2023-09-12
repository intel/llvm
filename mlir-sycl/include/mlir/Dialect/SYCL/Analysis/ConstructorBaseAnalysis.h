//===- ConstructorBaseAnalysis.h - Base analysis for SYCL types -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class for analyses that determine properties of instances of SYCL types
// based on their construction.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_ANALYSIS_CONSTRUCTORBASEANALYSIS_H
#define MLIR_DIALECT_SYCL_ANALYSIS_CONSTRUCTORBASEANALYSIS_H

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Polygeist/Analysis/DataFlowSolverWrapper.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Pass/AnalysisManager.h"

namespace mlir {
namespace sycl {

/// CRTP base class for analyses deriving properties of interest for SYCL
/// objects from their construction. The analyses derived from this class, i.e.,
/// `ConcreteAnalysis` must implement the following methods:
/// * `void finalizeInitialization(bool useRelaxedAliasing)`
/// * `bool isConstructorImpl(const polygeist::Definition &def)`
/// * `AnalysisResult getInformationImpl(const polygeist::Definition &def)`
///
/// The class representing information derived by the analysis, i.e.,
/// `AnalysisResult` must define a default constructor and must implement the
/// following methods:
/// * `bool isTop()` to indicate the top of the lattice has been reached.
/// * `AnalysisResult join(const AnalysisResult& other)` to join two results at
/// a merge point.
///
template <typename ConcreteAnalysis, typename AnalysisResult>
class ConstructorBaseAnalysis {
public:
  ConcreteAnalysis &initialize(bool useRelaxedAliasing = false) {

    // Initialize the dataflow solver
    aliasAnalysis = &am.getAnalysis<mlir::AliasAnalysis>();
    aliasAnalysis->addAnalysisImplementation(
        sycl::AliasAnalysis(useRelaxedAliasing));

    solver = std::make_unique<polygeist::DataFlowSolverWrapper>(*aliasAnalysis);

    // Populate the solver and run the analyses needed by this analysis.
    solver->loadWithRequiredAnalysis<polygeist::ReachingDefinitionAnalysis>(
        *aliasAnalysis);

    if (failed(solver->initializeAndRun(operation))) {
      operation->emitError("Failed to run required dataflow analyses");
      return *static_cast<ConcreteAnalysis *>(this);
    }

    // Let the derived class perform more initialization of necessary.
    static_cast<ConcreteAnalysis *>(this)->finalizeInitialization(
        useRelaxedAliasing);

    initialized = true;

    return *static_cast<ConcreteAnalysis *>(this);
  }

  ConstructorBaseAnalysis(Operation *op, AnalysisManager &mgr)
      : operation{op}, am{mgr}, solver{nullptr}, aliasAnalysis{nullptr} {}

protected:
  template <typename... SYCLType>
  std::optional<AnalysisResult> getInformationFromConstruction(Operation *op,
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

    if (!llvm::all_of(*mods, [&](const polygeist::Definition &def) {
          return isConstructor<SYCLType...>(def);
        }))
      return std::nullopt;

    auto pMods = reachingDef->getPotentialModifiers(operand, *solver);
    if (pMods) {
      if (!llvm::all_of(*pMods, [&](const polygeist::Definition &def) {
            return isConstructor<SYCLType...>(def);
          }))
        return std::nullopt;
    }

    bool first = true;
    AnalysisResult info;
    for (const polygeist::Definition &def : *mods) {
      if (first)
        info = getInformation<SYCLType...>(def);
      else
        info = info.join(getInformation<SYCLType...>(def), *aliasAnalysis);

      first = false;
      if (info.isTop())
        // Early return: As soon as joining the different information has
        // reached the top of the lattice, we can end processing.
        return info;
    }

    if (pMods) {
      for (const polygeist::Definition &def : *pMods) {
        info = info.join(getInformation<SYCLType...>(def), *aliasAnalysis);
        if (info.isTop())
          // Early return: As soon as joining the different information has
          // reached the top of the lattice, we can end processing.
          return info;
      }
    }

    return info;
  }

  Operation *operation;

  AnalysisManager &am;

  std::unique_ptr<polygeist::DataFlowSolverWrapper> solver;

  mlir::AliasAnalysis *aliasAnalysis;

private:
  template <typename... SYCLType>
  bool isConstructor(const polygeist::Definition &def) {
    if (!def.isOperation())
      return false;

    auto constructor =
        dyn_cast<sycl::SYCLHostConstructorOp>(def.getOperation());
    if (!constructor)
      return false;

    return isa<SYCLType...>(constructor.getType().getValue());
  }

  template <typename... SYCLType>
  AnalysisResult getInformation(const polygeist::Definition &def) {
    return static_cast<ConcreteAnalysis *>(this)
        ->template getInformationImpl<SYCLType...>(def);
  }

  bool initialized = false;
};
} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_ANALYSIS_CONSTRUCTORBASEANALYSIS_H
