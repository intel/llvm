//===- DataFlowSolverWrapper.h - Wrapper for a DataFlowSolver ----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utilities that allows dataflow analysis to declare and
// load any dataflow analysis they might require.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_POLYGEIST_ANALYSIS_DATAFLOWSOLVERWRAPPER_H
#define MLIR_DIALECT_POLYGEIST_ANALYSIS_DATAFLOWSOLVERWRAPPER_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir {
namespace polygeist {

class DataFlowSolverWrapper;

/// An interface that should be implemented by dataflow analysis in
/// order to declare which other dataflow analyses they might require.
template <class Derived> class RequiredDataFlowAnalyses {
public:
  /// Load dataflow analyses required by the \tparam Derived class in the \p
  /// solver.
  /// Note: derived classes must implement the static member function
  /// 'loadRequiredAnalysesImpl'.
  static void loadRequiredAnalyses(DataFlowSolverWrapper &solver) {
    Derived::loadRequiredAnalysesImpl(solver);
  }
};

/// Provides a way to load into a DataFlowSolver a dataflow analysis, along
/// with any other dataflow analyses it might require.
class DataFlowSolverWrapper : public DataFlowSolver {
public:
  DataFlowSolverWrapper(AliasAnalysis &aliasAnalysis)
      : aliasAnalysis(aliasAnalysis) {}

  AliasAnalysis &getAliasAnalysis() const { return aliasAnalysis; }

  /// Load an analysis into the solver, along with the dataflow analyses that
  /// the requested analysis requires.
  template <typename AnalysisT, typename... Args,
            typename = std::enable_if_t<std::is_base_of_v<
                RequiredDataFlowAnalyses<AnalysisT>, AnalysisT>>>
  void loadWithRequiredAnalysis(Args &&...args) {
    RequiredDataFlowAnalyses<AnalysisT>::loadRequiredAnalyses(*this);
    this->load<AnalysisT>(std::forward<Args>(args)...);
  }

private:
  AliasAnalysis &aliasAnalysis;
};

} // namespace polygeist
} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_ANALYSIS_DATAFLOWSOLVERWRAPPER_H
