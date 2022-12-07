//===--------------- PassAdaptors.h - SYCL Pass Adaptors ----------------- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This header defines Pass Adaptors which allows to run Function Passes
/// on a subset of functions from given Module.
/// LLVM does not provide a default utility for this purpose.
//===----------------------------------------------------------------------===//

#include "llvm/IR/PassManager.h"
#include "llvm/IR/PassManagerInternal.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

#ifndef LLVM_SYCL_IR_PASS_ADAPTORS
#define LLVM_SYCL_IR_PASS_ADAPTORS

namespace llvm {

/// Trivial adaptor that maps from a module to functions marked
/// by "sycl-framework" metadata. The implementation is mostly a copy-paste
/// of LLVM ModuleToFunctionPassAdaptor class.
///
/// Designed to allow composition of a FunctionPass(Manager) and
/// a ModulePassManager, by running the FunctionPass(Manager) over every
/// "sycl-framework "function in the module.
class ModuleToSYCLFrameworkFunctionPassAdaptor
    : public PassInfoMixin<ModuleToSYCLFrameworkFunctionPassAdaptor> {
public:
  using PassConceptT = detail::PassConcept<Function, FunctionAnalysisManager>;

  explicit ModuleToSYCLFrameworkFunctionPassAdaptor(
      std::unique_ptr<PassConceptT> Pass, bool EagerlyInvalidate)
      : Pass(std::move(Pass)), EagerlyInvalidate(EagerlyInvalidate) {}

  /// Runs the function pass across every "sycl-framework" function in the
  /// module.
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName);

  static bool isRequired() { return true; }

private:
  std::unique_ptr<PassConceptT> Pass;
  bool EagerlyInvalidate;
};

/// A function to deduce a function pass type and wrap it in the
/// templated adaptor.
template <typename FunctionPassT>
ModuleToSYCLFrameworkFunctionPassAdaptor
createModuleToSYCLFrameworkFunctionPassAdaptor(FunctionPassT &&Pass,
                                               bool EagerlyInvalidate = false) {
  using PassModelT =
      detail::PassModel<Function, FunctionPassT, PreservedAnalyses,
                        FunctionAnalysisManager>;
  // Do not use make_unique, it causes too many template instantiations,
  // causing terrible compile times.
  return ModuleToSYCLFrameworkFunctionPassAdaptor(
      std::unique_ptr<ModuleToSYCLFrameworkFunctionPassAdaptor::PassConceptT>(
          new PassModelT(std::forward<FunctionPassT>(Pass))),
      EagerlyInvalidate);
}

} // end namespace llvm

#endif // LLVM_SYCL_IR_PASS_ADAPTORS
