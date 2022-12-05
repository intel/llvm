//==-------------------------- FusionHelper.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FusionHelper.h"

#include "ModuleHelper.h"
#include "helper/ErrorHandling.h"
#include "llvm/IR/Constants.h"

using namespace llvm;

template <typename T>
static Metadata *getConstantIntMD(llvm::LLVMContext &LLVMContext, T Val) {
  return ConstantAsMetadata::get(
      ConstantInt::get(Type::getInt64Ty(LLVMContext), Val));
}

static Metadata *getConstantMD(llvm::LLVMContext &LLVMCtx,
                               llvm::StringRef Data) {
  return MDString::get(LLVMCtx, Data);
}

static Metadata *getMDParam(LLVMContext &LLVMCtx,
                            const jit_compiler::Parameter &Param) {
  return MDNode::get(LLVMCtx,
                     {getConstantIntMD<unsigned>(LLVMCtx, Param.KernelIdx),
                      getConstantIntMD<unsigned>(LLVMCtx, Param.ParamIdx)});
}

Expected<std::unique_ptr<Module>> helper::FusionHelper::addFusedKernel(
    llvm::Module *LLVMModule,
    const std::vector<FusedFunction> &FusedFunctions) {

  // Get a clean module, containing only the input functions for fusion and
  // functions (transitively) called from there.
  PROPAGATE_ERROR(NewMod, getCleanModule(LLVMModule, FusedFunctions))

  llvm::LLVMContext &LLVMCtx = NewMod->getContext();

  const char *MetadataKind = "sycl.kernel.fused";
  const char *ParameterMDKind = "sycl.kernel.param";
  const char *InternalizationMDKind = "sycl.kernel.promote";
  const char *InternalizationLSMDKind = "sycl.kernel.promote.localsize";
  const char *ConstantsMDKind = "sycl.kernel.constants";
  // The function type of each kernel stub is identical ("void()"),
  // the fusion pass will insert the correct arguments based on
  // the fused functions later on.
  auto *FT = FunctionType::get(Type::getVoidTy(LLVMCtx), false);

  // Create a function stub for each fused kernel with metadata information
  // about which kernels should be fused into this kernel.
  for (const auto &FF : FusedFunctions) {
    // The function stub has a generic name (not the specified name of
    // the fused kernel), because we cannot change a function's signature
    // in-place in the fusion pass. Instead, we create a function stub
    // with generic name and create the actual fused function with the
    // correct name and signature only during the fusion pass.
    auto *F = Function::Create(FT, GlobalValue::LinkageTypes::ExternalLinkage,
                               "fused_kernel", *NewMod);

    // Attach metadata to the function stub.
    // The metadata specifies the name of the fused kernel (as the
    // stub only has a generic name, see note above) and lists the names
    // of all kernels that should be fused into this function.
    // Kernel names may appear multiple times, resulting in
    // the corresponding function to be included multiple times. The functions
    // are included in the same order as given by the list.
    SmallVector<Metadata *> MDFusedKernels;
    for (const auto &FK : FF.FusedKernels) {
      MDFusedKernels.push_back(MDString::get(LLVMCtx, FK));
    }
    auto *FusedKernelList = MDNode::get(LLVMCtx, MDFusedKernels);
    auto *FusedNameMD = MDString::get(LLVMCtx, FF.FusedName);
    auto *MDList = MDNode::get(LLVMCtx, {FusedNameMD, FusedKernelList});
    assert(!F->hasMetadata(MetadataKind));
    // The metadata can be identified by this fixed string providing a kind.
    F->setMetadata(MetadataKind, MDList);

    // The user of this API may be able to determine that
    // the same value is used for multiple input functions in the fused kernel,
    // e.g. when using the output of one kernel as the input to another kernel.
    // This information is also given as metadata, more specifically a list of
    // tuples. Each tuple contains two pairs identifying the two identical
    // parameters, e.g. ((0,1),(2,3)) means that the second argument of the
    // first kernel is identical to the fourth argument to the third kernel.
    // Attach this information as metadata here.
    if (!FF.ParameterIdentities.empty()) {
      SmallVector<Metadata *> MDParameterIdentities;
      for (const auto &PI : FF.ParameterIdentities) {
        auto *LHS = getMDParam(LLVMCtx, PI.LHS);
        auto *RHS = getMDParam(LLVMCtx, PI.RHS);
        MDParameterIdentities.push_back(MDNode::get(LLVMCtx, {LHS, RHS}));
      }
      assert(!F->hasMetadata(ParameterMDKind));
      F->setMetadata(ParameterMDKind,
                     MDNode::get(LLVMCtx, MDParameterIdentities));
    }

    // The user of this API may provide information about what arguments should
    // be internalized via promotion to local or private memory. This
    // information is given as metadata, as two list of tuples (one for each
    // kind of internalization) representing what parameters should be
    // internalized.
    {
      const auto &Internalization = FF.ParameterInternalization;
      if (!Internalization.empty()) {
        SmallVector<Metadata *> MDInternalizationKind;
        SmallVector<Metadata *> MDInternalizationLocalSize;
        const auto EmplaceBackIntern = [&](const auto &Info, auto Str) {
          std::array<Metadata *, 2> MDs;
          MDs[0] = getMDParam(LLVMCtx, Info.Param);
          MDs[1] = MDString::get(LLVMCtx, Str);
          MDInternalizationKind.emplace_back(MDNode::get(LLVMCtx, MDs));
          MDs[1] = getConstantIntMD<std::size_t>(LLVMCtx, Info.LocalSize);
          MDInternalizationLocalSize.emplace_back(MDNode::get(LLVMCtx, MDs));
        };
        for (const auto &Info : Internalization) {
          constexpr StringLiteral LocalInternalizationStr{"local"};
          constexpr StringLiteral PrivateInternalizationStr{"private"};

          const auto S = [&]() -> StringRef {
            switch (Info.Intern) {
            case jit_compiler::Internalization::None:
              llvm_unreachable(
                  "Only a valid internalization kind should be used");
            case jit_compiler::Internalization::Local:
              return LocalInternalizationStr;
            case jit_compiler::Internalization::Private:
              return PrivateInternalizationStr;
            }
          }();
          EmplaceBackIntern(Info, S);
        }
        assert(!F->hasMetadata(InternalizationMDKind));
        assert(!F->hasMetadata(InternalizationLSMDKind));
        F->setMetadata(InternalizationMDKind,
                       MDNode::get(LLVMCtx, MDInternalizationKind));
        F->setMetadata(InternalizationLSMDKind,
                       MDNode::get(LLVMCtx, MDInternalizationLocalSize));
      }
    }

    // The user of this API may provide information about what scalar values
    // should be used to specialize the fused kernel.
    {
      const auto &Constants = FF.Constants;
      if (!Constants.empty()) {
        SmallVector<Metadata *> MDConstants;
        for (const auto &C : Constants) {
          std::array<Metadata *, 2> MDs;
          MDs[0] = getMDParam(LLVMCtx, C.Param);
          MDs[1] = getConstantMD(LLVMCtx, C.Value);
          MDConstants.emplace_back(MDNode::get(LLVMCtx, MDs));
        }
        assert(!F->hasMetadata(ConstantsMDKind));
        F->setMetadata(ConstantsMDKind, MDNode::get(LLVMCtx, MDConstants));
      }
    }
  }
  return std::move(NewMod);
}

Expected<std::unique_ptr<llvm::Module>> helper::FusionHelper::getCleanModule(
    llvm::Module *LLVMMod, const std::vector<FusedFunction> &FusedFunctions) {
  // Find all input functions in the input module. Report an error and return
  // nothing in case one of the input functions is not present in the input
  // module.
  SmallVector<Function *, 5> InputFunctions;
  for (const auto &FF : FusedFunctions) {
    for (const auto &IF : FF.FusedKernels) {
      auto *InputFunction = LLVMMod->getFunction(IF);
      if (!InputFunction) {
        return createStringError(
            inconvertibleErrorCode(),
            "Input function %s not present in the input module\n", IF.c_str());
      }
      InputFunctions.push_back(InputFunction);
    }
  }
  return helper::ModuleHelper::cloneAndPruneModule(LLVMMod, InputFunctions);
}
