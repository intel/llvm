//==------ SYCLKernelFusion.h - Pass to create fused kernel definition -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_PASSES_SYCLKERNELFUSION_H
#define SYCL_FUSION_PASSES_SYCLKERNELFUSION_H

#include "Kernel.h"
#include "target/TargetFusionInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

///
/// Pass to fuse multiple SYCL kernels into a single kernel function.
/// The fusion is triggered by a stub function carrying metadata
/// specifying the desired name of the fused kernel as well as the
/// list of kernels to fuse into the new kernel.
/// The definitions of all input functions must be present in the module
/// and are retained by this pass. Stub functions and metadata are
/// erased from the module.
///
/// This pass may introduce redundant function calls and some conditional branch
/// instructions and may benefit from running further optimization passes.
class SYCLKernelFusion : public llvm::PassInfoMixin<SYCLKernelFusion> {

public:
  constexpr static llvm::StringLiteral NDRangeMDKey{"sycl.kernel.nd-range"};
  constexpr static llvm::StringLiteral NDRangesMDKey{"sycl.kernel.nd-ranges"};

  constexpr SYCLKernelFusion() = default;
  constexpr explicit SYCLKernelFusion(jit_compiler::BarrierFlags BarriersFlags)
      : BarriersFlags{BarriersFlags} {}

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

  ///
  /// Default value for the BarriersFlags member.
  ///
  /// By default, correct ordering of memory operations to global memory is
  /// ensured.
  constexpr static jit_compiler::BarrierFlags DefaultBarriersFlags{
      jit_compiler::getLocalAndGlobalBarrierFlag()};

private:
  // This needs to be in sync with the metadata kind
  // used by the FusionHelper to make sure we can
  // locate our own metadata again.
  static constexpr auto MetadataKind = "sycl.kernel.fused";
  static constexpr auto ParameterMDKind = "sycl.kernel.param";

  using MDList = llvm::SmallVector<llvm::Metadata *, 16>;

  ///
  /// Helper struct to identify parameters by the index
  /// of their kernel in the list of input kernels and
  /// the index of the parameter within the argument list
  /// of that kernel.
  struct Parameter {
    unsigned KernelIdx;
    unsigned ParamIdx;

    bool operator==(const Parameter &Other) const {
      return KernelIdx == Other.KernelIdx && ParamIdx == Other.ParamIdx;
    }

    bool operator<(const Parameter &Other) const {
      if (KernelIdx < Other.KernelIdx) {
        return true;
      }
      return KernelIdx == Other.KernelIdx && ParamIdx < Other.ParamIdx;
    }
  };

  ///
  /// Helper struct to express that two parameters identified by kernel
  /// and parameter index) will have an identical value.
  /// The struct is constructed such that LHS > RHS holds, i.e., that
  /// the LHS parameter comes from a kernel appearing later in the list
  /// of input kernels or from the same kernel, but later in the parameter list.
  struct ParameterIdentity {
    Parameter LHS;
    Parameter RHS;

    ParameterIdentity() = delete;

    ParameterIdentity(Parameter Param1, Parameter Param2) {
      if (Param1 < Param2) {
        LHS = Param2;
        RHS = Param1;
      } else {
        LHS = Param1;
        RHS = Param2;
      }
    }

    bool operator==(const ParameterIdentity &Other) const {
      return LHS == Other.LHS && RHS == Other.RHS;
    }

    bool operator<(const ParameterIdentity &Other) const {
      return LHS < Other.LHS;
    }
  };

  llvm::Error
  fuseKernel(llvm::Module &M, llvm::Function &StubFunction,
             jit_compiler::SYCLModuleInfo *ModInfo,
             llvm::TargetFusionInfo &TargetInfo,
             llvm::SmallPtrSetImpl<llvm::Function *> &ToCleanUp) const;

  void canonicalizeParameters(
      llvm::SmallVectorImpl<ParameterIdentity> &Params) const;

  Parameter getParamFromMD(llvm::Metadata *MD) const;

  void attachFusedMetadata(
      llvm::Function *FusedFunction, const llvm::StringRef &Kind,
      const llvm::ArrayRef<llvm::Metadata *> FusedMetadata) const;

  void
  attachKernelAttributeMD(llvm::LLVMContext &LLVMCtx,
                          llvm::Function *FusedFunction,
                          jit_compiler::SYCLKernelInfo &FusedKernelInfo) const;

  using MutableParamKindList = llvm::SmallVector<jit_compiler::ParameterKind>;
  using MutableArgUsageMask = llvm::SmallVector<jit_compiler::ArgUsageUT>;
  using MutableAttributeList =
      llvm::SmallVector<jit_compiler::SYCLKernelAttribute>;

  void appendKernelInfo(MutableParamKindList &FusedParamKinds,
                        MutableArgUsageMask &FusedArgUsageMask,
                        MutableAttributeList &FusedAttributes,
                        jit_compiler::SYCLKernelInfo &InputInfo,
                        const llvm::ArrayRef<bool> ParamUseMask) const;

  void updateArgUsageMask(MutableArgUsageMask &NewMask,
                          jit_compiler::SYCLArgumentDescriptor &InputDef,
                          const llvm::ArrayRef<bool> ParamUseMask) const;

  using KernelAttributeList = jit_compiler::SYCLAttributeList;

  using KernelAttr = jit_compiler::SYCLKernelAttribute;
  using KernelAttrKind = jit_compiler::SYCLKernelAttribute::AttrKind;

  ///
  /// Indicates the result of merging two attributes of the same kind.
  enum class AttrMergeResult {
    AddAttr,    // Add the right-hand side of the merge to the list.
    KeepAttr,   // Keep the existing attribute in the list.
    RemoveAttr, // Remove the existing attribute from the list.
    UpdateAttr, // The existing attribute has been updated in place.
    Error       // Unable to merge the attribute values, throw error.
  };

  ///
  /// Flags to apply to the barrier to be introduced between fused kernels.
  ///
  const jit_compiler::BarrierFlags BarriersFlags{DefaultBarriersFlags};

  ///
  /// Merge the content of Other into Attributes, adding, removing or updating
  /// attributes as needed.
  void mergeKernelAttributes(MutableAttributeList &Attributes,
                             const KernelAttributeList &Other) const;

  ///
  /// Merge values for the same attribute.
  /// If Attr is nullptr, assumes that the existing list did not yet have an
  /// instance of this attribute.
  AttrMergeResult mergeAttribute(KernelAttr *Attr,
                                 const KernelAttr &Other) const;

  ///
  /// Merge the required workgroup size attribute.
  AttrMergeResult mergeReqdWorkgroupSize(KernelAttr *Attr,
                                         const KernelAttr &Other) const;

  ///
  /// Merge the workgroup size hint attribute.
  AttrMergeResult mergeWorkgroupSizeHint(KernelAttr *Attr,
                                         const KernelAttr &Other) const;

  ///
  /// Get the attribute with the specified kind from the list or return nullptr
  /// in case no such attribute is present.
  KernelAttr *getAttribute(MutableAttributeList &Attributes,
                           KernelAttrKind AttrKind) const;

  ///
  /// Add the attribute to the list.
  void addAttribute(MutableAttributeList &Attributes,
                    const KernelAttr &Attr) const;

  ///
  /// Remove the attribute with the specified kind from the list, if present.
  void removeAttribute(MutableAttributeList &Attributes,
                       KernelAttrKind AttrKind) const;

  ///
  /// Find the attribute with the specified kind in the list, or return the
  /// end() iterator if no such attribute is present.
  MutableAttributeList::iterator findAttribute(MutableAttributeList &Attributes,
                                               KernelAttrKind AttrKind) const;
};

#endif // SYCL_FUSION_PASSES_SYCLKERNELFUSION_H
