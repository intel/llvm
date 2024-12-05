//===----- SemaSYCL.h ------- Semantic Analysis for SYCL constructs -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis for SYCL constructs.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMASYCL_H
#define LLVM_CLANG_SEMA_SEMASYCL_H

#include "clang/AST/ASTFwd.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Cuda.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/SemaBase.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

namespace clang {
class Decl;
class ParsedAttr;

class CXXMethodDecl;
class MangleContext;
class SemaSYCL;

// TODO SYCL Integration header approach relies on an assumption that kernel
// lambda objects created by the host compiler and any of the device compilers
// will be identical wrt to field types, order and offsets. Some verification
// mechanism should be developed to enforce that.

// TODO FIXME SYCL Support for SYCL in FE should be refactored:
// - kernel identification and generation should be made a separate pass over
// AST. RecursiveASTVisitor + VisitFunctionTemplateDecl +
// FunctionTemplateDecl::getSpecializations() mechanism could be used for that.
// - All SYCL stuff on Sema level should be encapsulated into a single Sema
// field
// - Move SYCL stuff into a separate header

// Represents contents of a SYCL integration header file produced by a SYCL
// device compiler and used by SYCL host compiler (via forced inclusion into
// compiled SYCL source):
// - SYCL kernel names
// - SYCL kernel parameters and offsets of corresponding actual arguments
class SYCLIntegrationHeader {
public:
  // Kind of kernel's parameters as captured by the compiler in the
  // kernel lambda or function object
  enum kernel_param_kind_t {
    kind_first,
    kind_accessor = kind_first,
    kind_std_layout,
    kind_sampler,
    kind_pointer,
    kind_specialization_constants_buffer,
    kind_stream,
    kind_work_group_memory,
    kind_last = kind_work_group_memory
  };

public:
  SYCLIntegrationHeader(SemaSYCL &S);

  /// Emits contents of the header into given stream.
  void emit(raw_ostream &Out);

  /// Emits contents of the header into a file with given name.
  /// Returns true/false on success/failure.
  bool emit(StringRef MainSrc);

  ///  Signals that subsequent parameter descriptor additions will go to
  ///  the kernel with given name. Starts new kernel invocation descriptor.
  void startKernel(const FunctionDecl *SyclKernel, QualType KernelNameType,
                   SourceLocation Loc, bool IsESIMD, bool IsUnnamedKernel,
                   int64_t ObjSize);

  /// Adds a kernel parameter descriptor to current kernel invocation
  /// descriptor.
  void addParamDesc(kernel_param_kind_t Kind, int Info, unsigned Offset);

  /// Signals that addition of parameter descriptors to current kernel
  /// invocation descriptor has finished.
  void endKernel();

  /// Registers a specialization constant to emit info for it into the header.
  void addSpecConstant(StringRef IDName, QualType IDType);

  /// Update the names of a kernel description based on its SyclKernel.
  void updateKernelNames(const FunctionDecl *SyclKernel, StringRef Name,
                         StringRef StableName) {
    auto Itr = llvm::find_if(KernelDescs, [SyclKernel](const KernelDesc &KD) {
      return KD.SyclKernel == SyclKernel;
    });

    assert(Itr != KernelDescs.end() && "Unknown kernel description");
    Itr->updateKernelNames(Name, StableName);
  }

  /// Signals that emission of __sycl_device_global_registration type and
  /// declaration of variable __sycl_device_global_registrar of this type in
  /// integration header is required.
  void addDeviceGlobalRegistration() {
    NeedToEmitDeviceGlobalRegistration = true;
  }

  /// Signals that emission of __sycl_host_pipe_registration type and
  /// declaration of variable __sycl_host_pipe_registrar of this type in
  /// integration header is required.
  void addHostPipeRegistration() { NeedToEmitHostPipeRegistration = true; }

private:
  // Kernel actual parameter descriptor.
  struct KernelParamDesc {
    // Represents a parameter kind.
    kernel_param_kind_t Kind = kind_last;
    // If Kind is kind_scalar or kind_struct, then
    //   denotes parameter size in bytes (includes padding for structs)
    // If Kind is kind_accessor
    //   denotes access target; possible access targets are defined in
    //   access/access.hpp
    int Info = 0;
    // Offset of the captured parameter value in the lambda or function object.
    unsigned Offset = 0;

    KernelParamDesc() = default;
  };

  // Kernel invocation descriptor
  struct KernelDesc {
    /// sycl_kernel function associated with this kernel.
    const FunctionDecl *SyclKernel;

    /// Kernel name.
    std::string Name;

    /// Kernel name type.
    QualType NameType;

    /// Kernel name with stable lambda name mangling
    std::string StableName;

    SourceLocation KernelLocation;

    /// Whether this kernel is an ESIMD one.
    bool IsESIMDKernel;

    /// Descriptor of kernel actual parameters.
    SmallVector<KernelParamDesc, 8> Params;

    // If we are in unnamed kernel/lambda mode AND this is one that the user
    // hasn't provided an explicit name for.
    bool IsUnnamedKernel;

    /// Size of the kernel object.
    int64_t ObjSize = 0;

    KernelDesc(const FunctionDecl *SyclKernel, QualType NameType,
               SourceLocation KernelLoc, bool IsESIMD, bool IsUnnamedKernel,
               int64_t ObjSize)
        : SyclKernel(SyclKernel), NameType(NameType), KernelLocation(KernelLoc),
          IsESIMDKernel(IsESIMD), IsUnnamedKernel(IsUnnamedKernel),
          ObjSize(ObjSize) {}

    void updateKernelNames(StringRef Name, StringRef StableName) {
      this->Name = Name.str();
      this->StableName = StableName.str();
    }
  };

  /// Returns the latest invocation descriptor started by
  /// SYCLIntegrationHeader::startKernel
  KernelDesc *getCurKernelDesc() {
    return KernelDescs.size() > 0 ? &KernelDescs[KernelDescs.size() - 1]
                                  : nullptr;
  }

private:
  /// Keeps invocation descriptors for each kernel invocation started by
  /// SYCLIntegrationHeader::startKernel
  SmallVector<KernelDesc, 4> KernelDescs;

  using SpecConstID = std::pair<QualType, std::string>;

  /// Keeps specialization constants met in the translation unit. Maps spec
  /// constant's ID type to generated unique name. Duplicates are removed at
  /// integration header emission time.
  llvm::SmallVector<SpecConstID, 4> SpecConsts;

  SemaSYCL &S;

  /// Keeps track of whether declaration of __sycl_device_global_registration
  /// type and __sycl_device_global_registrar variable are required to emit.
  bool NeedToEmitDeviceGlobalRegistration = false;

  /// Keeps track of whether declaration of __sycl_host_pipe_registration
  /// type and __sycl_host_pipe_registrar variable are required to emit.
  bool NeedToEmitHostPipeRegistration = false;
};

class SYCLIntegrationFooter {
public:
  SYCLIntegrationFooter(SemaSYCL &S) : S(S) {}
  bool emit(StringRef MainSrc);
  void addVarDecl(const VarDecl *VD);

private:
  bool emit(raw_ostream &O);
  SemaSYCL &S;
  llvm::SmallVector<const VarDecl *> GlobalVars;
  void emitSpecIDName(raw_ostream &O, const VarDecl *VD);
};

// Handles max_global_work_dim.
// Returns a OneArgResult value; EqualToOne means all argument values are
// equal to one, NotEqualToOne means at least one argument value is not
// equal to one, and Unknown means that at least one of the argument values
// could not be determined.
enum class OneArgResult { Unknown, EqualToOne, NotEqualToOne };
static inline OneArgResult areAllArgsOne(const Expr *Args[], size_t Count) {
  for (size_t Idx = 0; Idx < Count; ++Idx) {
    const Expr *Arg = Args[Idx];
    // Optional arguments are considered trivially one.
    if (!Arg)
      return OneArgResult::EqualToOne;
    const auto *CE = dyn_cast<ConstantExpr>(Args[Idx]);
    if (!CE)
      return OneArgResult::Unknown;
    if (CE->getResultAsAPSInt() != 1)
      return OneArgResult::NotEqualToOne;
  }
  return OneArgResult::EqualToOne;
}

class SemaSYCL : public SemaBase {
private:
  // We store SYCL Kernels here and handle separately -- which is a hack.
  // FIXME: It would be best to refactor this.
  llvm::SetVector<Decl *> SyclDeviceDecls;
  // SYCL integration header instance for current compilation unit this Sema
  // is associated with.
  std::unique_ptr<SYCLIntegrationHeader> SyclIntHeader;
  std::unique_ptr<SYCLIntegrationFooter> SyclIntFooter;

  // We need to store the list of the sycl_kernel functions and their associated
  // generated OpenCL Kernels so we can go back and re-name these after the
  // fact.
  llvm::SmallVector<std::pair<const FunctionDecl *, FunctionDecl *>>
      SyclKernelsToOpenCLKernels;

  // Used to suppress diagnostics during kernel construction, since these were
  // already emitted earlier. Diagnosing during Kernel emissions also skips the
  // useful notes that shows where the kernel was called.
  bool DiagnosingSYCLKernel = false;

  llvm::DenseSet<const FunctionDecl *> SYCLKernelFunctions;

public:
  SemaSYCL(Sema &S);

  void CheckSYCLKernelCall(FunctionDecl *CallerFunc,
                           ArrayRef<const Expr *> Args);
  void CheckSYCLScopeAttr(CXXRecordDecl *Decl);

  /// Creates a SemaDiagnosticBuilder that emits the diagnostic if the current
  /// context is "used as device code".
  ///
  /// - If CurLexicalContext is a kernel function or it is known that the
  ///   function will be emitted for the device, emits the diagnostics
  ///   immediately.
  /// - If CurLexicalContext is a function and we are compiling
  ///   for the device, but we don't know that this function will be codegen'ed
  ///   for device yet, creates a diagnostic which is emitted if and when we
  ///   realize that the function will be codegen'ed.
  ///
  /// Example usage:
  ///
  /// Diagnose __float128 type usage only from SYCL device code if the current
  /// target doesn't support it
  /// if (!S.Context.getTargetInfo().hasFloat128Type() &&
  ///     S.getLangOpts().SYCLIsDevice)
  ///   DiagIfDeviceCode(Loc, diag::err_type_unsupported) << "__float128";
  SemaDiagnosticBuilder DiagIfDeviceCode(
      SourceLocation Loc, unsigned DiagID,
      DeviceDiagnosticReason Reason = DeviceDiagnosticReason::Sycl |
                                      DeviceDiagnosticReason::Esimd);

  void deepTypeCheckForDevice(SourceLocation UsedAt,
                              llvm::DenseSet<QualType> Visited,
                              ValueDecl *DeclToCheck);

  void addSyclOpenCLKernel(const FunctionDecl *SyclKernel,
                           FunctionDecl *OpenCLKernel) {
    SyclKernelsToOpenCLKernels.emplace_back(SyclKernel, OpenCLKernel);
  }

  void addSyclDeviceDecl(Decl *d) { SyclDeviceDecls.insert(d); }
  llvm::SetVector<Decl *> &syclDeviceDecls() { return SyclDeviceDecls; }

  void addSYCLKernelFunction(const FunctionDecl *FD) {
    SYCLKernelFunctions.insert(FD);
  }

  /// Lazily creates and returns SYCL integration header instance.
  SYCLIntegrationHeader &getSyclIntegrationHeader() {
    if (SyclIntHeader == nullptr)
      SyclIntHeader = std::make_unique<SYCLIntegrationHeader>(*this);
    return *SyclIntHeader.get();
  }

  SYCLIntegrationFooter &getSyclIntegrationFooter() {
    if (SyclIntFooter == nullptr)
      SyclIntFooter = std::make_unique<SYCLIntegrationFooter>(*this);
    return *SyclIntFooter.get();
  }

  void addSyclVarDecl(VarDecl *VD) {
    if (getLangOpts().SYCLIsDevice && !getLangOpts().SYCLIntFooter.empty())
      getSyclIntegrationFooter().addVarDecl(VD);
  }

  bool hasSyclIntegrationHeader() { return SyclIntHeader != nullptr; }
  bool hasSyclIntegrationFooter() { return SyclIntFooter != nullptr; }

  enum SYCLRestrictKind {
    KernelGlobalVariable,
    KernelRTTI,
    KernelNonConstStaticDataVariable,
    KernelCallVirtualFunction,
    KernelUseExceptions,
    KernelCallRecursiveFunction,
    KernelCallFunctionPointer,
    KernelAllocateStorage,
    KernelUseAssembly,
    KernelCallVariadicFunction,
    KernelCallUndefinedFunction,
    KernelConstStaticVariable
  };

  bool isDeclAllowedInSYCLDeviceCode(const Decl *D);
  void checkSYCLDeviceVarDecl(VarDecl *Var);
  void copySYCLKernelAttrs(CXXMethodDecl *CallOperator);
  void ConstructOpenCLKernel(FunctionDecl *KernelCallerFunc, MangleContext &MC);
  void SetSYCLKernelNames();
  void MarkDevices();
  void ProcessFreeFunction(FunctionDecl *FD);

  /// Get the number of fields or captures within the parsed type.
  ExprResult ActOnSYCLBuiltinNumFieldsExpr(ParsedType PT);
  ExprResult BuildSYCLBuiltinNumFieldsExpr(SourceLocation Loc,
                                           QualType SourceTy);

  /// Get a value based on the type of the given field number so that callers
  /// can wrap it in a decltype() to get the actual type of the field.
  ExprResult ActOnSYCLBuiltinFieldTypeExpr(ParsedType PT, Expr *Idx);
  ExprResult BuildSYCLBuiltinFieldTypeExpr(SourceLocation Loc,
                                           QualType SourceTy, Expr *Idx);

  /// Get the number of base classes within the parsed type.
  ExprResult ActOnSYCLBuiltinNumBasesExpr(ParsedType PT);
  ExprResult BuildSYCLBuiltinNumBasesExpr(SourceLocation Loc,
                                          QualType SourceTy);

  /// Get a value based on the type of the given base number so that callers
  /// can wrap it in a decltype() to get the actual type of the base class.
  ExprResult ActOnSYCLBuiltinBaseTypeExpr(ParsedType PT, Expr *Idx);
  ExprResult BuildSYCLBuiltinBaseTypeExpr(SourceLocation Loc, QualType SourceTy,
                                          Expr *Idx);

  bool checkAllowedSYCLInitializer(VarDecl *VD);

  /// Finishes analysis of the deferred functions calls that may be not
  /// properly declared for device compilation.
  void finalizeSYCLDelayedAnalysis(const FunctionDecl *Caller,
                                   const FunctionDecl *Callee,
                                   SourceLocation Loc,
                                   DeviceDiagnosticReason Reason);

  void performSYCLDelayedAttributesAnalaysis(const FunctionDecl *FD);

  /// Tells whether given variable is a SYCL explicit SIMD extension's "private
  /// global" variable - global variable in the private address space.
  bool isSYCLEsimdPrivateGlobal(VarDecl *VDecl) {
    return getLangOpts().SYCLIsDevice && VDecl->hasAttr<SYCLSimdAttr>() &&
           VDecl->hasGlobalStorage() &&
           (VDecl->getType().getAddressSpace() == LangAS::sycl_private);
  }

  template <typename AttrTy>
  static bool isTypeDecoratedWithDeclAttribute(QualType Ty) {
    const CXXRecordDecl *RecTy = Ty->getAsCXXRecordDecl();
    if (!RecTy)
      return false;

    if (RecTy->hasAttr<AttrTy>())
      return true;

    if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(RecTy)) {
      ClassTemplateDecl *Template = CTSD->getSpecializedTemplate();
      if (CXXRecordDecl *RD = Template->getTemplatedDecl())
        return RD->hasAttr<AttrTy>();
    }
    return false;
  }

  /// Check whether \p Ty corresponds to a SYCL type of name \p TypeName.
  static bool isSyclType(QualType Ty, SYCLTypeAttr::SYCLType TypeName);

  ExprResult BuildUniqueStableIdExpr(SourceLocation OpLoc,
                                     SourceLocation LParen,
                                     SourceLocation RParen, Expr *E);
  ExprResult ActOnUniqueStableIdExpr(SourceLocation OpLoc,
                                     SourceLocation LParen,
                                     SourceLocation RParen, Expr *E);
  ExprResult BuildUniqueStableNameExpr(SourceLocation OpLoc,
                                       SourceLocation LParen,
                                       SourceLocation RParen,
                                       TypeSourceInfo *TSI);
  ExprResult ActOnUniqueStableNameExpr(SourceLocation OpLoc,
                                       SourceLocation LParen,
                                       SourceLocation RParen,
                                       ParsedType ParsedTy);

  void handleKernelAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLTypeAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLDeviceAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLDeviceIndirectlyCallableAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLGlobalVarAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLRegisterNumAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelESimdVectorizeAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLDeviceHasAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLUsesAspectsAttr(Decl *D, const ParsedAttr &AL);
  void handleLaunchBoundsAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLWorkGroupSizeHintAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLReqdWorkGroupSizeAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelMaxWorkGroupSizeAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelMinWorkGroupsPerComputeUnitAttr(Decl *D,
                                                      const ParsedAttr &AL);
  void handleSYCLIntelMaxWorkGroupsPerMultiprocessorAttr(Decl *D,
                                                         const ParsedAttr &AL);
  void handleIntelReqdSubGroupSizeAttr(Decl *D, const ParsedAttr &AL);
  void handleIntelNamedSubGroupSizeAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelNumSimdWorkItemsAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelSchedulerTargetFmaxMhzAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelMaxGlobalWorkDimAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelNoGlobalWorkOffsetAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelUseStallEnableClustersAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelLoopFuseAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelInitiationIntervalAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelDoublePumpAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelSinglePumpAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelMemoryAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelRegisterAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelBankWidthAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelNumBanksAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelPrivateCopiesAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelMaxReplicatesAttr(Decl *D, const ParsedAttr &AL);
  void handleIntelSimpleDualPortAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelMergeAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelBankBitsAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelForcePow2DepthAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelPipeIOAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelMaxConcurrencyAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLAddIRAttributesFunctionAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLAddIRAttributesKernelParameterAttr(Decl *D,
                                                    const ParsedAttr &AL);
  void handleSYCLAddIRAttributesGlobalVariableAttr(Decl *D,
                                                   const ParsedAttr &AL);
  void handleSYCLAddIRAnnotationsMemberAttr(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelMaxWorkGroupSize(Decl *D, const ParsedAttr &AL);
  void handleSYCLIntelMinWorkGroupsPerComputeUnit(Decl *D,
                                                  const ParsedAttr &AL);
  void handleSYCLIntelMaxWorkGroupsPerMultiprocessor(Decl *D,
                                                     const ParsedAttr &AL);
  void handleSYCLScopeAttr(Decl *D, const ParsedAttr &AL);

  void checkSYCLAddIRAttributesFunctionAttrConflicts(Decl *D);

  SYCLReqdWorkGroupSizeAttr *
  mergeSYCLReqdWorkGroupSizeAttr(Decl *D, const SYCLReqdWorkGroupSizeAttr &A);
  SYCLIntelNumSimdWorkItemsAttr *
  mergeSYCLIntelNumSimdWorkItemsAttr(Decl *D,
                                     const SYCLIntelNumSimdWorkItemsAttr &A);
  SYCLIntelInitiationIntervalAttr *mergeSYCLIntelInitiationIntervalAttr(
      Decl *D, const SYCLIntelInitiationIntervalAttr &A);
  SYCLIntelSchedulerTargetFmaxMhzAttr *mergeSYCLIntelSchedulerTargetFmaxMhzAttr(
      Decl *D, const SYCLIntelSchedulerTargetFmaxMhzAttr &A);
  SYCLIntelMaxGlobalWorkDimAttr *
  mergeSYCLIntelMaxGlobalWorkDimAttr(Decl *D,
                                     const SYCLIntelMaxGlobalWorkDimAttr &A);
  SYCLIntelMinWorkGroupsPerComputeUnitAttr *
  mergeSYCLIntelMinWorkGroupsPerComputeUnitAttr(
      Decl *D, const SYCLIntelMinWorkGroupsPerComputeUnitAttr &A);
  SYCLIntelMaxWorkGroupsPerMultiprocessorAttr *
  mergeSYCLIntelMaxWorkGroupsPerMultiprocessorAttr(
      Decl *D, const SYCLIntelMaxWorkGroupsPerMultiprocessorAttr &A);
  SYCLIntelLoopFuseAttr *
  mergeSYCLIntelLoopFuseAttr(Decl *D, const SYCLIntelLoopFuseAttr &A);
  SYCLIntelESimdVectorizeAttr *
  mergeSYCLIntelESimdVectorizeAttr(Decl *D,
                                   const SYCLIntelESimdVectorizeAttr &A);
  SYCLIntelNoGlobalWorkOffsetAttr *mergeSYCLIntelNoGlobalWorkOffsetAttr(
      Decl *D, const SYCLIntelNoGlobalWorkOffsetAttr &A);
  SYCLIntelBankWidthAttr *
  mergeSYCLIntelBankWidthAttr(Decl *D, const SYCLIntelBankWidthAttr &A);
  SYCLIntelNumBanksAttr *
  mergeSYCLIntelNumBanksAttr(Decl *D, const SYCLIntelNumBanksAttr &A);
  SYCLIntelMaxReplicatesAttr *
  mergeSYCLIntelMaxReplicatesAttr(Decl *D, const SYCLIntelMaxReplicatesAttr &A);
  SYCLIntelForcePow2DepthAttr *
  mergeSYCLIntelForcePow2DepthAttr(Decl *D,
                                   const SYCLIntelForcePow2DepthAttr &A);
  SYCLIntelPipeIOAttr *mergeSYCLIntelPipeIOAttr(Decl *D,
                                                const SYCLIntelPipeIOAttr &A);
  SYCLIntelMaxConcurrencyAttr *
  mergeSYCLIntelMaxConcurrencyAttr(Decl *D,
                                   const SYCLIntelMaxConcurrencyAttr &A);
  SYCLAddIRAttributesFunctionAttr *mergeSYCLAddIRAttributesFunctionAttr(
      Decl *D, const SYCLAddIRAttributesFunctionAttr &A);
  SYCLAddIRAttributesKernelParameterAttr *
  mergeSYCLAddIRAttributesKernelParameterAttr(
      Decl *D, const SYCLAddIRAttributesKernelParameterAttr &A);
  SYCLAddIRAttributesGlobalVariableAttr *
  mergeSYCLAddIRAttributesGlobalVariableAttr(
      Decl *D, const SYCLAddIRAttributesGlobalVariableAttr &A);
  SYCLAddIRAnnotationsMemberAttr *
  mergeSYCLAddIRAnnotationsMemberAttr(Decl *D,
                                      const SYCLAddIRAnnotationsMemberAttr &A);
  SYCLDeviceHasAttr *mergeSYCLDeviceHasAttr(Decl *D,
                                            const SYCLDeviceHasAttr &A);
  SYCLUsesAspectsAttr *mergeSYCLUsesAspectsAttr(Decl *D,
                                                const SYCLUsesAspectsAttr &A);
  SYCLTypeAttr *mergeSYCLTypeAttr(Decl *D, const AttributeCommonInfo &CI,
                                  SYCLTypeAttr::SYCLType TypeName);
  SYCLWorkGroupSizeHintAttr *
  mergeSYCLWorkGroupSizeHintAttr(Decl *D, const SYCLWorkGroupSizeHintAttr &A);
  SYCLIntelMaxWorkGroupSizeAttr *
  mergeSYCLIntelMaxWorkGroupSizeAttr(Decl *D,
                                     const SYCLIntelMaxWorkGroupSizeAttr &A);
  IntelReqdSubGroupSizeAttr *
  mergeIntelReqdSubGroupSizeAttr(Decl *D, const IntelReqdSubGroupSizeAttr &A);
  IntelNamedSubGroupSizeAttr *
  mergeIntelNamedSubGroupSizeAttr(Decl *D, const IntelNamedSubGroupSizeAttr &A);

  static OffloadArch getOffloadArch(const TargetInfo &TI);
  static bool hasDependentExpr(Expr **Exprs, const size_t ExprsSize);

  /// Emit a diagnostic about the given attribute having a deprecated name, and
  /// also emit a fixit hint to generate the new attribute name.
  void diagnoseDeprecatedAttribute(const ParsedAttr &A, StringRef NewScope,
                                   StringRef NewName);

  /// Diagnoses an attribute in the 'intelfpga' namespace and suggests using
  /// the attribute in the 'intel' namespace instead.
  void checkDeprecatedSYCLAttributeSpelling(const ParsedAttr &A,
                                            StringRef NewName = "");

  bool checkValidFPGAMemoryAttributesVar(Decl *D);

  bool checkWorkGroupSize(const Expr *NSWIValue, const Expr *RWGSXDim,
                          const Expr *RWGSYDim, const Expr *RWGSZDim);
  bool checkMaxAllowedWorkGroupSize(const Expr *RWGSXDim, const Expr *RWGSYDim,
                                    const Expr *RWGSZDim, const Expr *MWGSXDim,
                                    const Expr *MWGSYDim, const Expr *MWGSZDim);
  bool areInvalidWorkGroupSizeAttrs(const Expr *MGValue, const Expr *XDim,
                                    const Expr *YDim, const Expr *ZDim);

  // If the declaration has a SYCLIntelMaxWorkGroupSizeAttr or
  // ReqdWorkGroupSizeAttr, check to see if they hold equal values
  // (1, 1, 1). Returns true if diagnosed.
  template <typename AttrTy>
  bool checkWorkGroupSizeAttrExpr(Decl *D, const AttributeCommonInfo &AL) {
    if (const auto *A = D->getAttr<AttrTy>()) {
      const Expr *Args[3] = {A->getXDim(), A->getYDim(), A->getZDim()};
      if (OneArgResult::NotEqualToOne == areAllArgsOne(Args, 3)) {
        Diag(A->getLocation(), diag::err_sycl_x_y_z_arguments_must_be_one)
            << A << AL;
        return true;
      }
    }
    return false;
  }

  bool anyWorkGroupSizesDiffer(const Expr *LHSXDim, const Expr *LHSYDim,
                               const Expr *LHSZDim, const Expr *RHSXDim,
                               const Expr *RHSYDim, const Expr *RHSZDim);
  bool allWorkGroupSizesSame(const Expr *LHSXDim, const Expr *LHSYDim,
                             const Expr *LHSZDim, const Expr *RHSXDim,
                             const Expr *RHSYDim, const Expr *RHSZDim);

  void addSYCLIntelPipeIOAttr(Decl *D, const AttributeCommonInfo &CI, Expr *E);
  void addSYCLDeviceHasAttr(Decl *D, const AttributeCommonInfo &CI,
                            Expr **Exprs, unsigned Size);
  void addSYCLUsesAspectsAttr(Decl *D, const AttributeCommonInfo &CI,
                              Expr **Exprs, unsigned Size);
  void addSYCLIntelLoopFuseAttr(Decl *D, const AttributeCommonInfo &CI,
                                Expr *E);
  void addSYCLIntelNumSimdWorkItemsAttr(Decl *D, const AttributeCommonInfo &CI,
                                        Expr *E);
  void addSYCLIntelSchedulerTargetFmaxMhzAttr(Decl *D,
                                              const AttributeCommonInfo &CI,
                                              Expr *E);
  void addSYCLIntelNoGlobalWorkOffsetAttr(Decl *D,
                                          const AttributeCommonInfo &CI,
                                          Expr *E);
  void addSYCLIntelMaxGlobalWorkDimAttr(Decl *D, const AttributeCommonInfo &CI,
                                        Expr *E);
  void addSYCLIntelMinWorkGroupsPerComputeUnitAttr(
      Decl *D, const AttributeCommonInfo &CI, Expr *E);
  void addSYCLIntelMaxWorkGroupsPerMultiprocessorAttr(
      Decl *D, const AttributeCommonInfo &CI, Expr *E);
  void addSYCLIntelMaxConcurrencyAttr(Decl *D, const AttributeCommonInfo &CI,
                                      Expr *E);
  void addSYCLIntelPrivateCopiesAttr(Decl *D, const AttributeCommonInfo &CI,
                                     Expr *E);
  void addSYCLIntelMaxReplicatesAttr(Decl *D, const AttributeCommonInfo &CI,
                                     Expr *E);
  void addSYCLIntelInitiationIntervalAttr(Decl *D,
                                          const AttributeCommonInfo &CI,
                                          Expr *E);
  void addSYCLIntelESimdVectorizeAttr(Decl *D, const AttributeCommonInfo &CI,
                                      Expr *E);
  void addSYCLAddIRAttributesFunctionAttr(Decl *D,
                                          const AttributeCommonInfo &CI,
                                          MutableArrayRef<Expr *> Args);
  void addSYCLAddIRAttributesKernelParameterAttr(Decl *D,
                                                 const AttributeCommonInfo &CI,
                                                 MutableArrayRef<Expr *> Args);
  void addSYCLAddIRAttributesGlobalVariableAttr(Decl *D,
                                                const AttributeCommonInfo &CI,
                                                MutableArrayRef<Expr *> Args);
  void addSYCLAddIRAnnotationsMemberAttr(Decl *D, const AttributeCommonInfo &CI,
                                         MutableArrayRef<Expr *> Args);
  void addSYCLWorkGroupSizeHintAttr(Decl *D, const AttributeCommonInfo &CI,
                                    Expr *XDim, Expr *YDim, Expr *ZDim);
  void addSYCLReqdWorkGroupSizeAttr(Decl *D, const AttributeCommonInfo &CI,
                                    Expr *XDim, Expr *YDim, Expr *ZDim);
  void addSYCLIntelMaxWorkGroupSizeAttr(Decl *D, const AttributeCommonInfo &CI,
                                        Expr *XDim, Expr *YDim, Expr *ZDim);
  void addSYCLIntelForcePow2DepthAttr(Decl *D, const AttributeCommonInfo &CI,
                                      Expr *E);
  void addSYCLIntelBankWidthAttr(Decl *D, const AttributeCommonInfo &CI,
                                 Expr *E);
  void addSYCLIntelNumBanksAttr(Decl *D, const AttributeCommonInfo &CI,
                                Expr *E);
  void addSYCLIntelBankBitsAttr(Decl *D, const AttributeCommonInfo &CI,
                                Expr **Exprs, unsigned Size);
  void addIntelReqdSubGroupSizeAttr(Decl *D, const AttributeCommonInfo &CI,
                                    Expr *E);
  void handleKernelEntryPointAttr(Decl *D, const ParsedAttr &AL);
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMASYCL_H
