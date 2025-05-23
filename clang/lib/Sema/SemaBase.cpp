#include "clang/Sema/SemaBase.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaCUDA.h"

namespace clang {

SemaBase::SemaBase(Sema &S) : SemaRef(S) {}

ASTContext &SemaBase::getASTContext() const { return SemaRef.Context; }
DiagnosticsEngine &SemaBase::getDiagnostics() const { return SemaRef.Diags; }
const LangOptions &SemaBase::getLangOpts() const { return SemaRef.LangOpts; }
DeclContext *SemaBase::getCurContext() const { return SemaRef.CurContext; }

SemaBase::ImmediateDiagBuilder::~ImmediateDiagBuilder() {
  // If we aren't active, there is nothing to do.
  if (!isActive())
    return;

  // Otherwise, we need to emit the diagnostic. First clear the diagnostic
  // builder itself so it won't emit the diagnostic in its own destructor.
  //
  // This seems wasteful, in that as written the DiagnosticBuilder dtor will
  // do its own needless checks to see if the diagnostic needs to be
  // emitted. However, because we take care to ensure that the builder
  // objects never escape, a sufficiently smart compiler will be able to
  // eliminate that code.
  Clear();

  // Dispatch to Sema to emit the diagnostic.
  SemaRef.EmitDiagnostic(DiagID, *this);
}

PartialDiagnostic SemaBase::PDiag(unsigned DiagID) {
  return PartialDiagnostic(DiagID, SemaRef.Context.getDiagAllocator());
}

const SemaBase::SemaDiagnosticBuilder &
operator<<(const SemaBase::SemaDiagnosticBuilder &Diag,
           const PartialDiagnostic &PD) {
  if (Diag.ImmediateDiag)
    PD.Emit(*Diag.ImmediateDiag);
  else if (Diag.PartialDiagId)
    Diag.getDeviceDeferredDiags()[Diag.Fn][*Diag.PartialDiagId]
        .getDiag()
        .second = PD;
  return Diag;
}

void SemaBase::SemaDiagnosticBuilder::AddFixItHint(
    const FixItHint &Hint) const {
  if (ImmediateDiag)
    ImmediateDiag->AddFixItHint(Hint);
  else if (PartialDiagId)
    getDeviceDeferredDiags()[Fn][*PartialDiagId].getDiag().second.AddFixItHint(
        Hint);
}

SemaBase::SemaDiagnosticBuilder::DeferredDiagnosticsType &
SemaBase::SemaDiagnosticBuilder::getDeviceDeferredDiags() const {
  if (S.InConstexprVarInit)
    return S.MaybeDeviceDeferredDiags;
  return S.DeviceDeferredDiags;
}

Sema::SemaDiagnosticBuilder SemaBase::Diag(SourceLocation Loc, unsigned DiagID,
                                           bool DeferHint) {
  bool IsError =
      getDiagnostics().getDiagnosticIDs()->isDefaultMappingAsError(DiagID);
  bool ShouldDefer = getLangOpts().CUDA && getLangOpts().GPUDeferDiag &&
                     DiagnosticIDs::isDeferrable(DiagID) &&
                     (DeferHint || SemaRef.DeferDiags || !IsError);
  auto SetIsLastErrorImmediate = [&](bool Flag) {
    if (IsError)
      SemaRef.IsLastErrorImmediate = Flag;
  };
  if (!ShouldDefer) {
    SetIsLastErrorImmediate(true);
    return SemaDiagnosticBuilder(SemaDiagnosticBuilder::K_Immediate, Loc,
                                 DiagID, SemaRef.getCurFunctionDecl(), SemaRef,
                                 DeviceDiagnosticReason::All);
  }

  SemaDiagnosticBuilder DB = getLangOpts().CUDAIsDevice
                                 ? SemaRef.CUDA().DiagIfDeviceCode(Loc, DiagID)
                                 : SemaRef.CUDA().DiagIfHostCode(Loc, DiagID);
  SetIsLastErrorImmediate(DB.isImmediate());
  return DB;
}

Sema::SemaDiagnosticBuilder SemaBase::Diag(SourceLocation Loc,
                                           const PartialDiagnostic &PD,
                                           bool DeferHint) {
  return Diag(Loc, PD.getDiagID(), DeferHint) << PD;
}

SemaBase::SemaDiagnosticBuilder SemaBase::DiagCompat(SourceLocation Loc,
                                                     unsigned CompatDiagId,
                                                     bool DeferHint) {
  return Diag(Loc,
              DiagnosticIDs::getCXXCompatDiagId(getLangOpts(), CompatDiagId),
              DeferHint);
}
} // namespace clang
