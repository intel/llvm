#include "llvm/SYCLLowerIR/LowerSPIRVAddressSpaceCasts.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

namespace llvm {

static bool visitCallInst(CallInst *CI) {
  Function *Callee = CI->getCalledFunction();
  if (!Callee || !Callee->isDeclaration())
    return false;

  // We are looking for this pattern where an LLVM addrspace cast
  // is followed by a call to a SPIR-V intrinsic:
  // %0 = addrspacecast ptr addrspace(1) %src to ptr addrspace(4)
  // %1 = call ptr addrspace(1) @cast(ptr addrspace(4) %0, ...)
  if (Callee->getName() != "_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi" &&
      Callee->getName() != "_Z40__spirv_GenericCastToPtrExplicit_ToLocalPvi" &&
      Callee->getName() != "_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePvi")
    return false;

  // Check if the first argument is an addrspacecast instruction
  Value *Arg = CI->getArgOperand(0);
  auto *DstType = cast<PointerType>(CI->getType());
  if (auto *ASC = dyn_cast<AddrSpaceCastInst>(Arg)) {
    Value *ReplacedValue = nullptr;
    if (ASC->getSrcAddressSpace() == DstType->getAddressSpace()) {
      ReplacedValue = ASC->getOperand(0);

      // Check if the addrspacecast argument is a null pointer constant,
      // and if so, we can replace the call with a null pointer constant.
      if (isa<ConstantPointerNull>(ReplacedValue))
        ReplacedValue = ConstantPointerNull::get(PointerType::get(
            DstType->getContext(), DstType->getAddressSpace()));

    } else {
      assert(ASC->getSrcAddressSpace() != 4);
      ReplacedValue = ConstantPointerNull::get(
          PointerType::get(ASC->getContext(), DstType->getAddressSpace()));
    }
    // Replace the call's result with the source of the addrspacecast
    CI->replaceAllUsesWith(ReplacedValue);
    CI->eraseFromParent();
    return true;
  }

  // Check if the first argument is a null pointer constant
  if (isa<ConstantPointerNull>(Arg)) {
    // Replace the call's result with a null pointer constant of the
    // destination address space
    auto *NullPtr = ConstantPointerNull::get(
        PointerType::get(DstType->getContext(), DstType->getAddressSpace()));
    CI->replaceAllUsesWith(NullPtr);
    CI->eraseFromParent();
    return true;
  }

  return false;
}

static bool visitAddrSpaceCastInst(AddrSpaceCastInst *ASC) {
  if (ASC->getDestAddressSpace() == 4 &&
      isa<ConstantPointerNull>(ASC->getOperand(0))) {
    // If the source is a null pointer, we can replace the addrspacecast
    // with a null pointer constant of the destination address space.
    auto *NullPtr = ConstantPointerNull::get(
        PointerType::get(ASC->getContext(), ASC->getDestAddressSpace()));
    ASC->replaceAllUsesWith(NullPtr);
    ASC->eraseFromParent();
    return true;
  }
  return false;
}

PreservedAnalyses
LowerSPIRVAddressSpaceCastsPass::run(Function &F, FunctionAnalysisManager &AM) {
  bool Changed = false;
  for (auto &BB : F) {
    for (auto It = BB.begin(), E = BB.end(); It != E;) {
      Instruction *Inst = &*(It++);
      if (!isa<CallInst>(Inst) && !isa<AddrSpaceCastInst>(Inst))
        continue;

      if (auto CI = dyn_cast<CallInst>(Inst))
        Changed = visitCallInst(CI) || Changed;
      else if (auto ASC = dyn_cast<AddrSpaceCastInst>(Inst))
        Changed = visitAddrSpaceCastInst(ASC) || Changed;
    }
  }
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace llvm
