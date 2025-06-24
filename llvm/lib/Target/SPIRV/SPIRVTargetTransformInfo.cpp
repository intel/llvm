#include "SPIRVTargetTransformInfo.h"
#include "llvm/IR/IntrinsicsSPIRV.h"

namespace llvm {

bool SPIRVTTIImpl::collectFlatAddressOperands(SmallVectorImpl<int> &OpIndexes,
                                              Intrinsic::ID IID) const {
  switch (IID) {
  case Intrinsic::spv_generic_cast_to_ptr_explicit:
    OpIndexes.push_back(0);
    return true;
  default:
    return false;
  }
}

Value *SPIRVTTIImpl::rewriteIntrinsicWithAddressSpace(IntrinsicInst *II,
                                                      Value *OldV,
                                                      Value *NewV) const {
  auto IntrID = II->getIntrinsicID();
  switch (IntrID) {
  case Intrinsic::spv_generic_cast_to_ptr_explicit: {
    unsigned NewAS = NewV->getType()->getPointerAddressSpace();
    unsigned DstAS = II->getType()->getPointerAddressSpace();
    return NewAS == DstAS ? NewV
                          : ConstantPointerNull::get(
                                PointerType::get(NewV->getContext(), DstAS));
  }
  default:
    return nullptr;
  }
}

} // namespace llvm
