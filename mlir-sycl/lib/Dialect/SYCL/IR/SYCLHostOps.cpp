#include "mlir/Dialect/SYCL/IR/SYCLOps.h"

using namespace mlir;
using namespace mlir::sycl;

LogicalResult SYCLHostConstructorOp::verify() {
  Type type = getType().getValue();
  if (!isSYCLType(type))
    return emitOpError("expecting a sycl type as constructed type. Got ")
           << type;
  return success();
}

void SYCLHostConstructorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // NOTE: This definition conservatively assumes that all (pointer) arguments
  // are written to. This is definitely true for the first argument ('this'
  // pointer), but a pessimistic assumption for the remaining arguments.
  auto *defaultResource = SideEffects::DefaultResource::get();
  // The `this` argument will always be written to
  effects.emplace_back(MemoryEffects::Write::get(), getDst(), defaultResource);
  // For the remaining non-scalar arguments we also assume they are written.
  for (auto value : getArgs()) {
    if (isa<MemRefType, LLVM::LLVMPointerType>(value.getType())) {
      effects.emplace_back(MemoryEffects::Write::get(), value, defaultResource);
    }
  }
}
