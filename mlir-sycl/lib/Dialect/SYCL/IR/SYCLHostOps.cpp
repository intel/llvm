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
  // NOTE: This definition assumes only the first (`this`) argument is written
  // to and the remaining arguments, if they are memory arguments, are only read
  // from. This is optimistic assumption, but enables better analysis and is
  // true for the types and properties we are most interested in right now.
  auto *defaultResource = SideEffects::DefaultResource::get();
  // The `this` argument will always be written to
  effects.emplace_back(MemoryEffects::Write::get(), getDst(), defaultResource);
  // The rest of the arguments will be scalar or read from
  for (auto value : getArgs())
    if (isa<MemRefType, LLVM::LLVMPointerType>(value.getType()))
      effects.emplace_back(MemoryEffects::Read::get(), value, defaultResource);
}
