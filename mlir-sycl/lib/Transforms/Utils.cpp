#include "Utils.h"

#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define INLINER_DEBUG_TYPE "inlining"
#define SYCL_METHOD_TO_SYCL_CALL_DEBUG_TYPE "sycl-method-to-sycl-call"

/// Custom definition to print messages coming from clients.
#define LLVM_DEBUG(X)                                                          \
  DEBUG_WITH_TYPE(INLINER_DEBUG_TYPE, X);                                      \
  DEBUG_WITH_TYPE(SYCL_METHOD_TO_SYCL_CALL_DEBUG_TYPE, X)

using namespace mlir;
using namespace sycl;

static Value adaptArgumentForSYCLCall(OpBuilder &rewriter, Location loc,
                                      Value original, Type targetType) {
  if (original.getType() == targetType)
    return original;

  const auto mt = targetType.cast<MemRefType>();
  const auto thisType = original.getType().cast<MemRefType>();
  const llvm::ArrayRef<int64_t> targetShape = mt.getShape();
  const Type targetElementType = mt.getElementType();
  const unsigned targetMemSpace = mt.getMemorySpaceAsInt();

  assert(mt.getLayout() == thisType.getLayout() && "Invalid layout mismatch");

  if (targetShape != thisType.getShape()) {
    original = rewriter.create<memref::CastOp>(
        loc,
        MemRefType::get(targetShape, thisType.getElementType(),
                        thisType.getLayout(), thisType.getMemorySpace()),
        original);
    LLVM_DEBUG(llvm::dbgs() << "  MemRef cast needed: " << original << "\n");
  }

  if (thisType.getMemorySpaceAsInt() != targetMemSpace) {
    original = rewriter.create<memref::MemorySpaceCastOp>(
        loc,
        MemRefType::get(targetShape, thisType.getElementType(),
                        thisType.getLayout().getAffineMap(), targetMemSpace),
        original);
    LLVM_DEBUG(llvm::dbgs()
               << "  Address space cast needed: " << original << "\n");
  }

  if (thisType.getElementType() != targetElementType) {
    original = rewriter.create<SYCLCastOp>(loc, targetType, original);
    LLVM_DEBUG(llvm::dbgs() << "  sycl.cast inserted: " << original << "\n");
  }

  return original;
}

SmallVector<Value>
mlir::sycl::adaptArgumentsForSYCLCall(OpBuilder &builder,
                                      SYCLMethodOpInterface method) {
  SmallVector<Value> transformed;
  transformed.reserve(method->getNumOperands());
  const auto loc = method.getLoc();
  const auto targetTypes = method.getArgumentTypes();
  std::transform(method->operand_begin(), method->operand_end(),
                 targetTypes.begin(), std::back_inserter(transformed),
                 [&](auto Val, auto Ty) {
                   return adaptArgumentForSYCLCall(builder, loc, Val, Ty);
                 });
  assert(ValueRange{transformed}.getTypes() == targetTypes);
  return transformed;
}
