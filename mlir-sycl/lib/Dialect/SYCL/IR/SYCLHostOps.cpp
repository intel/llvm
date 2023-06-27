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
