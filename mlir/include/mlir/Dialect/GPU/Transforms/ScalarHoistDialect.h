//===- ScalarHoistDialect.h - scalar_hoist dialect ----------------*- C++-*-===//
//
// Minimal dialect for patent demonstration: wraps host-side scalar
// precomputation in explicit dialect ops visible in IR dumps.
//
// Uses allowUnknownOperations() so we can create ops like
// "scalar_hoist.precompute" and "scalar_hoist.yield" without TableGen.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_TRANSFORMS_SCALARHOISTDIALECT_H
#define MLIR_DIALECT_GPU_TRANSFORMS_SCALARHOISTDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace scalar_hoist {

class ScalarHoistDialect : public Dialect {
public:
  explicit ScalarHoistDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<ScalarHoistDialect>()) {
    // Accept any op under "scalar_hoist.*" without formal registration.
    // Ops print in generic format: "scalar_hoist.precompute"(%x) ({...})
    allowUnknownOperations();
  }

  static StringRef getDialectNamespace() { return "scalar_hoist"; }
};

} // namespace scalar_hoist
} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::scalar_hoist::ScalarHoistDialect)

#endif // MLIR_DIALECT_GPU_TRANSFORMS_SCALARHOISTDIALECT_H
