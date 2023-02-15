#ifndef MLIR_CONVERSION_POLYGEISTTOLLVM_POLYGEISTTOLLVMPASS_H
#define MLIR_CONVERSION_POLYGEISTTOLLVM_POLYGEISTTOLLVMPASS_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class LowerToLLVMOptions;

#define GEN_PASS_DECL_CONVERTPOLYGEISTTOLLVM
#include "mlir/Conversion/PolygeistPasses.h.inc"
#undef GEN_PASS_DECL_CONVERTPOLYGEISTTOLLVM

std::unique_ptr<Pass> createConvertPolygeistToLLVMPass();
std::unique_ptr<Pass>
createConvertPolygeistToLLVMPass(const LowerToLLVMOptions &options);
} // namespace mlir

#endif // MLIR_CONVERSION_POLYGEISTTOLLVM_POLYGEISTTOLLVMPASS_H
