set(LLVM_TARGET_DEFINITIONS PolygeistPasses.td)
mlir_tablegen(PolygeistPasses.h.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(MLIRPolygeistConversionPassIncGen)
