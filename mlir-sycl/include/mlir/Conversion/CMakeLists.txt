set(LLVM_TARGET_DEFINITIONS SYCLPasses.td)
mlir_tablegen(SYCLPasses.h.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(MLIRSYCLConversionPassIncGen)
