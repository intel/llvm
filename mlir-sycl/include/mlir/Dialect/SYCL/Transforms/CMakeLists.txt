set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name SYCL)
add_public_tablegen_target(MLIRSYCLTransformsIncGen)
add_dependencies(mlir-headers MLIRSYCLTransformsIncGen)

add_mlir_sycl_doc(Passes SYCLPasses ./ -gen-pass-doc)
