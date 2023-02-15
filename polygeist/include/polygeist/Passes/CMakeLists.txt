set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name polygeist)
add_public_tablegen_target(MLIRPolygeistPassIncGen)

add_mlir_doc(Passes PolygeistPasses ./ -gen-pass-doc)
