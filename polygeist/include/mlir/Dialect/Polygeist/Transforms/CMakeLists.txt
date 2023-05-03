set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Polygeist)
add_public_tablegen_target(MLIRPolygeistPassIncGen)
add_dependencies(mlir-headers MLIRPolygeistPassIncGen)

add_polygeist_doc(Passes PolygeistPasses ./ -gen-pass-doc)
