add_mlir_dialect(PolygeistOps polygeist)
add_mlir_doc(PolygeistDialect -gen-dialect-doc PolygeistDialect Polygeist/)
add_mlir_doc(PolygeistOps -gen-op-doc PolygeistOps Polygeist/)

add_subdirectory(Passes)