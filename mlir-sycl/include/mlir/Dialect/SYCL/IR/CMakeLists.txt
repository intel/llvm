add_mlir_dialect(SYCLOps sycl)
add_mlir_doc(SYCLBase SYCLBase Dialects/ -gen-dialect-doc)
add_mlir_doc(SYCLOps SYCLOps Dialects/ -gen-op-doc)
add_mlir_doc(SYCLTypes SYCLTypes Dialects/ -gen-typedef-doc)

add_mlir_interface(SYCLMethodOpInterface)
add_mlir_doc(SYCLMethodOpInterface SYCLMethodOpInterface Dialects/ -gen-op-interface-docs)

set(LLVM_TARGET_DEFINITIONS SYCLAttributes.td)
mlir_tablegen(SYCLAttributes.h.inc -gen-enum-decls)
mlir_tablegen(SYCLAttributes.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRSYCLAttributesIncGen)
add_dependencies(mlir-headers MLIRSYCLAttributesIncGen)
add_mlir_doc(SYCLAttributes SYCLAttributes Dialects/ -gen-attrdef-doc)
