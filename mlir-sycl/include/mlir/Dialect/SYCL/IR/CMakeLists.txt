add_mlir_dialect(SYCLOps sycl)
add_mlir_doc(SYCLBase SYCLBase Dialects/ -gen-dialect-doc)
add_mlir_doc(SYCLOps SYCLOps Dialects/ -gen-op-doc)
add_mlir_doc(SYCLTypes SYCLTypes Dialects/ -gen-typedef-doc)

add_mlir_interface(SYCLOpInterfaces)
add_mlir_doc(SYCLOpInterfaces SYCLOpInterfaces Dialects/ -gen-op-interface-docs)

set(LLVM_TARGET_DEFINITIONS SYCLOpAttributes.td)
mlir_tablegen(SYCLOpAttributes.h.inc -gen-enum-decls)
mlir_tablegen(SYCLOpAttributes.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRSYCLOpAttributeIncGen)
add_dependencies(mlir-headers MLIRSYCLOpAttributeIncGen)
add_mlir_doc(SYCLOpAttributes SYCLOpAttributes Dialects/ -gen-attrdef-doc)
