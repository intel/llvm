add_mlir_dialect(SYCLOps sycl)
add_mlir_doc(SYCLOps SYCLOps Dialects/ -gen-op-doc -gen-op-interface-doc)

set(LLVM_TARGET_DEFINITIONS SYCLOpInterfaces.td)
mlir_tablegen(SYCLOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(SYCLOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(SYCLOpInterfacesIncGen)
add_dependencies(mlir-generic-headers SYCLOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS SYCLOpAttributes.td)
mlir_tablegen(SYCLOpAttributes.h.inc -gen-enum-decls)
mlir_tablegen(SYCLOpAttributes.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRSYCLOpAttributeIncGen)
add_dependencies(mlir-headers MLIRSYCLOpAttributeIncGen)
