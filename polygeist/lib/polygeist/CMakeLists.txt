add_mlir_dialect_library(MLIRPolygeist
Dialect.cpp
Ops.cpp

ADDITIONAL_HEADER_DIRS
${PROJECT_SOURCE_DIR}/include/polygeist

DEPENDS
MLIRPolygeistOpsIncGen

LINK_LIBS PUBLIC
MLIRIR
MLIRMemRefDialect
MLIRAffineDialect
MLIRSCFDialect
MLIRLLVMDialect
MLIROpenMPDialect
MLIRDataLayoutInterfaces
)
add_subdirectory(Passes)
