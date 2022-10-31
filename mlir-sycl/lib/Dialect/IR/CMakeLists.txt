if (TARGET MLIRArithmetic)
  set(MLIR_LIBS_TO_LINK
    MLIRArithmetic
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRControlFlowInterfaces
    MLIRIR
    MLIRSideEffectInterfaces
    MLIRVectorInterfaces
    MLIRStandard
    MLIRMemRef
    MLIRSCF
    )
else()
  set(MLIR_LIBS_TO_LINK
    MLIRArithDialect
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRControlFlowInterfaces
    MLIRIR
    MLIRSideEffectInterfaces
    MLIRVectorInterfaces
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    )
endif()

add_mlir_dialect_library(MLIRSYCLDialect
  SYCLTraits.cpp
  SYCLOps.cpp
  SYCLOpsDialect.cpp
  SYCLOpsTypes.cpp
  SYCLOpsAlias.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_SYCL_SOURCE_DIR}/mlir/Dialect/SYCL

  DEPENDS
  MLIRSYCLOpsIncGen

  LINK_LIBS PUBLIC
  ${MLIR_LIBS_TO_LINK}
  )
