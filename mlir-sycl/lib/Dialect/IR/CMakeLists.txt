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
    MLIRPolygeistDialect
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
    MLIRPolygeistDialect
    MLIRSCFDialect
    )
endif()

add_mlir_dialect_library(MLIRSYCLDialect
  SYCLTraits.cpp
  SYCLOps.cpp
  SYCLAttributes.cpp
  SYCLDialect.cpp
  SYCLTypes.cpp
  MethodUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_SYCL_SOURCE_DIR}/mlir/Dialect/SYCL

  DEPENDS
  MLIRSYCLOpsIncGen

  LINK_LIBS PUBLIC
  ${MLIR_LIBS_TO_LINK}
  )
