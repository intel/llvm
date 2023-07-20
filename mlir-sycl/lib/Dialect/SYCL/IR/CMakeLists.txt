if (TARGET MLIRArithmetic)
  set(MLIR_LIBS_TO_LINK
    MLIRArithmetic
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRControlFlowInterfaces
    MLIRGPUDialect
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
    MLIRGPUDialect
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
  MethodUtils.cpp
  SYCLAttributes.cpp
  SYCLDialect.cpp
  SYCLHostTraits.cpp
  SYCLMethodOpInterface.cpp
  SYCLOps.cpp
  SYCLTraits.cpp
  SYCLTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_SYCL_SOURCE_DIR}/include/mlir/Dialect/SYCL

  DEPENDS
  MLIRSYCLOpsIncGen

  LINK_LIBS PUBLIC
  ${MLIR_LIBS_TO_LINK}
  )
