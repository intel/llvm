# Copyright (C) Codeplay Software Limited

#===--- CMakeLists.txt -----------------------------------------------------===#
#
# MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

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
    MLIRArithmeticDialect
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRControlFlowInterfaces
    MLIRIR
    MLIRSideEffectInterfaces
    MLIRVectorInterfaces
    MLIRMemRefDialect
    MLIRSCFDialect
    )
endif()

add_mlir_dialect_library(MLIRSYCLDialect
  SYCLOpsDialect.cpp
  SYCLOpsTypes.cpp
  SYCLOpsAlias.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/SYCL

  DEPENDS
  MLIRSYCLOpsIncGen

  LINK_LIBS PUBLIC
  ${MLIR_LIBS_TO_LINK}
  )
