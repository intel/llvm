set(LIBS
  MLIRMlirOptMain
  MLIRSYCLDialect
  MLIRSYCLToGPU
  MLIRSYCLToLLVM
  MLIRSYCLToMath
  MLIRSYCLToSPIRV
  MLIRSYCLTransforms
)

if(MLIR_INCLUDE_TESTS)
  set(TEST_LIBS
    MLIRSYCLTestAnalysis
  )
endif()

add_mlir_tool(sycl-mlir-opt
  sycl-mlir-opt.cpp

  DEPENDS
  ${LIBS}
  ${TEST_LIBS}
  )
target_link_libraries(sycl-mlir-opt PRIVATE ${LIBS} ${TEST_LIBS})
mlir_check_all_link_libraries(sycl-mlir-opt)
