set(LIBS
  MLIRMlirOptMain
  MLIRSYCLDialect
  MLIRSYCLToGPU
  MLIRSYCLToLLVM
  MLIRSYCLTransforms
)

add_mlir_tool(sycl-mlir-opt
  sycl-mlir-opt.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(sycl-mlir-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(sycl-mlir-opt)
