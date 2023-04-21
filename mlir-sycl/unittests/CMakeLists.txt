set_target_properties(MLIRSYCLUnitTests PROPERTIES FOLDER "MLIR SYCL Tests")

function(add_mlir_unittest test_dirname)
  add_unittest(MLIRSYCLUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Analysis)
