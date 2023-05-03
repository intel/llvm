set_target_properties(MLIRPolygeistUnitTests PROPERTIES FOLDER "MLIR Polygeist Tests")

function(add_mlir_unittest test_dirname)
  add_unittest(MLIRPolygeistUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Analysis)
