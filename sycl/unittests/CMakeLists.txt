add_custom_target(PiUnitTests)
set_target_properties(PiUnitTests PROPERTIES FOLDER "Tests")

function(add_llvm_unittest test_dirname)
  add_unittest(PiUnitTests ${test_dirname} ${ARGN})
endfunction()

function(add_llvm_unittest_with_input_files test_dirname)
  add_unittest_with_input_files(PiUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(pi)