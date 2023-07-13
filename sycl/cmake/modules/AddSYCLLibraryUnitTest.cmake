# add_sycl_library_unittest(test_suite_name sycl_extra_flags 
#                           file1.cpp file2.cpp ...)
#
# sycl_extra_flags: Clang extra compiler flags, e.g. 
#                   "-fsycl-unnamed-lambdas;-fsycl-device-code-split"
#
# Will compile the list of files together using clang.
# Produces a single binary using all the .cpp files
# named 'test_suite_name' at ${CMAKE_CURRENT_BINARY_DIR}.
macro(add_sycl_library_unittest test_suite_name sycl_extra_flags)

  add_custom_target(check-sycl-${test_suite_name}
    ${CMAKE_COMMAND} -E env
    LD_LIBRARY_PATH="${CMAKE_BINARY_DIR}/lib"
    LLVM_PROFILE_FILE="${SYCL_COVERAGE_PATH}/${test_dirname}.profraw"
    env SYCL_CONFIG_FILE_NAME=null.cfg
    env SYCL_DEVICELIB_NO_FALLBACK=1
    env SYCL_CACHE_DIR="${CMAKE_BINARY_DIR}/sycl_cache"
    ${CMAKE_CURRENT_BINARY_DIR}/${test_suite_name}
  )

  add_dependencies(check-sycl-unittests-libs check-sycl-${test_suite_name})

  if (SYCL_ENABLE_COVERAGE)
    list(APPEND sycl_extra_flags "-fprofile-instr-generate")
    list(APPEND sycl_extra_flags "-fcoverage-mapping")
  endif()
  
  get_target_property(GTEST_INCLUDES llvm_gtest INCLUDE_DIRECTORIES)
  foreach(_dir ${GTEST_INCLUDES})
    # Avoid -I when _dir contains an empty generator expression.
    list(APPEND INCLUDE_COMPILER_STRING "$<$<BOOL:${_dir}>:-I${_dir}>")
  endforeach()

  add_sycl_executable(
    # 'Tests' suffix needed by lit framework to detect the tests
    ${test_suite_name}Tests
    OPTIONS
      ${sycl_extra_flags}
      ${INCLUDE_COMPILER_STRING}
    LIBRARIES
      llvm_gtest_main
      llvm_gtest
      LLVMSupport
      LLVMDemangle
      pthread
      dl
      ncurses
    SOURCES
      ${ARGN}
    DEPENDANTS
      check-sycl-${test_suite_name}
      SYCLUnitTests)

endmacro()

