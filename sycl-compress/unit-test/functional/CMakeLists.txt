# if (NOT EXISTS ${XPTI_DIR})
#   message (FATAL_ERROR "Undefined XPTI_DIR variable: Must be set for tests to work!")
# endif()
# include_directories(${XPTI_DIR}/include)

add_executable(SYCLCompressUnitTests EXCLUDE_FROM_ALL
  api-test.cpp
)

# Silence warnings from GTest
target_include_directories(SYCLCompressUnitTests SYSTEM PRIVATE
  ${gtest_SOURCE_DIR}/include
  ${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include
)

add_dependencies(SYCLCompressUnitTests sycl-compress)

target_link_libraries(SYCLCompressUnitTests PRIVATE
    llvm_gtest
    llvm_gtest_main
    sycl-compress
    LLVMSupport
)

add_test(NAME SYCLCompressTests COMMAND SYCLCompressUnitTests)

add_custom_target(check-sycl-compress-unittest COMMAND
  $<TARGET_PROPERTY:SYCLCompressUnitTests,RUNTIME_OUTPUT_DIRECTORY>/SYCLCompressUnitTests)
add_dependencies(check-sycl-compress-unittest SYCLCompressUnitTests)
add_dependencies(check-sycl-compress check-sycl-compress-unittest)
