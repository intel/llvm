if (NOT EXISTS ${XPTI_DIR})
  message (FATAL_ERROR "Undefined XPTI_DIR variable: Must be set for tests to work!")
endif()
include_directories(${XPTI_DIR}/include)

# Only download Google Test outside of LLVM tree.
if (NOT DEFINED LLVM_EXTERNAL_XPTIFW_SOURCE_DIR)
  message(STATUS "Building XPTI outside of LLVM Project...")
  # Download and unpack googletest at configure time
  configure_file(../CMakeLists.txt.in googletest-download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
  if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
  if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
  endif()

  # Prevent overriding the parent project's compiler/linker
  # settings on Windows
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

  # Add googletest directly to our build. This defines
  # the llvm_gtest and llvm_gtest_main targets.
  add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                  ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                  EXCLUDE_FROM_ALL)

  # The llvm_gtest/llvm_gtest_main targets carry header search
  # path dependencies automatically when using CMake 2.8.11 or
  # later. Otherwise we have to add them here ourselves.
  if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
  endif()
endif()

add_executable(XPTIFWUnitTests EXCLUDE_FROM_ALL
  xpti_api_tests.cpp
  xpti_correctness_tests.cpp
)

# Silence warnings from GTest
target_include_directories(XPTIFWUnitTests SYSTEM PRIVATE
  ${gtest_SOURCE_DIR}/include
  ${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include
)

if (TARGET LLVMSupport)
  set(SUPPORT_LIB LLVMSupport)
endif()

if (DEFINED LLVM_EXTERNAL_XPTIFW_SOURCE_DIR)
  # Use of SUPPORT_LIB: https://github.com/intel/llvm/pull/4652
  # LLVM's Google Test library requires LLVM Support. Typically it is
  # linked statically, but in case of shared libs configuration users
  # are required to manually specify link dependency. In other LLVM
  # sub-projects there's usually some sort of helper function (like
  # add_llvm_library), that automatically prepends LLVM prefix. But
  # XPTIFW has requirement to be buildable both in-tree and out-of-tree.
  # With the wrong library name (Support) if's body was skipped, but 
  #since the Support library was statically linked with gtest, it was
  # not required. Post-commit job with shared libs exposed this mistake.
  target_link_libraries(XPTIFWUnitTests PRIVATE
    llvm_gtest
    llvm_gtest_main
    ${xptifw_lib}
    ${SUPPORT_LIB}
  )
else()
  target_link_libraries(XPTIFWUnitTests PRIVATE
    gtest
    gtest_main
    ${xptifw_lib}
  )
endif()

add_test(NAME XPTIFWTests COMMAND XPTIFWUnitTests)

add_custom_target(check-xptifw-unittest COMMAND
  $<TARGET_PROPERTY:XPTIFWUnitTests,RUNTIME_OUTPUT_DIRECTORY>/XPTIFWUnitTests)
add_dependencies(check-xptifw-unittest XPTIFWUnitTests)
add_dependencies(check-xptifw check-xptifw-unittest)
