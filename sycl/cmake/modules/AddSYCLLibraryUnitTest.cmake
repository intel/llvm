# add_sycl_library_unittest(test_suite_name sycl_extra_flags 
#                           file1.cpp file2.cpp ...)
#
# sycl_extra_flags: Clang extra compiler flags, e.g. 
#                   "-fsycl-unnamed-lambdas;-fsycl-device-code-split"
#
# Will compile the list of files together using clang.
# Produces a single binary using all the .cpp files
# named 'test_suite_name' at ${CMAKE_CURRENT_BINARY_DIR}.
macro(add_sycl_library_unittest test_suite_name)
  cmake_parse_arguments(ARG
    ""
    ""
    "SYCL_EXTRA_FLAGS;SOURCES"
    ${ARGN})

  set(CXX_COMPILER clang++)
  if(MSVC)
    set(CXX_COMPILER clang-cl.exe)
    set(LIB_POSTFIX ".lib")
  else()
    set(LIB_PREFIX "-l")
  endif()

  set(DEVICE_COMPILER_EXECUTABLE ${LLVM_RUNTIME_OUTPUT_INTDIR}/${CXX_COMPILER})
  set(_OUTPUT_BIN ${CMAKE_CURRENT_BINARY_DIR}/${test_suite_name}Tests)
  set(_TESTS_TARGET ${test_suite_name}Tests)
  set(_BIN_TARGET ${_TESTS_TARGET}_bin)
  set(_TARGET_DEPENDENCIES 
      "llvm_gtest_main;llvm_gtest;LLVMTestingSupport;LLVMSupport;LLVMDemangle")

  foreach(_lib ${_TARGET_DEPENDENCIES})
    list(APPEND _LIBRARIES $<TARGET_FILE:${_lib}>)
  endforeach()
  
  # Enable exception handling on Windows
  if(WIN32)
    list(APPEND _INTERNAL_EXTRA_FLAGS "/EHs")
    set(LIBRARIES_PREFIX ${CMAKE_BINARY_DIR}/lib/)
    list(APPEND _LIBRARIES $<TARGET_FILE:sycl>)
  endif()

  if(UNIX)
    foreach(_lib "pthread;dl;ncurses")
      list(APPEND _LIBRARIES "-l${_lib}">)
    endforeach()
  endif()

  if (SYCL_ENABLE_COVERAGE)
    list(APPEND _INTERNAL_EXTRA_FLAGS "-fprofile-instr-generate")
    list(APPEND _INTERNAL_EXTRA_FLAGS "-fcoverage-mapping")
  endif()

  get_target_property(GTEST_INCLUDES llvm_gtest INCLUDE_DIRECTORIES)
  foreach(_dir ${GTEST_INCLUDES})
    # Avoid -I when _dir contains an empty generator expression.
    list(APPEND INCLUDE_COMPILER_STRING "$<$<BOOL:${_dir}>:-I${_dir}>")
  endforeach()

  add_custom_target(${_BIN_TARGET}
    COMMAND ${DEVICE_COMPILER_EXECUTABLE} -fsycl ${ARG_SOURCES}
      -o ${_OUTPUT_BIN}
      ${ARG_SYCL_EXTRA_FLAGS}
      ${_INTERNAL_EXTRA_FLAGS}
      ${INCLUDE_COMPILER_STRING}
      ${_LIBRARIES}
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/${_TESTS_TARGET}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND_EXPAND_LISTS)

  add_dependencies(${_BIN_TARGET} sycl)
  foreach(_lib ${ARG_LIBRARIES})
      add_dependencies(${_BIN_TARGET} ${_TARGET_DEPENDENCIES})
  endforeach()

  add_dependencies(SYCLUnitTests ${_BIN_TARGET})

  add_executable(${_TESTS_TARGET} IMPORTED GLOBAL)
  set_target_properties(${_TESTS_TARGET} PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR})

  # Check target (Unix Only)
  add_custom_target(check-sycl-${test_suite_name}
    ${CMAKE_COMMAND} -E 
    env LD_LIBRARY_PATH="${CMAKE_BINARY_DIR}/lib"
    env SYCL_CONFIG_FILE_NAME=null.cfg
    env SYCL_DEVICELIB_NO_FALLBACK=1
    env SYCL_CACHE_DIR="${CMAKE_BINARY_DIR}/sycl_cache"
    ${CMAKE_CURRENT_BINARY_DIR}/${test_suite_name}
  )
  
  add_dependencies(check-sycl-${test_suite_name} ${_BIN_TARGET})
  add_dependencies(check-sycl-unittests-libs check-sycl-${test_suite_name})
  
endmacro()
