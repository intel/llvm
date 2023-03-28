cmake_minimum_required(VERSION 3.15)

message("Configuring SYCL End-to-End Tests")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(sycl-e2e-test-suite CXX)
  set(SYCL_TEST_E2E_STANDALONE TRUE)
endif()

if(SYCL_TEST_E2E_STANDALONE)
  if( NOT OpenCL_LIBRARY )
    find_package(OpenCL)
  endif()
endif() # Standalone.

if(SYCL_TEST_E2E_STANDALONE)
  set(SYCL_CXX_COMPILER ${CMAKE_CXX_COMPILER})
else()
  set(SYCL_CXX_COMPILER "${LLVM_BINARY_DIR}/bin/clang++")
endif() # Standalone.

find_package(Threads REQUIRED)
set(SYCL_THREADS_LIB ${CMAKE_THREAD_LIBS_INIT})

if(NOT LLVM_LIT)
  find_program(LLVM_LIT
               NAMES llvm-lit lit.py lit
               PATHS "${LLVM_MAIN_SRC_DIR}/utils/lit"
               DOC "Path to lit.py")
endif()

set(SYCL_E2E_TESTS_LIT_FLAGS "-sv" CACHE STRING "Flags used when running lit")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
               "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg")

if(SYCL_TEST_E2E_TARGETS)
  message("Configure iterative execution on multiple backends")
  add_custom_target(check-sycl-e2e)
  foreach(TARGET_STR ${SYCL_TEST_E2E_TARGETS})
    string(REPLACE ":" ";" TARGET_LIST ${TARGET_STR})
    list (GET TARGET_LIST 0 TARGET_BE)
    list (GET TARGET_LIST 1 TARGET_DEVICES)

    if ("${TARGET_BE}" STREQUAL "")
      message(FATAL_ERROR
        "invalid empty target backend specification in SYCL_TEST_E2E_TARGETS")
    elseif("${TARGET_DEVICES}" STREQUAL "")
      message(FATAL_ERROR
        "invalid empty target device specification in SYCL_TEST_E2E_TARGETS")
    endif()
    message("Run on ${TARGET_DEVICES} for ${TARGET_BE}")

    string(REPLACE "," "_" TARGET check-sycl-e2e-${TARGET_BE}-${TARGET_DEVICES})

    file(STRINGS "xfail_tests.txt" XFAIL_TESTS_LIST)
    string(REPLACE ";" "\;" XFAIL_TESTS "${XFAIL_TESTS_LIST}")
    add_custom_target(${TARGET}
      COMMAND ${Python3_EXECUTABLE} ${LLVM_LIT} ${SYCL_E2E_TESTS_LIT_FLAGS} --xfail '${XFAIL_TESTS}' --param sycl_be=${TARGET_BE} --param target_devices=${TARGET_DEVICES} .
      COMMENT "Running the SYCL tests for ${TARGET} backend"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      USES_TERMINAL
    )
    set_target_properties(${TARGET} PROPERTIES FOLDER "SYCL Level Zero tests")
    add_dependencies(check-sycl-e2e ${TARGET})

  endforeach()
endif(SYCL_TEST_E2E_TARGETS)

add_subdirectory(External)
add_subdirectory(ExtraTests)
