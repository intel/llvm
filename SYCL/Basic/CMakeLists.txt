set(LLVM_TOOLS_DIR "${LLVM_BINARY_DIR}/bin/")

set(SYCL_INCLUDE "${SYCL_INCLUDE_BUILD_DIR}")
set(SYCL_TOOLS_SRC_DIR "${PROJECT_SOURCE_DIR}/tools/")
set(LLVM_BUILD_BINARY_DIRS "${LLVM_BINARY_DIR}/bin/")
set(LLVM_BUILD_LIBRARY_DIRS "${LLVM_BINARY_DIR}/lib/")

find_package(Threads REQUIRED)
set(SYCL_THREADS_LIB ${CMAKE_THREAD_LIBS_INIT})

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in" "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg")

if(CHECK_SYCL_ALL)
  message("Configure iterative execution on multiple backends")
  add_custom_target(check-sycl-all)
  foreach(BACKEND ${CHECK_SYCL_ALL})
    string(REPLACE ":" ";" BE_LIST ${BACKEND})
    list (GET BE_LIST 0 TARGET_BE)
    list (GET BE_LIST 1 TARGET_DEVICES)
    message("Run on ${TARGET_DEVICES} for ${TARGET_BE}")
    add_custom_target(check-sycl-${TARGET_BE}
      COMMAND ${TEST_SUITE_LIT} ${TEST_SUITE_LIT_FLAGS} --param sycl_be=${TARGET_BE} --param target_devices=${TARGET_DEVICES} .
      COMMENT "Running the SYCL tests for ${TARGET_BE} backend"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS ${TEST_SUITE_TARGETS}
      USES_TERMINAL
    )
    set_target_properties(check-sycl-${TARGET_BE} PROPERTIES FOLDER "SYCL Level Zero tests")
    add_dependencies(check-sycl-all check-sycl-${TARGET_BE})

  endforeach()
endif(CHECK_SYCL_ALL)
