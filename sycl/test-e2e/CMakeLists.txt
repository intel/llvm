if( NOT OpenCL_LIBRARY )
  find_package(OpenCL)
endif()

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

    if ("${TARGET_BE}" STREQUAL "")
      message(FATAL_ERROR
        "invalid empty target backend specification in CHECK_SYCL_ALL")
    elseif("${TARGET_DEVICES}" STREQUAL "")
      message(FATAL_ERROR
        "invalid empty target device specification in CHECK_SYCL_ALL")
    endif()
    message("Run on ${TARGET_DEVICES} for ${TARGET_BE}")

    string(REPLACE "," "_" TARGET check-sycl-${TARGET_BE}-${TARGET_DEVICES})

    if (WIN32)
      set(SYCL_PYTHON python.exe)
    else()
      set(SYCL_PYTHON python3)
    endif()

    add_custom_target(${TARGET}
      COMMAND ${SYCL_PYTHON} ${TEST_SUITE_LIT} ${TEST_SUITE_LIT_FLAGS} --param sycl_be=${TARGET_BE} --param target_devices=${TARGET_DEVICES} .
      COMMENT "Running the SYCL tests for ${TARGET} backend"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS ${TEST_SUITE_TARGETS}
      USES_TERMINAL
    )
    set_target_properties(${TARGET} PROPERTIES FOLDER "SYCL Level Zero tests")
    add_dependencies(check-sycl-all ${TARGET})

  endforeach()
endif(CHECK_SYCL_ALL)

add_subdirectory(External)
add_subdirectory(ExtraTests)
