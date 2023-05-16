if(DEFINED SYCL_EXTRA_TESTS_SRC)
  message("-- Run extra tests from ${SYCL_EXTRA_TESTS_SRC}")
  add_custom_target( ExtraTests ALL
    DEPENDS   ${SYCL_EXTRA_TESTS_SRC}
    COMMAND   ${CMAKE_COMMAND} -E remove_directory
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests
    COMMAND   ${CMAKE_COMMAND} -E copy_directory
                 ${SYCL_EXTRA_TESTS_SRC}
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests
    COMMENT   "Deploying extra SYCL tests..."
  )
  set_property(GLOBAL APPEND PROPERTY TEST_SUITE_TARGETS ExtraTests)
endif()
