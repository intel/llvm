configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

add_lit_testsuite(check-sycl-opencl "Running the SYCL regression tests for OpenCL"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${RT_TEST_ARGS}
  PARAMS "SYCL_PLUGIN=opencl"
  DEPENDS ${SYCL_TEST_DEPS}
  EXCLUDE_FROM_CHECK_ALL
  )
set_target_properties(check-sycl-opencl PROPERTIES FOLDER "SYCL OpenCL device tests")

add_lit_testsuite(check-sycl-level-zero "Running the SYCL regression tests for Level Zero"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${RT_TEST_ARGS}
  PARAMS "SYCL_PLUGIN=level_zero"
  DEPENDS ${SYCL_TEST_DEPS}
  EXCLUDE_FROM_CHECK_ALL
  )
set_target_properties(check-sycl-level-zero PROPERTIES FOLDER "SYCL Level_Zero device tests")

if( TARGET check-sycl)
  add_dependencies(check-sycl check-sycl-opencl check-sycl-level-zero)
endif()

if(SYCL_BUILD_PI_CUDA)
  add_lit_testsuite(check-sycl-cuda-on-device "Running the SYCL regression tests for CUDA"
    ${CMAKE_CURRENT_BINARY_DIR}
    ARGS ${RT_TEST_ARGS}
    PARAMS "SYCL_PLUGIN=cuda"
    DEPENDS ${SYCL_TEST_DEPS}
    EXCLUDE_FROM_CHECK_ALL
  )
  set_target_properties(check-sycl-cuda-on-device PROPERTIES FOLDER "SYCL CUDA device tests")
  if(TARGET check-sycl-cuda)
    add_dependencies(check-sycl-cuda check-sycl-cuda-on-device)
  endif()
endif()

if(SYCL_BUILD_PI_ROCM)
  add_lit_testsuite(check-sycl-rocm-on-device "Running the SYCL regression tests for ROCm"
    ${CMAKE_CURRENT_BINARY_DIR}
    ARGS ${RT_TEST_ARGS}
    PARAMS "SYCL_PLUGIN=rocm"
    DEPENDS ${SYCL_TEST_DEPS}
    EXCLUDE_FROM_CHECK_ALL
  )
  set_target_properties(check-sycl-rocm-on-device PROPERTIES FOLDER "SYCL ROCm device tests")
  if(TARGET check-sycl-rocm)
    add_dependencies(check-sycl-rocm check-sycl-rocm-on-device)
  endif()
endif()
