set(LLVM_TOOLS_DIR "${LLVM_BINARY_DIR}/bin/")

get_target_property(SYCL_BINARY_DIR sycl-toolchain BINARY_DIR)

set(SYCL_INCLUDE "${dst_dir}")

set(RT_TEST_ARGS ${RT_TEST_ARGS} "-v")
set(DEPLOY_RT_TEST_ARGS ${DEPLOY_RT_TEST_ARGS} "-v -D SYCL_TOOLS_DIR=${CMAKE_INSTALL_PREFIX}/bin -D SYCL_LIBS_DIR=${CMAKE_INSTALL_PREFIX}/lib -D SYCL_INCLUDE=${dst_deploy_dir}")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
  )

list(APPEND SYCL_TEST_DEPS
  sycl-toolchain
  FileCheck
  not
  get_device_count_by_type
  llvm-config
  )

list(APPEND SYCL_DEPLOY_TEST_DEPS
  ${SYCL_TEST_DEPS}
  deploy-sycl-toolchain
  )

add_lit_testsuite(check-sycl "Running the SYCL regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${RT_TEST_ARGS}
  PARAMS "SYCL_BE=PI_OPENCL"
  DEPENDS ${SYCL_TEST_DEPS}
  )

add_lit_testsuite(check-sycl-deploy "Running the SYCL regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${DEPLOY_RT_TEST_ARGS}
  PARAMS "SYCL_BE=PI_OPENCL"
  DEPENDS ${SYCL_DEPLOY_TEST_DEPS}
  EXCLUDE_FROM_CHECK_ALL
  )
set_target_properties(check-sycl PROPERTIES FOLDER "SYCL tests")

add_lit_testsuites(SYCL ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${SYCL_TEST_DEPS}
  )

if(SYCL_BUILD_PI_CUDA)
  add_lit_testsuite(check-sycl-cuda "Running the SYCL regression tests for CUDA"
    ${CMAKE_CURRENT_BINARY_DIR}
    ARGS ${RT_TEST_ARGS}
    PARAMS "SYCL_BE=PI_CUDA"
    DEPENDS ${SYCL_TEST_DEPS}
  )

  set_target_properties(check-sycl-cuda PROPERTIES FOLDER "SYCL CUDA tests")

  add_lit_testsuites(SYCL-CUDA ${CMAKE_CURRENT_SOURCE_DIR}
    PARAMS "SYCL_BE=PI_CUDA"
    DEPENDS ${SYCL_TEST_DEPS}
  )
endif()
