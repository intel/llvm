set(LLVM_TOOLS_DIR "${LLVM_BINARY_DIR}/bin/")

get_target_property(SYCL_BINARY_DIR sycl-toolchain BINARY_DIR)

set(SYCL_INCLUDE "${SYCL_INCLUDE_BUILD_DIR}")
set(SYCL_TOOLS_SRC_DIR "${PROJECT_SOURCE_DIR}/tools/")
set(LLVM_BUILD_BINARY_DIRS "${LLVM_BINARY_DIR}/bin/")
set(LLVM_BUILD_LIBRARY_DIRS "${LLVM_BINARY_DIR}/lib/")

set(RT_TEST_ARGS ${RT_TEST_ARGS} "-v")
set(DEPLOY_RT_TEST_ARGS ${DEPLOY_RT_TEST_ARGS} "-v -D SYCL_TOOLS_DIR=${CMAKE_INSTALL_PREFIX}/bin -D SYCL_LIBS_DIR=${CMAKE_INSTALL_PREFIX}/lib${LLVM_LIBDIR_SUFFIX} -D SYCL_INCLUDE=${CMAKE_INSTALL_PREFIX}/${SYCL_INCLUDE_DIR}")

find_package(Threads REQUIRED)
set(SYCL_THREADS_LIB ${CMAKE_THREAD_LIBS_INIT})

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
  llvm-config
  llvm-cxxdump
  llvm-dis
  llvm-readobj
  )

list(APPEND SYCL_DEPLOY_TEST_DEPS
  ${SYCL_TEST_DEPS}
  deploy-sycl-toolchain
  )

set(SYCL_TEST_EXCLUDE "")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  # Debug compiler is very slow on sycl tests, tests may timeout sporadically
  set(SYCL_TEST_EXCLUDE EXCLUDE_FROM_CHECK_ALL)
  message(STATUS "Note: SYCL tests are excluded in check-all for Debug compiler")
endif()

add_lit_testsuite(check-sycl-deploy "Running the SYCL regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${DEPLOY_RT_TEST_ARGS}
  DEPENDS ${SYCL_DEPLOY_TEST_DEPS}
  EXCLUDE_FROM_CHECK_ALL
  )
set_target_properties(check-sycl-deploy PROPERTIES FOLDER "SYCL tests")

set(TRIPLES "spir64-unknown-unknown")
if (SYCL_BUILD_BACKEND_CUDA OR (SYCL_BUILD_BACKEND_HIP AND "${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "NVIDIA"))
  set(TRIPLES "${TRIPLES},nvptx64-nvidia-cuda")
endif()
if ((SYCL_BUILD_BACKEND_HIP AND "${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "AMD"))
  set(TRIPLES "${TRIPLES},amdgcn-amd-amdhsa")
endif()

add_lit_testsuite(check-sycl-combined-triples "Running device-agnostic SYCL regression tests for all available triples"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${RT_TEST_ARGS}
  PARAMS "SYCL_TRIPLE=${TRIPLES}"
  DEPENDS ${SYCL_TEST_DEPS}
  ${SYCL_TEST_EXCLUDE}
  )

add_custom_target(check-sycl)
add_dependencies(check-sycl check-sycl-combined-triples)
set_target_properties(check-sycl PROPERTIES FOLDER "SYCL tests")

add_lit_testsuite(check-sycl-spirv "Running device-agnostic SYCL regression tests for SPIR-V"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${RT_TEST_ARGS}
  PARAMS "SYCL_TRIPLE=spir64-unknown-unknown"
  DEPENDS ${SYCL_TEST_DEPS}
  ${SYCL_TEST_EXCLUDE}
  EXCLUDE_FROM_CHECK_ALL
  )

add_lit_testsuite(check-sycl-dumps "Running ABI dump tests only"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${RT_TEST_ARGS}
  PARAMS "SYCL_LIB_DUMPS_ONLY=True"
  DEPENDS sycl-runtime-libraries llvm-readobj
  EXCLUDE_FROM_CHECK_ALL
  )

if(SYCL_BUILD_BACKEND_CUDA)
  add_lit_testsuite(check-sycl-ptx "Running device-agnostic SYCL regression tests for NVidia PTX"
    ${CMAKE_CURRENT_BINARY_DIR}
    ARGS ${RT_TEST_ARGS}
    PARAMS "SYCL_TRIPLE=nvptx64-nvidia-cuda"
    DEPENDS ${SYCL_TEST_DEPS}
    ${SYCL_TEST_EXCLUDE}
    EXCLUDE_FROM_CHECK_ALL
  )

  add_custom_target(check-sycl-cuda)
  add_dependencies(check-sycl-cuda check-sycl-ptx)
endif()

if(SYCL_BUILD_BACKEND_HIP)
  add_custom_target(check-sycl-hip)
  if("${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "NVIDIA")
    add_lit_testsuite(check-sycl-hip-ptx "Running device-agnostic SYCL regression tests for HIP NVidia PTX"
      ${CMAKE_CURRENT_BINARY_DIR}
      ARGS ${RT_TEST_ARGS}
      PARAMS "SYCL_TRIPLE=nvptx64-nvidia-cuda"
      DEPENDS ${SYCL_TEST_DEPS}
      ${SYCL_TEST_EXCLUDE}
      EXCLUDE_FROM_CHECK_ALL
    )

    add_dependencies(check-sycl-hip check-sycl-hip-ptx)
  elseif("${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "AMD")
    add_lit_testsuite(check-sycl-hip-gcn "Running device-agnostic SYCL regression tests for HIP AMDGCN"
      ${CMAKE_CURRENT_BINARY_DIR}
      ARGS ${RT_TEST_ARGS}
      PARAMS "SYCL_TRIPLE=amdgcn-amd-amdhsa"
      DEPENDS ${SYCL_TEST_DEPS}
      ${SYCL_TEST_EXCLUDE}
      EXCLUDE_FROM_CHECK_ALL
    )

    add_dependencies(check-sycl-hip check-sycl-hip-gcn)
  else()
    message(FATAL_ERROR "SYCL_BUILD_PI_HIP_PLATFORM must be set to either 'AMD' or 'NVIDIA' (set to: '${SYCL_BUILD_PI_HIP_PLATFORM}')")
  endif()
endif()

if(SYCL_ENABLE_KERNEL_FUSION)
  add_dependencies(check-sycl check-sycl-fusion)
endif(SYCL_ENABLE_KERNEL_FUSION)
