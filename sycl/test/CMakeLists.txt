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
  get_device_count_by_type
  llvm-config
  llvm-cxxdump
  llvm-readobj
  )

list(APPEND SYCL_DEPLOY_TEST_DEPS
  ${SYCL_TEST_DEPS}
  deploy-sycl-toolchain
  )

add_lit_testsuite(check-sycl-deploy "Running the SYCL regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${DEPLOY_RT_TEST_ARGS}
  PARAMS "SYCL_PLUGIN=opencl"
  DEPENDS ${SYCL_DEPLOY_TEST_DEPS}
  EXCLUDE_FROM_CHECK_ALL
  )
set_target_properties(check-sycl-deploy PROPERTIES FOLDER "SYCL tests")

add_lit_testsuite(check-sycl-spirv "Running device-agnostic SYCL regression tests for SPIR-V"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${RT_TEST_ARGS}
  PARAMS "SYCL_TRIPLE=spir64-unknown-unknown-sycldevice"
  DEPENDS ${SYCL_TEST_DEPS}
  EXCLUDE_FROM_CHECK_ALL
  )

add_custom_target(check-sycl)
add_dependencies(check-sycl check-sycl-spirv)
set_target_properties(check-sycl PROPERTIES FOLDER "SYCL tests")

if(SYCL_BUILD_PI_CUDA)
  add_lit_testsuite(check-sycl-ptx "Running device-agnostic SYCL regression tests for NVidia PTX"
    ${CMAKE_CURRENT_BINARY_DIR}
    ARGS ${RT_TEST_ARGS}
    PARAMS "SYCL_TRIPLE=nvptx64-nvidia-cuda-sycldevice;SYCL_PLUGIN=cuda"
    DEPENDS ${SYCL_TEST_DEPS}
    EXCLUDE_FROM_CHECK_ALL
  )

  add_custom_target(check-sycl-cuda)
  add_dependencies(check-sycl-cuda check-sycl-ptx)
  add_dependencies(check-sycl check-sycl-cuda)
endif()

if(SYCL_BUILD_PI_ROCM)
  add_custom_target(check-sycl-rocm)
  if("${SYCL_BUILD_PI_ROCM_PLATFORM}" STREQUAL "NVIDIA")
    add_lit_testsuite(check-sycl-rocm-ptx "Running device-agnostic SYCL regression tests for ROCm NVidia PTX"
      ${CMAKE_CURRENT_BINARY_DIR}
      ARGS ${RT_TEST_ARGS}
      PARAMS "SYCL_TRIPLE=nvptx64-nvidia-cuda-sycldevice;SYCL_PLUGIN=rocm"
      DEPENDS ${SYCL_TEST_DEPS}
      EXCLUDE_FROM_CHECK_ALL
    )

    add_dependencies(check-sycl-rocm check-sycl-rocm-ptx)
  elseif("${SYCL_BUILD_PI_ROCM_PLATFORM}" STREQUAL "AMD")
    add_lit_testsuite(check-sycl-rocm-gcn "Running device-agnostic SYCL regression tests for ROCm AMDGCN"
      ${CMAKE_CURRENT_BINARY_DIR}
      ARGS ${RT_TEST_ARGS}
      PARAMS "SYCL_TRIPLE=amdgcn-amd-amdhsa-sycldevice;SYCL_PLUGIN=rocm"
      DEPENDS ${SYCL_TEST_DEPS}
      EXCLUDE_FROM_CHECK_ALL
    )

    add_dependencies(check-sycl-rocm check-sycl-rocm-gcn)
  else()
    message(FATAL_ERROR "SYCL_BUILD_PI_ROCM_PLATFORM must be set to either 'AMD' or 'NVIDIA' (set to: '${SYCL_BUILD_PI_ROCM_PLATFORM}')")
  endif()

  add_dependencies(check-sycl check-sycl-rocm)
endif()
add_subdirectory(on-device)
