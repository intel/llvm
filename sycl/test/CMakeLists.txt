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

# TEST_INCLUDE_PATH is used for syntax-only verification of type information.
list(APPEND test_includes ${SYCL_INCLUDE})
list(APPEND test_includes ${SYCL_SOURCE_DIR}/source)
list(APPEND test_includes ${BOOST_UNORDERED_INCLUDE_DIRS})
if(SYCL_ENABLE_EXTENSION_JIT)
  list(APPEND test_includes ${LLVM_EXTERNAL_SYCL_JIT_SOURCE_DIR}/jit-compiler/include)
  list(APPEND test_includes ${LLVM_EXTERNAL_SYCL_JIT_SOURCE_DIR}/common/include)
endif(SYCL_ENABLE_EXTENSION_JIT)
list(JOIN test_includes ":" TEST_INCLUDE_PATH)

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

add_lit_testsuite(check-sycl-dumps "Running ABI dump tests only"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${RT_TEST_ARGS}
  PARAMS "SYCL_LIB_DUMPS_ONLY=True"
  DEPENDS FileCheck sycl-runtime-libraries llvm-readobj
  EXCLUDE_FROM_CHECK_ALL
  )

if(NOT WIN32 AND SYCL_ENABLE_EXTENSION_JIT)
  # lit-based testing of JIT passes isn't supported on Windows.
  add_dependencies(check-sycl check-sycl-jit)
endif()
