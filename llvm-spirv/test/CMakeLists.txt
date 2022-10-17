llvm_canonicalize_cmake_booleans(SPIRV_SKIP_CLANG_BUILD)
llvm_canonicalize_cmake_booleans(SPIRV_SKIP_DEBUG_INFO_TESTS)

# required by lit.site.cfg.py.in
get_target_property(LLVM_SPIRV_DIR llvm-spirv BINARY_DIR)
set(LLVM_SPIRV_TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if(SPIRV_TOOLS_FOUND)
  find_program(SPIRV_TOOLS_SPIRV_AS NAMES spirv-as PATHS ${SPIRV_TOOLS_PREFIX}/bin)
  if(SPIRV_TOOLS_SPIRV_AS)
    set(SPIRV_TOOLS_SPIRV_AS_FOUND True)
  endif()

  find_program(SPIRV_TOOLS_SPIRV_LINK NAMES spirv-link PATHS ${SPIRV_TOOLS_PREFIX}/bin)
  if(SPIRV_TOOLS_SPIRV_LINK)
    set(SPIRV_TOOLS_SPIRV_LINK_FOUND True)
  endif()

  find_program(SPIRV_TOOLS_SPIRV_VAL NAMES spirv-val PATHS ${SPIRV_TOOLS_PREFIX}/bin)
  if(SPIRV_TOOLS_SPIRV_VAL)
    set(SPIRV_TOOLS_SPIRV_VAL_FOUND True)
  endif()

  set(SPIRV_TOOLS_BINDIR "${SPIRV_TOOLS_PREFIX}/bin")
endif()

if(NOT SPIRV_TOOLS_SPIRV_AS)
  message(WARNING "spirv-as not found! SPIR-V assembly tests will not be run.")
  set(SPIRV_TOOLS_SPIRV_AS_FOUND False)
endif()

if(NOT SPIRV_TOOLS_SPIRV_LINK)
  message(WARNING "spirv-link not found! SPIR-V test involving the linker will not be run.")
  set(SPIRV_TOOLS_SPIRV_LINK_FOUND False)
endif()

if(NOT SPIRV_TOOLS_SPIRV_VAL)
  message(WARNING "spirv-val not found! SPIR-V generated for test suite will not be validated.")
  set(SPIRV_TOOLS_SPIRV_VAL_FOUND False)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

if(NOT LLVM_SPIRV_BUILD_EXTERNAL)
  set(LLVM_SPIRV_TEST_DEPS
    FileCheck
    count
    llvm-as
    llvm-config
    llvm-dis
    not
  )
  if(NOT SPIRV_SKIP_CLANG_BUILD)
    list(APPEND LLVM_SPIRV_TEST_DEPS
      clang
    )
  endif(NOT SPIRV_SKIP_CLANG_BUILD)
  if(NOT SPIRV_SKIP_DEBUG_INFO_TESTS)
    list(APPEND LLVM_SPIRV_TEST_DEPS
      llc
      llvm-dwarfdump
      llvm-objdump
      llvm-readelf
      llvm-readobj
    )
  endif(NOT SPIRV_SKIP_DEBUG_INFO_TESTS)
endif(NOT LLVM_SPIRV_BUILD_EXTERNAL)


add_lit_testsuite(check-llvm-spirv "Running the LLVM-SPIRV regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS
    --verbose
  DEPENDS
    ${LLVM_SPIRV_TEST_DEPS}
    llvm-spirv
)

# to enable a custom test target on cmake below 3.11
# starting with 3.11 "test" is only reserved if ENABLE_TESTING(ON)
if(LLVM_SPIRV_BUILD_EXTERNAL)
  cmake_policy(PUSH)
  if(POLICY CMP0037 AND ${CMAKE_VERSION} VERSION_LESS "3.11.0")
    cmake_policy(SET CMP0037 OLD)
  endif(POLICY CMP0037 AND ${CMAKE_VERSION} VERSION_LESS "3.11.0")
  add_custom_target(test
    DEPENDS
      check-llvm-spirv
  )
  cmake_policy(POP)
endif(LLVM_SPIRV_BUILD_EXTERNAL)
