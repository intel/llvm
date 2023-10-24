cmake_minimum_required(VERSION 3.13.4)

if(NOT DEFINED BASE_LLVM_VERSION)
  set (BASE_LLVM_VERSION 18.0.0)
endif(NOT DEFINED BASE_LLVM_VERSION)
set(LLVM_SPIRV_VERSION ${BASE_LLVM_VERSION}.0)

include(FetchContent)
include(FindPkgConfig)

option(LLVM_SPIRV_INCLUDE_TESTS
  "Generate build targets for the llvm-spirv lit tests."
  ${LLVM_INCLUDE_TESTS})

if (NOT DEFINED LLVM_SPIRV_BUILD_EXTERNAL)
  # check if we build inside llvm or not
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LLVM_SPIRV_BUILD_EXTERNAL YES)
  endif(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
endif (NOT DEFINED LLVM_SPIRV_BUILD_EXTERNAL)

if(LLVM_SPIRV_BUILD_EXTERNAL)
  # Make sure llvm-spirv gets built when building outside the llvm tree.
  set(LLVM_BUILD_TOOLS ON)
endif(LLVM_SPIRV_BUILD_EXTERNAL)

# Download spirv.hpp from the official SPIRV-Headers repository.
# One can skip this step by manually setting
# LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR path.
if(NOT DEFINED LLVM_TOOL_SPIRV_HEADERS_BUILD AND
   NOT DEFINED LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR)
  set(LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/SPIRV-Headers")
  message(STATUS "SPIR-V Headers location is not specified. Will try to download
          spirv.hpp from https://github.com/KhronosGroup/SPIRV-Headers into
          ${LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR}")
  file(READ spirv-headers-tag.conf SPIRV_HEADERS_TAG)
  # Strip the potential trailing newline from tag
  string(STRIP "${SPIRV_HEADERS_TAG}" SPIRV_HEADERS_TAG)
  FetchContent_Declare(spirv-headers
    GIT_REPOSITORY    https://github.com/KhronosGroup/SPIRV-Headers.git
    GIT_TAG           ${SPIRV_HEADERS_TAG}
    SOURCE_DIR ${LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR}
  )
  FetchContent_MakeAvailable(spirv-headers)
else()
  if(NOT DEFINED LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR)
    # This means LLVM_TOOL_SPIRV_HEADERS_BUILD is defined, therefore
    # SPIRV-Headers exist as a subproject.
    set(LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR
      "${CMAKE_SOURCE_DIR}/projects/SPIRV-Headers")
    if(NOT EXISTS ${LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR})
      message(FATAL_ERROR "No location specified for SPIRV-Headers.
              Try setting the LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR
              path or put the project into the llvm/projects folder
              under the name 'SPIRV-Headers'")
    endif()
  endif()
  message(STATUS "Using SPIR-V Headers from
          ${LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR}")
endif()

if(LLVM_SPIRV_BUILD_EXTERNAL)
  project(LLVM_SPIRV
    VERSION
      ${LLVM_SPIRV_VERSION}
    LANGUAGES
      CXX
      C
  )

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  if(LLVM_SPIRV_INCLUDE_TESTS)
    set(LLVM_TEST_COMPONENTS
      llvm-as
      llvm-dis
    )
  endif(LLVM_SPIRV_INCLUDE_TESTS)

  find_package(LLVM ${BASE_LLVM_VERSION} REQUIRED
    COMPONENTS
      Analysis
      BitReader
      BitWriter
      CodeGen
      Core
      Passes
      Support
      TargetParser
      TransformUtils
      ${LLVM_TEST_COMPONENTS}
  )
  set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${LLVM_CMAKE_DIR}
  )
  include(AddLLVM)
  include(HandleLLVMOptions)

  message(STATUS "Found LLVM: ${LLVM_VERSION}")

  option(CCACHE_ALLOWED "allow use of ccache" TRUE)
  find_program(CCACHE_EXE_FOUND ccache)
  if(CCACHE_EXE_FOUND AND CCACHE_ALLOWED)
    message(STATUS "Found ccache: ${CCACHE_EXE_FOUND}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
  endif()
endif()

set(LLVM_SPIRV_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)

# first try locating SPIRV-Tools via pkgconfig (the old way)
pkg_search_module(SPIRV_TOOLS SPIRV-Tools)
if (NOT SPIRV_TOOLS_FOUND)
  # then try locating SPIRV-Tools via cmake (the new way)
  find_package(SPIRV-Tools)
  find_package(SPIRV-Tools-tools)
  if (SPIRV-Tools_FOUND AND SPIRV-Tools-tools_FOUND)
    set(SPIRV_TOOLS_FOUND TRUE)
    # check for the existance of library targets in the found packages
    if(TARGET SPIRV-Tools-shared)
      # use the shared libary target if present
      set(SPIRV-Tools-library SPIRV-Tools-shared)
    elseif(TARGET SPIRV-Tools-static)
      # otherwise fallback to the static library target
      set(SPIRV-Tools-library SPIRV-Tools-static)
    else()
      message(FATAL_ERROR "Found SPIRV-Tools package but neither "
              "SPIRV-Tools-shared or SPIRV-Tools-static targets exist.")
    endif()
    set(SPIRV_TOOLS_LDFLAGS ${SPIRV-Tools-library})
    get_target_property(SPIRV_TOOLS_INCLUDE_DIRS ${SPIRV-Tools-library} INTERFACE_INCLUDE_DIRECTORIES)
  endif()
endif()

if (NOT SPIRV_TOOLS_FOUND)
  message(STATUS "SPIRV-Tools not found; project will be built without "
          "--spirv-tools-dis support.")
endif()


add_subdirectory(lib/SPIRV)
add_subdirectory(tools/llvm-spirv)
if(LLVM_SPIRV_INCLUDE_TESTS)
  add_subdirectory(test)
endif(LLVM_SPIRV_INCLUDE_TESTS)

install(
  FILES
    ${LLVM_SPIRV_INCLUDE_DIRS}/LLVMSPIRVLib.h
    ${LLVM_SPIRV_INCLUDE_DIRS}/LLVMSPIRVOpts.h
    ${LLVM_SPIRV_INCLUDE_DIRS}/LLVMSPIRVExtensions.inc
  DESTINATION
    ${CMAKE_INSTALL_PREFIX}/include/LLVMSPIRVLib
)

configure_file(LLVMSPIRVLib.pc.in ${CMAKE_BINARY_DIR}/LLVMSPIRVLib.pc @ONLY)
install(
  FILES
    ${CMAKE_BINARY_DIR}/LLVMSPIRVLib.pc
  DESTINATION
    ${CMAKE_INSTALL_PREFIX}/lib${LLVM_LIBDIR_SUFFIX}/pkgconfig
)
