cmake_minimum_required(VERSION 3.20.0)

project(sycl-solution)
# Requirements
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(SYCL_ENABLE_WERROR "Treat all warnings as errors in SYCL project" OFF)
option(SYCL_DISABLE_STL_ASSERTIONS "Disable assertions in STL containers" ON)
option(SYCL_ADD_DEV_VERSION_POSTFIX "Adds -V postfix to version string" ON)
option(SYCL_ENABLE_COVERAGE "Enables code coverage for runtime and unit tests" OFF)
option(SYCL_ENABLE_STACK_PRINTING "Enables stack printing on crashes of SYCL applications" OFF)
option(SYCL_LIB_WITH_DEBUG_SYMBOLS "Builds SYCL runtime libraries with debug symbols" OFF)

if (NOT SYCL_COVERAGE_PATH)
  set(SYCL_COVERAGE_PATH "${CMAKE_CURRENT_BINARY_DIR}/profiles")
endif()

if (SYCL_ENABLE_PLUGINS)
  message(WARNING "SYCL_ENABLE_PLUGINS has been renamed, please use SYCL_ENABLE_BACKENDS instead")
  set(SYCL_ENABLE_BACKENDS "${SYCL_ENABLE_PLUGINS}" CACHE STRING "Backends enabled for SYCL" FORCE)
endif()

# If SYCL_ENABLE_BACKENDS is undefined, we default to enabling OpenCL and Level
# Zero backends.
if (NOT DEFINED SYCL_ENABLE_BACKENDS)
  set(SYCL_ENABLE_BACKENDS "opencl;level_zero;level_zero_v2")
endif()

# Option to enable JIT, this in turn makes kernel fusion and spec constant
# materialization possible.
option(SYCL_ENABLE_EXTENSION_JIT "Enable extension to JIT kernels" ON)

if (NOT XPTI_INCLUDES)
  set(XPTI_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../xpti/include)
endif()

if (NOT XPTI_PROXY_SRC)
  set(XPTI_PROXY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../xpti/src/xpti_proxy.cpp)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddSYCLExecutable)
include(AddSYCL)
include(SYCLUtils)

if(MSVC)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  # Skip asynchronous C++ exceptions catching and assume "extern C" functions
  # never throw C++ exceptions.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")

  # Add PDB debug information
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(CheckLinkerFlag)
  check_linker_flag(CXX "LINKER:/DEBUG" LINKER_SUPPORTS_DEBUG)
  if(LINKER_SUPPORTS_DEBUG)
    # sccache is not compatible with /Zi flag
    if (CMAKE_CXX_COMPILER_LAUNCHER STREQUAL "sccache")
      # CMake may put /Zi by default
      if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
        string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
      elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
      elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
        string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
      endif()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Z7")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")
    endif()
    add_link_options("LINKER:/DEBUG")

    # Enable unreferenced removal and ICF in Release mode.
    check_linker_flag(CXX "LINKER:/OPT:REF,/OPT:ICF" LINKER_SUPPORTS_OPTS)
    if (LINKER_SUPPORTS_OPTS AND uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
      add_link_options("LINKER:/OPT:REF" "LINKER:/OPT:ICF")
    endif()
  endif()
endif()

include(FetchEmhash)
include(FetchUnifiedRuntime)

# The change in SYCL_MAJOR_VERSION must be accompanied with the same update in
# llvm/clang/lib/Driver/CMakeLists.txt.
#
# See doc/developer/ABIPolicyGuide.md for the meaning when in the middle of
# development cycle.
set(SYCL_MAJOR_VERSION 8)
set(SYCL_MINOR_VERSION 0)
set(SYCL_PATCH_VERSION 0)

set(SYCL_DEV_ABI_VERSION 0)
if (SYCL_ADD_DEV_VERSION_POSTFIX)
  set(SYCL_VERSION_POSTFIX "-${SYCL_DEV_ABI_VERSION}")
endif()
set(SYCL_VERSION_STRING "${SYCL_MAJOR_VERSION}.${SYCL_MINOR_VERSION}.${SYCL_PATCH_VERSION}${SYCL_VERSION_POSTFIX}")

define_property(GLOBAL PROPERTY SYCL_TOOLCHAIN_INSTALL_COMPONENTS
  BRIEF_DOCS "List of components to deploy with SYCL toolchain"
  FULL_DOCS "List of components to deploy with SYCL toolchain"
)

# enable all warnings by default
if (MSVC)
  set(CMAKE_CXX_FLAGS "/W4 ${CMAKE_CXX_FLAGS}")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

if(SYCL_ENABLE_WERROR)
  if(MSVC)
    set(CMAKE_CXX_FLAGS "/WX ${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
endif()

# Create a soft option for enabling or disabling the instrumentation
# of the SYCL runtime and expect enabling
option(SYCL_ENABLE_XPTI_TRACING "Enable tracing of SYCL constructs" OFF)

# Get clang's version
include(VersionFromVCS)
set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

# If CLANG_VERSION_* is specified, use it, if not use LLVM_VERSION_*.
if(NOT DEFINED CLANG_VERSION_MAJOR)
  set(CLANG_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
endif()
if(NOT DEFINED CLANG_VERSION_MINOR)
  set(CLANG_VERSION_MINOR ${LLVM_VERSION_MINOR})
endif()
if(NOT DEFINED CLANG_VERSION_PATCHLEVEL)
  set(CLANG_VERSION_PATCHLEVEL ${LLVM_VERSION_PATCH})
endif()
# Unlike PACKAGE_VERSION, CLANG_VERSION does not include LLVM_VERSION_SUFFIX.
set(CLANG_VERSION "${CLANG_VERSION_MAJOR}.${CLANG_VERSION_MINOR}.${CLANG_VERSION_PATCHLEVEL}")

set(SYCL_INCLUDE_DIR "include")
set(SYCL_INCLUDE_BUILD_DIR ${LLVM_BINARY_DIR}/${SYCL_INCLUDE_DIR})

add_llvm_external_project(opencl)
list(FIND LLVM_ENABLE_PROJECTS opencl OPENCL_FOUND)
if(OPENCL_FOUND EQUAL -1)
  message(FATAL_ERROR "opencl external project required but not found.")
endif()

# Copy OpenCL Headers into sycl headers build directory
# Compiler does automatic lookup bin/../include based on clang binary location,
# e.g. when run LIT tests
file(COPY ${OpenCL_INCLUDE_DIR}/CL DESTINATION ${SYCL_INCLUDE_BUILD_DIR})

# Include OpenCL Headers into final bundle.
install(DIRECTORY ${OpenCL_INCLUDE_DIR}/CL
        DESTINATION ${SYCL_INCLUDE_DIR}
        COMPONENT OpenCL-Headers)

# Option for enabling building the SYCL major release preview library.
option(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB "Enable build of the SYCL major release preview library" ON)

# Needed for feature_test.hpp
if ("cuda" IN_LIST SYCL_ENABLE_BACKENDS)
  set(SYCL_BUILD_BACKEND_CUDA ON)
endif()
if ("hip" IN_LIST SYCL_ENABLE_BACKENDS)
  set(SYCL_BUILD_BACKEND_HIP ON)
endif()
if ("opencl" IN_LIST SYCL_ENABLE_BACKENDS)
  set(SYCL_BUILD_BACKEND_OPENCL ON)
endif()
if ("level_zero" IN_LIST SYCL_ENABLE_BACKENDS)
  set(SYCL_BUILD_BACKEND_LEVEL_ZERO ON)
endif()
if ("native_cpu" IN_LIST SYCL_ENABLE_BACKENDS)
  set(SYCL_BUILD_BACKEND_NATIVE_CPU ON)
endif()

# Set backend macros based on whether we can compile kernels for the target
# rather than if we're building the runtime adapter.
set(SYCL_EXT_ONEAPI_BACKEND_CUDA ${LLVM_HAS_NVPTX_TARGET})
set(SYCL_EXT_ONEAPI_BACKEND_HIP ${LLVM_HAS_AMDGPU_TARGET})

# Configure SYCL version macro
set(sycl_inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(sycl_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/source)
string(TIMESTAMP __SYCL_COMPILER_VERSION "%Y%m%d")
configure_file("source/version.hpp.in" "${SYCL_INCLUDE_BUILD_DIR}/sycl/version.hpp")
configure_file("source/feature_test.hpp.in" "${SYCL_INCLUDE_BUILD_DIR}/sycl/feature_test.hpp")

# Generate device_aspect_macros.hpp
find_package(Python3 REQUIRED)
add_custom_command(
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/source/device_aspect_macros_generator.py ${CMAKE_CURRENT_SOURCE_DIR}/include/sycl ${SYCL_INCLUDE_BUILD_DIR}/sycl
  OUTPUT  ${SYCL_INCLUDE_BUILD_DIR}/sycl/device_aspect_macros.hpp
  COMMENT "Generating device_aspect_macros.hpp"
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/sycl/info/aspects.def
          ${CMAKE_CURRENT_SOURCE_DIR}/include/sycl/info/aspects_deprecated.def
)

add_custom_target(sycl-device-aspect-macros-header
  DEPENDS "${SYCL_INCLUDE_BUILD_DIR}/sycl/device_aspect_macros.hpp"
)

# Install generated headers.
install(FILES
  "${SYCL_INCLUDE_BUILD_DIR}/sycl/feature_test.hpp"
  "${SYCL_INCLUDE_BUILD_DIR}/sycl/version.hpp"
  "${SYCL_INCLUDE_BUILD_DIR}/sycl/device_aspect_macros.hpp"
  DESTINATION "${SYCL_INCLUDE_DIR}/sycl"
  COMPONENT sycl-headers)

# This is workaround to detect changes (add or modify) in subtree which
# are not detected by copy_directory command.
# TODO: detect and process remove header/directory case
file(GLOB_RECURSE HEADERS_IN_SYCL_DIR CONFIGURE_DEPENDS "${sycl_inc_dir}/sycl/*")
file(GLOB_RECURSE HEADERS_IN_CL_DIR CONFIGURE_DEPENDS "${sycl_inc_dir}/CL/*")
file(GLOB_RECURSE HEADERS_IN_STD_DIR CONFIGURE_DEPENDS "${sycl_inc_dir}/std/*")
file(GLOB_RECURSE HEADERS_IN_SYCLCOMPAT_DIR CONFIGURE_DEPENDS "${sycl_inc_dir}/syclcompat/*" "${sycl_inc_dir}/syclcompat.hpp")

string(REPLACE "${sycl_inc_dir}" "${SYCL_INCLUDE_BUILD_DIR}"
  OUT_HEADERS_IN_SYCL_DIR "${HEADERS_IN_SYCL_DIR}")
string(REPLACE "${sycl_inc_dir}/CL" "${SYCL_INCLUDE_BUILD_DIR}/CL"
  OUT_HEADERS_IN_CL_DIR "${HEADERS_IN_CL_DIR}")
string(REPLACE "${sycl_inc_dir}" "${SYCL_INCLUDE_BUILD_DIR}"
  OUT_HEADERS_IN_STD_DIR "${HEADERS_IN_STD_DIR}")
string(REPLACE "${sycl_inc_dir}" "${SYCL_INCLUDE_BUILD_DIR}"
  OUT_HEADERS_IN_SYCLCOMPAT_DIR "${HEADERS_IN_SYCLCOMPAT_DIR}")

set(OUT_UR_HEADERS
  ${SYCL_INCLUDE_BUILD_DIR}/ur_api.h
  ${SYCL_INCLUDE_BUILD_DIR}/ur_api_funcs.def
  ${SYCL_INCLUDE_BUILD_DIR}/ur_print.hpp)
set(UR_HEADERS_TO_COPY
  ${UNIFIED_RUNTIME_INCLUDE_DIR}/ur_api.h
  ${UNIFIED_RUNTIME_INCLUDE_DIR}/ur_api_funcs.def
  ${UNIFIED_RUNTIME_INCLUDE_DIR}/ur_print.hpp)

# Copy SYCL headers from sources to build directory
add_custom_target(sycl-headers
  DEPENDS ${OUT_HEADERS_IN_SYCL_DIR}
          ${OUT_HEADERS_IN_CL_DIR}
          ${OUT_HEADERS_IN_STD_DIR}
          ${OUT_HEADERS_IN_SYCLCOMPAT_DIR}
          ${OUT_UR_HEADERS}
          sycl-device-aspect-macros-header
          )

add_custom_command(
  OUTPUT  ${OUT_HEADERS_IN_SYCL_DIR}
          ${OUT_HEADERS_IN_CL_DIR}
          ${OUT_HEADERS_IN_STD_DIR}
          ${OUT_HEADERS_IN_SYCLCOMPAT_DIR}
          ${OUT_UR_HEADERS}
  DEPENDS ${HEADERS_IN_SYCL_DIR}
          ${HEADERS_IN_CL_DIR}
          ${HEADERS_IN_STD_DIR}
          ${HEADERS_IN_SYCLCOMPAT_DIR}
          ${UR_HEADERS_TO_COPY}
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${sycl_inc_dir}/sycl ${SYCL_INCLUDE_BUILD_DIR}/sycl
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${sycl_inc_dir}/CL ${SYCL_INCLUDE_BUILD_DIR}/CL
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${sycl_inc_dir}/std ${SYCL_INCLUDE_BUILD_DIR}/std
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${sycl_inc_dir}/syclcompat ${SYCL_INCLUDE_BUILD_DIR}/syclcompat
  COMMAND ${CMAKE_COMMAND} -E copy ${sycl_inc_dir}/syclcompat.hpp ${SYCL_INCLUDE_BUILD_DIR}/syclcompat.hpp
  COMMAND ${CMAKE_COMMAND} -E copy ${UR_HEADERS_TO_COPY} ${SYCL_INCLUDE_BUILD_DIR}
  COMMENT "Copying SYCL headers ...")

# Copy SYCL headers from source to install directory
install(DIRECTORY "${sycl_inc_dir}/sycl" DESTINATION ${SYCL_INCLUDE_DIR} COMPONENT sycl-headers)
install(DIRECTORY "${sycl_inc_dir}/CL" DESTINATION ${SYCL_INCLUDE_DIR}/ COMPONENT sycl-headers)
install(DIRECTORY "${sycl_inc_dir}/std" DESTINATION ${SYCL_INCLUDE_DIR} COMPONENT sycl-headers)
install(DIRECTORY "${sycl_inc_dir}/syclcompat" DESTINATION ${SYCL_INCLUDE_DIR} COMPONENT sycl-headers)
install(FILES "${sycl_inc_dir}/syclcompat.hpp" DESTINATION ${SYCL_INCLUDE_DIR} COMPONENT sycl-headers)
install(FILES "${UNIFIED_RUNTIME_INCLUDE_DIR}/ur_api.h" DESTINATION ${SYCL_INCLUDE_DIR}
  COMPONENT sycl-headers)
install(FILES "${UNIFIED_RUNTIME_INCLUDE_DIR}/ur_api_funcs.def" DESTINATION ${SYCL_INCLUDE_DIR}
  COMPONENT sycl-headers)
install(FILES "${UNIFIED_RUNTIME_INCLUDE_DIR}/ur_print.hpp" DESTINATION ${SYCL_INCLUDE_DIR}
  COMPONENT sycl-headers)

if (WIN32)
  set(SYCL_RT_LIBS sycl${SYCL_MAJOR_VERSION})
  if(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB)
    list(APPEND SYCL_RT_LIBS sycl${SYCL_MAJOR_VERSION}-preview)
  endif()
  # Do we really support non-MSVC ABI on WIN?
  if (MSVC)
    list(APPEND SYCL_RT_LIBS sycl${SYCL_MAJOR_VERSION}d)
    if(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB)
      list(APPEND SYCL_RT_LIBS sycl${SYCL_MAJOR_VERSION}-previewd)
    endif()
  endif()
else()
  set(SYCL_RT_LIBS sycl)
  if(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB)
    list(APPEND SYCL_RT_LIBS sycl-preview)
  endif()
endif()

# This function allows building multiple libraries with the same options.
# Currently used by add_sycl_library and add_sycl_rt_library.
# Currently handles a gcc workaround
function( add_common_options LIB_NAME)
    # Workaround for bug in GCC version 5 and higher.
    # More information https://bugs.launchpad.net/ubuntu/+source/gcc-5/+bug/1568899
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
        CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
      target_link_libraries(${ARGV0} PRIVATE gcc_s gcc)
    endif()
endfunction(add_common_options)

if (LLVM_ENABLE_ASSERTIONS AND NOT SYCL_DISABLE_STL_ASSERTIONS AND NOT WIN32)
  if(LLVM_LIBCXX_USED)
    add_definitions(-D_LIBCPP_DEBUG=1)
    set(SYCL_CLANG_EXTRA_FLAGS "${SYCL_CLANG_EXTRA_FLAGS} -D_LIBCPP_DEBUG=1")
  else()
    add_definitions(-D_GLIBCXX_ASSERTIONS=1)
    set(SYCL_CLANG_EXTRA_FLAGS "${SYCL_CLANG_EXTRA_FLAGS} -D_GLIBCXX_ASSERTIONS=1")
  endif()
endif()

set(SYCL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# SYCL runtime library
add_subdirectory( source )

# Auxilliary extras for SYCL headers/library
if (NOT WIN32)
  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/gdb/libsycl.so-gdb.py"
    RENAME "libsycl.so.${SYCL_VERSION_STRING}-gdb.py"
    DESTINATION "lib${LLVM_LIBDIR_SUFFIX}/"
    COMPONENT sycl-headers-extras)
endif()

if(SYCL_ENABLE_XPTI_TRACING AND
    DEFINED LLVM_EXTERNAL_PROJECTS AND
    (NOT "xpti" IN_LIST LLVM_EXTERNAL_PROJECTS OR NOT "xptifw" IN_LIST LLVM_EXTERNAL_PROJECTS))
  message(FATAL_ERROR "SYCL_ENABLE_XPTI_TRACING=ON but XPTI is not going to be built")
endif()

if (SYCL_ENABLE_XPTI_TRACING)
  if (MSVC)
    set(XPTIFW_LIBS xpti xptid xptifw xptifwd)
  else()
    set(XPTIFW_LIBS xpti xptifw)
  endif()
endif()

# SYCL toolchain builds all components: compiler, libraries, headers, etc.
add_custom_target(sycl-compiler
  DEPENDS append-file
          clang
          clang-offload-wrapper
          clang-offload-bundler
          clang-offload-deps
          clang-offload-extract
          clang-offload-packager
          clang-linker-wrapper
          file-table-tform
          llc
          llvm-ar
          llvm-foreach
          llvm-spirv
          llvm-link
          llvm-objcopy
          spirv-to-ir-wrapper
          sycl-post-link
          opencl-aot
)

add_custom_target( sycl-runtime-libraries
  DEPENDS ${SYCL_RT_LIBS}
)

add_custom_target( sycl-toolchain ALL
  DEPENDS sycl-runtime-libraries
          sycl-compiler
          sycl-ls
          syclbin-dump
          ${XPTIFW_LIBS}
  COMMENT "Building SYCL compiler toolchain..."
)

if (WIN32)
  add_dependencies(sycl-toolchain ur_win_proxy_loader unified-runtimed-build)
endif()

if("cuda" IN_LIST SYCL_ENABLE_BACKENDS)
  add_dependencies(sycl-toolchain clang-nvlink-wrapper)
endif()

# Enable new IN_LIST operator.
cmake_policy(SET CMP0057 NEW)

if (libdevice IN_LIST LLVM_ENABLE_PROJECTS)
  add_dependencies(sycl-toolchain libsycldevice)
endif()

if (SYCL_ENABLE_STACK_PRINTING)
  add_dependencies(sycl-toolchain llvm-symbolizer)
endif()

option(SYCL_INCLUDE_TESTS
  "Generate build targets for the SYCL unit tests."
  ${LLVM_INCLUDE_TESTS})

# Ensure that HIP platform is uppercase, to match buildbot's output.
if(NOT "${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "")
  string(TOUPPER ${SYCL_BUILD_PI_HIP_PLATFORM} SYCL_BUILD_PI_HIP_PLATFORM)
endif()

add_subdirectory(tools)

if (WIN32)
  add_subdirectory(ur_win_proxy_loader)
endif()

if(SYCL_INCLUDE_TESTS)
  if(NOT LLVM_INCLUDE_TESTS)
      message(FATAL_ERROR
        "Can't build SYCL tests without LLVM_INCLUDE_TESTS enabled.")
  endif()
  if(EXISTS ${LLVM_THIRD_PARTY_DIR}/unittest/googletest/include/gtest/gtest.h)
    add_subdirectory(unittests)
    list(APPEND SYCL_TEST_DEPS SYCLUnitTests)
  endif()
  add_subdirectory(test)
endif()

get_property(SYCL_TOOLCHAIN_DEPS GLOBAL PROPERTY SYCL_TOOLCHAIN_INSTALL_COMPONENTS)
# Package deploy support
# Listed here are component names contributing the package
set( SYCL_TOOLCHAIN_DEPLOY_COMPONENTS
     append-file
     clang
     clang-offload-wrapper
     clang-offload-bundler
     clang-offload-deps
     clang-offload-extract
     clang-offload-packager
     clang-linker-wrapper
     file-table-tform
     llc
     llvm-ar
     llvm-foreach
     llvm-spirv
     llvm-link
     llvm-objcopy
     spirv-to-ir-wrapper
     sycl-post-link
     sycl-ls
     syclbin-dump
     clang-resource-headers
     OpenCL-Headers
     opencl-aot
     sycl-headers
     sycl-headers-extras
     sycl
     libsycldevice
     unified-memory-framework
     unified-runtime-loader
     ${XPTIFW_LIBS}
     ${SYCL_TOOLCHAIN_DEPS}
)

if("cuda" IN_LIST SYCL_ENABLE_BACKENDS)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS clang-nvlink-wrapper)
endif()

if (WIN32)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS ur_win_proxy_loader)
endif()

if (TARGET sycl-prof)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS sycl-prof)
endif()
if (TARGET sycl-sanitize)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS sycl-sanitize)
endif()
if (TARGET sycl-trace)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS sycl-trace)
endif()

if(OpenCL_INSTALL_KHRONOS_ICD_LOADER AND TARGET OpenCL-ICD)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS OpenCL-ICD)
endif()

# Build and install lld as part of the sycl-toolchain if available
if("lld" IN_LIST LLVM_ENABLE_PROJECTS)
  add_dependencies(sycl-toolchain lld)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS lld)
endif()

if("libclc" IN_LIST LLVM_ENABLE_PROJECTS)
  add_dependencies(sycl-toolchain libspirv-builtins)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS libspirv-builtins)
endif()

if("cuda" IN_LIST SYCL_ENABLE_BACKENDS)
  # Ensure that libclc is enabled.
  list(FIND LLVM_ENABLE_PROJECTS libclc LIBCLC_FOUND)
  if( LIBCLC_FOUND EQUAL -1 )
    message(FATAL_ERROR
      "CUDA support requires adding \"libclc\" to the CMake argument \"LLVM_ENABLE_PROJECTS\"")
  endif()

  add_dependencies(sycl-toolchain ur_adapter_cuda)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS ur_adapter_cuda)
endif()

if("hip" IN_LIST SYCL_ENABLE_BACKENDS)
  # Ensure that libclc is enabled.
  list(FIND LLVM_ENABLE_PROJECTS libclc LIBCLC_FOUND)
  if( LIBCLC_FOUND EQUAL -1 )
    message(FATAL_ERROR
      "HIP support requires adding \"libclc\" to the CMake argument \"LLVM_ENABLE_PROJECTS\"")
  endif()

  if(NOT TARGET lld AND "${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "AMD")
    message(FATAL_ERROR
      "HIP support requires adding \"lld\" to the CMake argument \"LLVM_ENABLE_PROJECTS\"")
  endif()

  add_dependencies(sycl-toolchain ur_adapter_hip)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS ur_adapter_hip)
endif()

if(SYCL_INSTALL_DEVICE_CONFIG_FILE)
  add_dependencies(sycl-toolchain DeviceConfigFile)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS DeviceConfigFile)
endif()

# Use it as fake dependency in order to force another command(s) to execute.
add_custom_command(OUTPUT __force_it
  COMMAND "${CMAKE_COMMAND}" -E echo
)
#Serialize installation to avoid missing components due to build race conditions
set(__chain_dep __force_it)

set(manifest_list)
foreach( comp ${SYCL_TOOLCHAIN_DEPLOY_COMPONENTS} )
  message( STATUS "Adding component ${comp} to deploy")

  set (manifest_file ${CMAKE_CURRENT_BINARY_DIR}/install_manifest_${comp}.txt)
  add_custom_command(OUTPUT ${manifest_file}
    COMMAND "${CMAKE_COMMAND}"
    "-DCMAKE_INSTALL_COMPONENT=${comp}"
    -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
    DEPENDS  ${__chain_dep}
    COMMENT "Deploying component ${comp}"
    USES_TERMINAL
  )
  list(APPEND manifest_list ${manifest_file})
  set(__chain_dep ${manifest_file})
endforeach( comp )

add_custom_target(deploy-sycl-toolchain
  DEPENDS sycl-toolchain ${manifest_list}
)

# SYCL Runtime documentation
add_subdirectory(doc)
# SYCL End-to-End tests
add_subdirectory(test-e2e)
