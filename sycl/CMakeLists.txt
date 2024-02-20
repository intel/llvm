cmake_minimum_required(VERSION 3.14)

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

# If SYCL_ENABLE_PLUGINS is undefined, we default to enabling OpenCL and Level
# Zero plugins.
if (NOT DEFINED SYCL_ENABLE_PLUGINS)
  set(SYCL_ENABLE_PLUGINS "opencl;level_zero")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddSYCLExecutable)
include(AddSYCL)
include(SYCLUtils)

# The change in SYCL_MAJOR_VERSION must be accompanied with the same update in
# llvm/clang/lib/Driver/CMakeLists.txt.
#
# See doc/developer/ABIPolicyGuide.md for the meaning when in the middle of
# development cycle.
set(SYCL_MAJOR_VERSION 7)
set(SYCL_MINOR_VERSION 2)
set(SYCL_PATCH_VERSION 0)

set(SYCL_DEV_ABI_VERSION 8)
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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-deprecated-declarations")
endif()

if(SYCL_ENABLE_WERROR)
  if(MSVC)
    set(CMAKE_CXX_FLAGS "/WX ${CMAKE_CXX_FLAGS}")
    add_definitions(
      -wd4996 # Suppress 'function': was declared deprecated'
    )
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
endif()

# Create a soft option for enabling or disabling the instrumentation
# of the SYCL runtime and expect enabling
option(SYCL_ENABLE_XPTI_TRACING "Enable tracing of SYCL constructs" OFF)

if(MSVC)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  # Skip asynchronous C++ exceptions catching and assume "extern C" functions
  # never throw C++ exceptions.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")

  # Add PDB debug information
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(LLVMCheckLinkerFlag)
  llvm_check_linker_flag(CXX "/DEBUG" LINKER_SUPPORTS_DEBUG)
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
    add_link_options("/DEBUG")

    # Enable unreferenced removal and ICF in Release mode.
    llvm_check_linker_flag(CXX "/OPT:REF /OPT:ICF" LINKER_SUPPORTS_OPTS)
    if (LINKER_SUPPORTS_OPTS AND uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
      add_link_options("/OPT:REF" "/OPT:ICF")
    endif()
  endif()
endif()

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
file(COPY ${OpenCL_INCLUDE_DIR}/CL
     DESTINATION ${SYCL_INCLUDE_BUILD_DIR}/sycl)

# Include OpenCL Headers into final bundle.
install(DIRECTORY ${OpenCL_INCLUDE_DIR}/CL
        DESTINATION ${SYCL_INCLUDE_DIR}/sycl
        COMPONENT OpenCL-Headers)

# Option to enable online kernel fusion via a JIT compiler
option(SYCL_ENABLE_KERNEL_FUSION "Enable kernel fusion via JIT compiler" ON)
if(SYCL_ENABLE_KERNEL_FUSION AND WIN32)
  message(WARNING "Kernel fusion not yet supported on Windows")
  set(SYCL_ENABLE_KERNEL_FUSION OFF CACHE 
      BOOL "Kernel fusion not yet supported on Windows" FORCE)
endif()

# Option for enabling building the SYCL major release preview library.
option(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB "Enable build of the SYCL major release preview library" ON)

# Needed for feature_test.hpp
if ("cuda" IN_LIST SYCL_ENABLE_PLUGINS)
  set(SYCL_BUILD_PI_CUDA ON)
endif()
if ("hip" IN_LIST SYCL_ENABLE_PLUGINS)
  set(SYCL_BUILD_PI_HIP ON)
endif()
if ("opencl" IN_LIST SYCL_ENABLE_PLUGINS)
  set(SYCL_BUILD_PI_OPENCL ON)
endif()
if ("level_zero" IN_LIST SYCL_ENABLE_PLUGINS)
  set(SYCL_BUILD_PI_LEVEL_ZERO ON)
endif()
if ("native_cpu" IN_LIST SYCL_ENABLE_PLUGINS)
  set(SYCL_BUILD_NATIVE_CPU ON)
endif()

# Configure SYCL version macro
set(sycl_inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(sycl_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(sycl_plugin_dir ${CMAKE_CURRENT_SOURCE_DIR}/plugins)
string(TIMESTAMP __SYCL_COMPILER_VERSION "%Y%m%d")
configure_file("source/version.hpp.in" "${SYCL_INCLUDE_BUILD_DIR}/sycl/version.hpp")
configure_file("source/feature_test.hpp.in" "${SYCL_INCLUDE_BUILD_DIR}/sycl/feature_test.hpp")

# Install generated headers.
install(FILES
  "${SYCL_INCLUDE_BUILD_DIR}/sycl/feature_test.hpp"
  "${SYCL_INCLUDE_BUILD_DIR}/sycl/version.hpp"
  DESTINATION "${SYCL_INCLUDE_DIR}/sycl"
  COMPONENT sycl-headers)

include(AddBoostMp11Headers)
include(FetchBoostUnorderedHeaders)

# This is workaround to detect changes (add or modify) in subtree which
# are not detected by copy_directory command.
# TODO: detect and process remove header/directory case
file(GLOB_RECURSE HEADERS_IN_SYCL_DIR CONFIGURE_DEPENDS "${sycl_inc_dir}/sycl/*")
file(GLOB_RECURSE HEADERS_IN_CL_DIR CONFIGURE_DEPENDS "${sycl_inc_dir}/CL/*")
file(GLOB_RECURSE HEADERS_IN_STD_DIR CONFIGURE_DEPENDS "${sycl_inc_dir}/std/*")
file(GLOB_RECURSE HEADERS_IN_SYCLCOMPAT_DIR CONFIGURE_DEPENDS "${sycl_inc_dir}/syclcompat/*" "${sycl_inc_dir}/syclcompat.hpp")

string(REPLACE "${sycl_inc_dir}" "${SYCL_INCLUDE_BUILD_DIR}"
  OUT_HEADERS_IN_SYCL_DIR "${HEADERS_IN_SYCL_DIR}")
string(REPLACE "${sycl_inc_dir}/CL" "${SYCL_INCLUDE_BUILD_DIR}/sycl/CL"
  OUT_HEADERS_IN_CL_DIR "${HEADERS_IN_CL_DIR}")
string(REPLACE "${sycl_inc_dir}" "${SYCL_INCLUDE_BUILD_DIR}"
  OUT_HEADERS_IN_STD_DIR "${HEADERS_IN_STD_DIR}")
string(REPLACE "${sycl_inc_dir}" "${SYCL_INCLUDE_BUILD_DIR}"
  OUT_HEADERS_IN_SYCLCOMPAT_DIR "${HEADERS_IN_SYCLCOMPAT_DIR}")

# Copy SYCL headers from sources to build directory
add_custom_target(sycl-headers
  DEPENDS ${OUT_HEADERS_IN_SYCL_DIR}
          ${OUT_HEADERS_IN_CL_DIR}
          ${OUT_HEADERS_IN_STD_DIR}
          ${OUT_HEADERS_IN_SYCLCOMPAT_DIR}
          boost_mp11-headers)

add_custom_command(
  OUTPUT  ${OUT_HEADERS_IN_SYCL_DIR}
          ${OUT_HEADERS_IN_CL_DIR}
          ${OUT_HEADERS_IN_STD_DIR}
          ${OUT_HEADERS_IN_SYCLCOMPAT_DIR}
  DEPENDS ${HEADERS_IN_SYCL_DIR}
          ${HEADERS_IN_CL_DIR}
          ${HEADERS_IN_STD_DIR}
          ${HEADERS_IN_SYCLCOMPAT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${sycl_inc_dir}/sycl ${SYCL_INCLUDE_BUILD_DIR}/sycl
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${sycl_inc_dir}/CL ${SYCL_INCLUDE_BUILD_DIR}/sycl/CL
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${sycl_inc_dir}/std ${SYCL_INCLUDE_BUILD_DIR}/std
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${sycl_inc_dir}/syclcompat ${SYCL_INCLUDE_BUILD_DIR}/syclcompat
  COMMAND ${CMAKE_COMMAND} -E copy ${sycl_inc_dir}/syclcompat.hpp ${SYCL_INCLUDE_BUILD_DIR}/syclcompat.hpp
  COMMENT "Copying SYCL headers ...")

# Copy SYCL headers from source to install directory
install(DIRECTORY "${sycl_inc_dir}/sycl" DESTINATION ${SYCL_INCLUDE_DIR} COMPONENT sycl-headers)
install(DIRECTORY "${sycl_inc_dir}/CL" DESTINATION ${SYCL_INCLUDE_DIR}/sycl COMPONENT sycl-headers)
install(DIRECTORY "${sycl_inc_dir}/std" DESTINATION ${SYCL_INCLUDE_DIR} COMPONENT sycl-headers)
install(DIRECTORY ${BOOST_MP11_DESTINATION_DIR} DESTINATION ${SYCL_INCLUDE_DIR}/sycl/detail COMPONENT boost_mp11-headers)
install(DIRECTORY "${sycl_inc_dir}/syclcompat" DESTINATION ${SYCL_INCLUDE_DIR} COMPONENT sycl-headers)
install(FILES "${sycl_inc_dir}/syclcompat.hpp" DESTINATION ${SYCL_INCLUDE_DIR} COMPONENT sycl-headers)

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
# Currently used by sycl and plugins library.
# Currently handles linking with libcxx support and gcc workaround
function( add_common_options LIB_NAME)
  if (SYCL_USE_LIBCXX)
    if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR
      (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
      if ((NOT (DEFINED SYCL_LIBCXX_INCLUDE_PATH)) OR (NOT (DEFINED SYCL_LIBCXX_LIBRARY_PATH)))
        message(FATAL_ERROR "When building with libc++ SYCL_LIBCXX_INCLUDE_PATHS and"
                            "SYCL_LIBCXX_LIBRARY_PATH should be set")
      endif()
      target_link_libraries(${LIB_NAME} PRIVATE "-L${SYCL_LIBCXX_LIBRARY_PATH}" -Wl,-rpath,${SYCL_LIBCXX_LIBRARY_PATH} -nodefaultlibs -lc++ -lc++abi -lm -lc -lgcc_s -lgcc)
      target_compile_options(${LIB_NAME} PRIVATE -nostdinc++)
      target_include_directories(${LIB_NAME} PRIVATE "${SYCL_LIBCXX_INCLUDE_PATH}")
      if (ARGC EQUAL 2)
        target_compile_options(${ARGV1} PRIVATE -nostdinc++)
        target_include_directories(${ARGV1} PRIVATE "${SYCL_LIBCXX_INCLUDE_PATH}")
      endif()
    else()
      message(FATAL_ERROR "Build with libc++ is not yet supported for this compiler")
    endif()
  else()
    # Workaround for bug in GCC version 5 and higher.
    # More information https://bugs.launchpad.net/ubuntu/+source/gcc-5/+bug/1568899
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
        CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
      target_link_libraries(${ARGV0} PRIVATE gcc_s gcc)
    endif()
  endif()
endfunction(add_common_options)

if (LLVM_ENABLE_ASSERTIONS AND NOT SYCL_DISABLE_STL_ASSERTIONS AND NOT WIN32)
  if(SYCL_USE_LIBCXX)
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

if (SYCL_ENABLE_XPTI_TRACING)
  set(XPTIFW_LIBS xpti xptifw)
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

add_custom_target( sycl-toolchain
  DEPENDS sycl-runtime-libraries
          sycl-compiler
          sycl-ls
          ${XPTIFW_LIBS}
  COMMENT "Building SYCL compiler toolchain..."
)

if (WIN32)
  add_dependencies(sycl-toolchain pi_win_proxy_loader)
endif()

# Enable new IN_LIST operator.
cmake_policy(SET CMP0057 NEW)

if (libdevice IN_LIST LLVM_ENABLE_PROJECTS)
  add_dependencies(sycl-toolchain libsycldevice)
endif()

if (SYCL_ENABLE_XPTI_TRACING)
  add_dependencies( sycl-toolchain xpti)
  if (MSVC)
    add_dependencies( sycl-toolchain xptid)
  endif()
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

# Plugin Library
add_subdirectory( plugins )

add_subdirectory(tools)

if (WIN32)
  add_subdirectory(pi_win_proxy_loader)
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
     boost_mp11-headers
     clang
     clang-offload-wrapper
     clang-offload-bundler
     clang-offload-deps
     clang-offload-extract
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
     clang-resource-headers
     OpenCL-Headers
     opencl-aot
     sycl-headers
     sycl-headers-extras
     sycl
     libsycldevice
     level-zero-sycl-dev
     ${XPTIFW_LIBS}
     ${SYCL_TOOLCHAIN_DEPS}
)

if (WIN32)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS pi_win_proxy_loader)
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

if("cuda" IN_LIST SYCL_ENABLE_PLUGINS)
  # Ensure that libclc is enabled.
  list(FIND LLVM_ENABLE_PROJECTS libclc LIBCLC_FOUND)
  if( LIBCLC_FOUND EQUAL -1 )
    message(FATAL_ERROR
      "CUDA support requires adding \"libclc\" to the CMake argument \"LLVM_ENABLE_PROJECTS\"")
  endif()

  add_dependencies(sycl-toolchain pi_cuda)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS pi_cuda)
endif()

if("hip" IN_LIST SYCL_ENABLE_PLUGINS)
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

  add_dependencies(sycl-toolchain pi_hip)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS pi_hip)
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
