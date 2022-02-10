cmake_minimum_required(VERSION 3.14)

project(sycl-solution)
# Requirements
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(SYCL_ENABLE_WERROR "Treat all warnings as errors in SYCL project" OFF)
option(SYCL_DISABLE_STL_ASSERTIONS "Disable assertions in STL containers" OFF)
option(SYCL_ADD_DEV_VERSION_POSTFIX "Adds -V postfix to version string" ON)
option(SYCL_ENABLE_COVERAGE "Enables code coverage for runtime and unit tests" OFF)

if (NOT SYCL_COVERAGE_PATH)
  set(SYCL_COVERAGE_PATH "${CMAKE_CURRENT_BINARY_DIR}/profiles")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddSYCLExecutable)
include(SYCLUtils)

set(SYCL_MAJOR_VERSION 5)
set(SYCL_MINOR_VERSION 5)
set(SYCL_PATCH_VERSION 0)
set(SYCL_DEV_ABI_VERSION 0)
if (SYCL_ADD_DEV_VERSION_POSTFIX)
  set(SYCL_VERSION_POSTFIX "-${SYCL_DEV_ABI_VERSION}")
endif()
set(SYCL_VERSION_STRING "${SYCL_MAJOR_VERSION}.${SYCL_MINOR_VERSION}.${SYCL_PATCH_VERSION}${SYCL_VERSION_POSTFIX}")

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
  set(LLVM_REQUIRES_EH ON)
  set(LLVM_REQUIRES_RTTI ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")

  # Add PDB debug information
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(LLVMCheckLinkerFlag)
  llvm_check_linker_flag(CXX "/DEBUG" LINKER_SUPPORTS_DEBUG)
  if(LINKER_SUPPORTS_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")
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

option(SYCL_BUILD_PI_CUDA
  "Enables the CUDA backend for the Plugin Interface" OFF)

option(SYCL_BUILD_PI_HIP
  "Enables the HIP backend for the Plugin Interface" OFF)

# Configure SYCL version macro
set(sycl_inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(sycl_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(sycl_plugin_dir ${CMAKE_CURRENT_SOURCE_DIR}/plugins)
string(TIMESTAMP __SYCL_COMPILER_VERSION "%Y%m%d")
set(version_header "${sycl_inc_dir}/CL/sycl/version.hpp")
configure_file("${version_header}.in" "${version_header}")
set(feature_header "${sycl_inc_dir}/CL/sycl/feature_test.hpp")
configure_file("${feature_header}.in" "${feature_header}")

# This is workaround to detect changes (add or modify) in subtree which
# are not detected by copy_directory command.
# TODO: detect and process remove header/directory case
file(GLOB_RECURSE HEADERS_IN_SYCL_DIR CONFIGURE_DEPENDS "${sycl_inc_dir}/sycl/*")
file(GLOB_RECURSE HEADERS_IN_CL_DIR CONFIGURE_DEPENDS "${sycl_inc_dir}/CL/*")
string(REPLACE "${sycl_inc_dir}" "${SYCL_INCLUDE_BUILD_DIR}"
  OUT_HEADERS_IN_SYCL_DIR "${HEADERS_IN_SYCL_DIR}")
string(REPLACE "${sycl_inc_dir}/CL" "${SYCL_INCLUDE_BUILD_DIR}/sycl/CL"
  OUT_HEADERS_IN_CL_DIR "${HEADERS_IN_CL_DIR}")

# Copy SYCL headers from sources to build directory
add_custom_target(sycl-headers
  DEPENDS ${OUT_HEADERS_IN_SYCL_DIR}
          ${OUT_HEADERS_IN_CL_DIR})

add_custom_command(
  OUTPUT  ${OUT_HEADERS_IN_SYCL_DIR}
          ${OUT_HEADERS_IN_CL_DIR}
  DEPENDS ${HEADERS_IN_SYCL_DIR}
          ${HEADERS_IN_CL_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${sycl_inc_dir}/sycl ${SYCL_INCLUDE_BUILD_DIR}/sycl
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${sycl_inc_dir}/CL ${SYCL_INCLUDE_BUILD_DIR}/sycl/CL
  COMMENT "Copying SYCL headers ...")

# Copy SYCL headers from source to install directory
install(DIRECTORY "${sycl_inc_dir}/sycl" DESTINATION ${SYCL_INCLUDE_DIR} COMPONENT sycl-headers)
install(DIRECTORY "${sycl_inc_dir}/CL" DESTINATION ${SYCL_INCLUDE_DIR}/sycl COMPONENT sycl-headers)

set(SYCL_RT_LIBS sycl)
if (MSVC)
  list(APPEND SYCL_RT_LIBS sycld)
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
          file-table-tform
          llc
          llvm-ar
          llvm-foreach
          llvm-no-spir-kernel
          llvm-spirv
          llvm-link
          llvm-objcopy
          spirv-to-ir-wrapper
          sycl-post-link
          opencl-aot
)

add_custom_target( sycl-toolchain
  DEPENDS ${SYCL_RT_LIBS}
          sycl-compiler
          sycl-ls
          ${XPTIFW_LIBS}
  COMMENT "Building SYCL compiler toolchain..."
)

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

if(SYCL_INCLUDE_TESTS)
  if(NOT LLVM_INCLUDE_TESTS)
      message(FATAL_ERROR
        "Can't build SYCL tests without LLVM_INCLUDE_TESTS enabled.")
  endif()
  if(EXISTS ${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include/gtest/gtest.h)
    add_subdirectory(unittests)
    list(APPEND SYCL_TEST_DEPS SYCLUnitTests)
  endif()
  add_subdirectory(test)
endif()

# Package deploy support
# Listed here are component names contributing the package
set( SYCL_TOOLCHAIN_DEPLOY_COMPONENTS
     append-file
     clang
     clang-offload-wrapper
     clang-offload-bundler
     clang-offload-deps
     clang-offload-extract
     file-table-tform
     level-zero-loader
     level-zero-headers
     llc
     llvm-ar
     llvm-foreach
     llvm-no-spir-kernel
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
     pi_opencl
     pi_level_zero
     libsycldevice
     ${XPTIFW_LIBS}
)

if(OpenCL_INSTALL_KHRONOS_ICD_LOADER AND TARGET OpenCL-ICD)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS OpenCL-ICD)
endif()

if(SYCL_BUILD_PI_CUDA)
  # Ensure that libclc is enabled.
  list(FIND LLVM_ENABLE_PROJECTS libclc LIBCLC_FOUND)
  if( LIBCLC_FOUND EQUAL -1 )
    message(FATAL_ERROR
      "CUDA support requires adding \"libclc\" to the CMake argument \"LLVM_ENABLE_PROJECTS\"")
  endif()

  add_dependencies(sycl-toolchain libspirv-builtins pi_cuda)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS libspirv-builtins pi_cuda)
endif()

if(SYCL_BUILD_PI_HIP)
  # Ensure that libclc is enabled.
  list(FIND LLVM_ENABLE_PROJECTS libclc LIBCLC_FOUND)
  if( LIBCLC_FOUND EQUAL -1 )
    message(FATAL_ERROR
      "HIP support requires adding \"libclc\" to the CMake argument \"LLVM_ENABLE_PROJECTS\"")
  endif()

  add_dependencies(sycl-toolchain libspirv-builtins pi_hip)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS libspirv-builtins pi_hip)
endif()

if(SYCL_BUILD_PI_ESIMD_EMULATOR)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS pi_esimd_emulator libcmrt-headers)
  if (MSVC)
    list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS libcmrt-libs libcmrt-dlls)
  else()
    list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS libcmrt-sos)
  endif()
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

add_subdirectory(examples)
