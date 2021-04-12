cmake_minimum_required(VERSION 3.14)

project(sycl-solution)
# Requirements
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(SYCL_ENABLE_WERROR "Treat all warnings as errors in SYCL project" OFF)
option(SYCL_DISABLE_STL_ASSERTIONS "Disable assertions in STL containers" OFF)
option(SYCL_ADD_DEV_VERSION_POSTFIX "Adds -V postfix to version string" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddSYCLExecutable)

set(SYCL_MAJOR_VERSION 5)
set(SYCL_MINOR_VERSION 1)
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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
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
set(SYCL_INCLUDE_DEPLOY_DIR ${CMAKE_INSTALL_PREFIX}/${SYCL_INCLUDE_DIR})

# Find OpenCL headers and libraries installed in the system and use them to
# build SYCL runtime.
# WARNING: use with caution, building SYCL runtime with OpenCL implementation
# instead of Khronos ICD loader might cause build and/or portability issues.
option(OpenCL_BUILD_WITH_SYSTEM_SDK "Build SYCL with OpenCL coming from the build system" OFF)

if( (OpenCL_INCLUDE_DIR AND OpenCL_LIBRARY) OR OpenCL_BUILD_WITH_SYSTEM_SDK)
  find_package(OpenCL)
endif()

include(ExternalProject)

# By default, whenever cmake is re-run, the external project's sources will be
# updated, which might be annoying - let's have a possibility to disable that as
# not every external project which is used here is being updated often
# Single toggle to control all external projects
option(SYCL_EXTERNAL_PROJECTS_SKIP_AUTO_UPDATE
  "Perform automatic update step for external projects (OpenCL Headers, OpenCL ICD, etc.)" OFF)
# More precise controls for each particular external project are documented
# near each ExternalProject_Add

if( NOT OpenCL_INCLUDE_DIRS )
  message(STATUS "OpenCL_INCLUDE_DIRS is missing. Will try to download OpenCL headers from github.com")
  set(OpenCL_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/OpenCL/inc")
  if (NOT DEFINED SYCL_EP_OCL_HEADERS_SKIP_AUTO_UPDATE)
    set(SYCL_EP_OCL_HEADERS_SKIP_AUTO_UPDATE ${SYCL_EXTERNAL_PROJECTS_SKIP_AUTO_UPDATE})
  endif()
  ExternalProject_Add(ocl-headers
    GIT_REPOSITORY    https://github.com/KhronosGroup/OpenCL-Headers.git
    GIT_TAG           23710f1b99186065c1768fc3098ba681adc0f253
    UPDATE_DISCONNECTED ${SYCL_EP_OCL_HEADERS_SKIP_AUTO_UPDATE}
    SOURCE_DIR        ${OpenCL_INCLUDE_DIRS}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ${CMAKE_COMMAND} -E copy_directory ${OpenCL_INCLUDE_DIRS}/CL ${SYCL_INCLUDE_BUILD_DIR}/sycl/CL
    INSTALL_COMMAND   ""
    STEP_TARGETS      build
    COMMENT           "Downloading OpenCL headers."
    LOG_DOWNLOAD 1
    LOG_UPDATE 1
    LOG_BUILD 1
  )
  add_definitions(-DCL_TARGET_OPENCL_VERSION=220)
else()
  add_custom_target( ocl-headers ALL
    DEPENDS ${OpenCL_INCLUDE_DIRS}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${OpenCL_INCLUDE_DIRS}/CL ${SYCL_INCLUDE_BUILD_DIR}/sycl/CL
    COMMENT "Copying OpenCL headers ..."
  )
endif()

if( NOT OpenCL_LIBRARIES )
  message(STATUS "OpenCL_LIBRARIES is missing. Will try to download OpenCL ICD Loader from github.com")
  if(MSVC)
    set(OpenCL_LIBRARIES
        "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}OpenCL${CMAKE_STATIC_LIBRARY_SUFFIX}")
    list(APPEND AUX_CMAKE_FLAGS -DOPENCL_ICD_LOADER_REQUIRE_WDK=OFF)
  else()
    set(OpenCL_LIBRARIES
        "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}OpenCL${CMAKE_SHARED_LIBRARY_SUFFIX}")
  endif()
  if (CMAKE_C_COMPILER)
    list(APPEND AUX_CMAKE_FLAGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
  endif()
  if (CMAKE_CXX_COMPILER)
    list(APPEND AUX_CMAKE_FLAGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
  endif()
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/icd_build)
  set(OpenCL_ICD_LOADER_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/OpenCL/icd")
  if (NOT DEFINED SYCL_EP_OCL_ICD_SKIP_AUTO_UPDATE)
    set(SYCL_EP_OCL_ICD_SKIP_AUTO_UPDATE ${SYCL_EXTERNAL_PROJECTS_SKIP_AUTO_UPDATE})
  endif()
  ExternalProject_Add(ocl-icd
    GIT_REPOSITORY    https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
    GIT_TAG           5f8249691ec8c25775789498951f8e9eb62c201d
    UPDATE_DISCONNECTED ${SYCL_EP_OCL_ICD_SKIP_AUTO_UPDATE}
    SOURCE_DIR        ${OpenCL_ICD_LOADER_SOURCE_DIR}
    BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/icd_build"
    INSTALL_DIR       "${CMAKE_CURRENT_BINARY_DIR}/icd_install"
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
               -DOPENCL_ICD_LOADER_HEADERS_DIR=${OpenCL_INCLUDE_DIRS}
               -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
               -DCMAKE_INSTALL_LIBDIR:PATH=lib${LLVM_LIBDIR_SUFFIX}
               ${AUX_CMAKE_FLAGS}
    STEP_TARGETS      configure,build,install
    DEPENDS           ocl-headers
    BUILD_BYPRODUCTS ${OpenCL_LIBRARIES}
    LOG_DOWNLOAD 1
    LOG_UPDATE 1
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1
  )
  ExternalProject_Add_Step(ocl-icd llvminstall
    COMMAND ${CMAKE_COMMAND} -E copy_directory <INSTALL_DIR>/ ${LLVM_BINARY_DIR}
    COMMENT "Installing ocl-icd into the LLVM binary directory"
    DEPENDEES install
  )
  # Optionally install the ICD library along with LLVM.
  option(OpenCL_INSTALL_KHRONOS_ICD_LOADER
    "Install the Khronos ICD Loader built as part of LLVM" OFF)
  if (OpenCL_INSTALL_KHRONOS_ICD_LOADER)
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/icd_install/"
      DESTINATION "."
      COMPONENT opencl-icd
    )
  endif()
else()
  file(GLOB ICD_LOADER_SRC "${OpenCL_LIBRARIES}*")
  file(COPY ${ICD_LOADER_SRC} DESTINATION ${LLVM_LIBRARY_OUTPUT_INTDIR})
  add_custom_target(ocl-icd DEPENDS ${OpenCL_LIBRARIES} COMMENT "Copying OpenCL ICD Loader ...")
endif()

set(OPENCL_INCLUDE "${OpenCL_INCLUDE_DIRS}")

add_library (OpenCL-Headers INTERFACE)
add_library (OpenCL::Headers ALIAS OpenCL-Headers)
target_include_directories(OpenCL-Headers
  INTERFACE ${OPENCL_INCLUDE}
)
install(DIRECTORY ${OPENCL_INCLUDE}/CL
  DESTINATION ${SYCL_INCLUDE_DEPLOY_DIR}/sycl
  COMPONENT opencl-headers
)

option(SYCL_BUILD_PI_CUDA
  "Enables the CUDA backend for the Plugin Interface" OFF)

# Configure SYCL version macro
set(sycl_inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(sycl_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(sycl_plugin_dir ${CMAKE_CURRENT_SOURCE_DIR}/plugins)
string(TIMESTAMP __SYCL_COMPILER_VERSION "%Y%m%d")
set(version_header "${sycl_inc_dir}/CL/sycl/version.hpp")
configure_file("${version_header}.in" "${version_header}")

# Copy SYCL headers
add_custom_target(sycl-headers ALL
COMMAND ${CMAKE_COMMAND} -E copy_directory ${sycl_inc_dir}/sycl ${SYCL_INCLUDE_BUILD_DIR}/sycl
COMMAND ${CMAKE_COMMAND} -E copy_directory ${sycl_inc_dir}/CL ${SYCL_INCLUDE_BUILD_DIR}/sycl/CL
COMMENT "Copying SYCL headers ...")

# Configure SYCL headers
install(DIRECTORY "${sycl_inc_dir}/sycl" DESTINATION ${SYCL_INCLUDE_DEPLOY_DIR} COMPONENT sycl-headers)
install(DIRECTORY "${sycl_inc_dir}/CL" DESTINATION ${SYCL_INCLUDE_DEPLOY_DIR}/sycl COMPONENT sycl-headers)

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

if (LLVM_ENABLE_ASSERTIONS AND NOT SYCL_DISABLE_STL_ASSERTIONS)
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
  set(XPTIFW_LIBS xptifw)
endif()

# SYCL toolchain builds all components: compiler, libraries, headers, etc.
add_custom_target( sycl-toolchain
  DEPENDS ${SYCL_RT_LIBS}
          clang
          clang-offload-wrapper
          clang-offload-bundler
          clang-offload-deps
          file-table-tform
          llc
          llvm-ar
          llvm-foreach
          llvm-no-spir-kernel
          llvm-spirv
          llvm-link
          llvm-objcopy
          sycl-post-link
          sycl-ls
          ${XPTIFW_LIBS}
  COMMENT "Building SYCL compiler toolchain..."
)

# Check if opencl-aot tool is enabled.
list(FIND LLVM_ENABLE_PROJECTS opencl-aot OPENCL_AOT_FOUND)
if(NOT OPENCL_AOT_FOUND EQUAL -1)
  add_dependencies(sycl-toolchain opencl-aot)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS opencl-aot)
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

# Plugin Library
add_subdirectory( plugins )

add_subdirectory(tools)

if(SYCL_INCLUDE_TESTS)
  if(EXISTS ${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include/gtest/gtest.h)
    add_subdirectory(unittests)
    list(APPEND SYCL_TEST_DEPS SYCLUnitTests)
  endif()
  add_subdirectory(test)
endif()

# Package deploy support
# Listed here are component names contributing the package
set( SYCL_TOOLCHAIN_DEPLOY_COMPONENTS
     clang
     clang-offload-wrapper
     clang-offload-bundler
     clang-offload-deps
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
     sycl-post-link
     sycl-ls
     clang-resource-headers
     opencl-headers
     sycl-headers
     sycl-headers-extras
     sycl
     pi_opencl
     pi_level_zero
     libsycldevice
     ${XPTIFW_LIBS}
)

if(OpenCL_INSTALL_KHRONOS_ICD_LOADER AND TARGET ocl-icd)
  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS opencl-icd)
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
