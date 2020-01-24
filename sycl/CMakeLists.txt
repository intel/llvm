cmake_minimum_required(VERSION 3.2)

project(sycl-solution)
# Requirements
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
option(SYCL_ENABLE_WERROR "Treat all warnings as errors in SYCL project" OFF)

if(SYCL_ENABLE_WERROR)
  if(MSVC)
    set(CMAKE_CXX_FLAGS "/W4 /WX ${CMAKE_CXX_FLAGS}")
    add_definitions(
      -wd4996 # Suppress 'function': was declared deprecated'
    )
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wno-deprecated-declarations")
  endif()
endif()

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

set(LLVM_INST_INC_DIRECTORY "lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION}/include")
set(dst_dir ${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION}/include)

# Find OpenCL headers and libraries installed in the system and use them to
# build SYCL runtime.
# WARNING: use with caution, building SYCL runtime with OpenCL implementation
# instead of Khronos ICD loader might cause build and/or portability issues.
option(OpenCL_BUILD_WITH_SYSTEM_SDK OFF)

if( (OpenCL_INCLUDE_DIR AND OpenCL_LIBRARY) OR OpenCL_BUILD_WITH_SYSTEM_SDK)
  find_package(OpenCL)
endif()

include(ExternalProject)

if( NOT OpenCL_INCLUDE_DIRS )
  message("OpenCL_INCLUDE_DIRS is missed. Will try to download OpenCL headers from github.com")
  set(OpenCL_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/OpenCL/inc")
  ExternalProject_Add(ocl-headers
    GIT_REPOSITORY    https://github.com/KhronosGroup/OpenCL-Headers.git
    GIT_TAG           origin/master
    SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/OpenCL/inc"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ${CMAKE_COMMAND} -E copy_directory ${OpenCL_INCLUDE_DIRS}/CL ${dst_dir}/CL
    INSTALL_COMMAND   ""
    STEP_TARGETS      build
    COMMENT           "Downloading OpenCL headers."
  )
else()
  add_custom_target( ocl-headers ALL
    DEPENDS ${OpenCL_INCLUDE_DIRS}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${OpenCL_INCLUDE_DIRS}/CL ${dst_dir}/CL
    COMMENT "Copying OpenCL headers ..."
  )
endif()

if( NOT OpenCL_LIBRARIES )
  message("OpenCL_LIBRARIES is missed. Will try to download OpenCL ICD Loader from github.com")
  if(MSVC)
    set(OpenCL_LIBRARIES
        "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}OpenCL${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(AUX_CMAKE_FLAGS "${AUX_CMAKE_FLAGS} -DOPENCL_ICD_LOADER_REQUIRE_WDK=OFF")
  else()
    set(OpenCL_LIBRARIES
        "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}OpenCL${CMAKE_SHARED_LIBRARY_SUFFIX}")
  endif()
  if (CMAKE_C_COMPILER)
      set(AUX_CMAKE_FLAGS "${AUX_CMAKE_FLAGS} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
  endif()
  if (CMAKE_CXX_COMPILER)
      set(AUX_CMAKE_FLAGS "${AUX_CMAKE_FLAGS} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
  endif()
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/icd_build)
  ExternalProject_Add(ocl-icd
    GIT_REPOSITORY    https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
    GIT_TAG           origin/master
    SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/OpenCL/icd"
    BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/icd_build"
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
               -DOPENCL_ICD_LOADER_HEADERS_DIR=${CMAKE_CURRENT_BINARY_DIR}/OpenCL/inc
               -DCMAKE_INSTALL_PREFIX=${LLVM_BINARY_DIR}
               -DCMAKE_INSTALL_LIBDIR:PATH=lib
               ${AUX_CMAKE_FLAGS}
    STEP_TARGETS      configure,build,install
    DEPENDS           ocl-headers
    BUILD_BYPRODUCTS ${OpenCL_LIBRARIES}
  )
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
  DESTINATION ${LLVM_INST_INC_DIRECTORY}
  COMPONENT opencl-headers
)

# Configure SYCL version macro
set(sycl_inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)
string(TIMESTAMP __SYCL_COMPILER_VERSION "%Y%m%d")
set(version_header "${sycl_inc_dir}/CL/sycl/version.hpp")
configure_file("${version_header}.in" "${version_header}")

# Copy SYCL headers
add_custom_target(sycl-headers ALL
COMMAND ${CMAKE_COMMAND} -E copy_directory ${sycl_inc_dir}/CL ${dst_dir}/CL
COMMENT "Copying SYCL headers ...")

# Configure SYCL headers
install(DIRECTORY "${sycl_inc_dir}/." DESTINATION "${LLVM_INST_INC_DIRECTORY}" COMPONENT sycl-headers)

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
          target_compile_options(${LIB_NAME} PRIVATE -nostdinc++)
          if ((NOT (DEFINED SYCL_LIBCXX_INCLUDE_PATH)) OR (NOT (DEFINED SYCL_LIBCXX_LIBRARY_PATH)))
              message(FATAL_ERROR "When building with libc++ SYCL_LIBCXX_INCLUDE_PATHS and"
                                  "SYCL_LIBCXX_LIBRARY_PATH should be set")
          endif()
          target_include_directories(${LIB_NAME} PRIVATE "${SYCL_LIBCXX_INCLUDE_PATH}")
          target_link_libraries(${LIB_NAME} PRIVATE "-L${SYCL_LIBCXX_LIBRARY_PATH}" -nodefaultlibs -lc++ -lc++abi -lm -lc -lgcc_s -lgcc)
      else()
          message(FATAL_ERROR "Build with libc++ is not yet supported for this compiler")
      endif()
  else()

# Workaround for bug in GCC version 5 and higher.
# More information https://bugs.launchpad.net/ubuntu/+source/gcc-5/+bug/1568899
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
        CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
    target_link_libraries(${LIB_NAME} PRIVATE gcc_s gcc)
    endif()

  endif()
endfunction(add_common_options)

# SYCL runtime library
add_subdirectory( source )

# SYCL toolchain builds all components: compiler, libraries, headers, etc.
add_custom_target( sycl-toolchain
  DEPENDS ${SYCL_RT_LIBS}
          pi_opencl
          clang
          clang-offload-wrapper
          clang-offload-bundler
          llc
          llvm-ar
          llvm-foreach
          llvm-no-spir-kernel
          llvm-spirv
          llvm-link
          llvm-objcopy
          sycl-post-link
  COMMENT "Building SYCL compiler toolchain..."
)

if (NOT DEFINED LLVM_INCLUDE_TESTS)
  set(LLVM_INCLUDE_TESTS ON)
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
     llc
     llvm-ar
     llvm-foreach
     llvm-no-spir-kernel
     llvm-spirv
     llvm-link
     llvm-objcopy
     sycl-post-link
     clang-resource-headers
     opencl-headers
     sycl-headers
     sycl
     pi_opencl
)

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
