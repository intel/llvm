cmake_minimum_required(VERSION 3.2)

project(sycl-solution)
# Requirements
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

option(SYCL_DEPLOY_OPENCL OFF)

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

find_package(OpenCL)

include(ExternalProject)

if( NOT OpenCL_INCLUDE_DIRS )
  message ("OpenCL_INCLUDE_DIRS is missed. Try to download headers from github.com")
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
  message("OpenCL header have been found under ${OpenCL_INCLUDE_DIRS}.")
  add_custom_target( ocl-headers ALL
    DEPENDS ${OpenCL_INCLUDE_DIRS}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${OpenCL_INCLUDE_DIRS}/CL ${dst_dir}/CL
    COMMENT "Copying OpenCL headers ..."
  )
endif()

if( NOT OpenCL_LIBRARIES )
  message("OpenCL_LIBRARIES is missed. Try to build from GitHub sources.")
  set(OpenCL_LIBRARIES "${LLVM_LIBRARY_OUTPUT_INTDIR}/libOpenCL.so")
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/icd_build)
  ExternalProject_Add(ocl-icd
    GIT_REPOSITORY    https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
    GIT_TAG           origin/master
    SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/OpenCL/icd"
    BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/icd_build"
    CONFIGURE_COMMAND ${CMAKE_COMMAND} "${CMAKE_CURRENT_BINARY_DIR}/OpenCL/icd" -DCMAKE_INSTALL_PREFIX=${LLVM_BINARY_DIR}
    BUILD_COMMAND     make C_INCLUDE_PATH=${CMAKE_CURRENT_BINARY_DIR}/OpenCL/inc
    INSTALL_COMMAND   make install 
    STEP_TARGETS      configure,build,install
    DEPENDS           ocl-headers
  )
else()
  message("OpenCL loader has been found: ${OpenCL_LIBRARIES}.")
  file(GLOB ICD_LOADER_SRC "${OpenCL_LIBRARIES}*")
  file(COPY ${ICD_LOADER_SRC} DESTINATION ${LLVM_LIBRARY_OUTPUT_INTDIR})
  add_custom_target(ocl-icd DEPENDS ${OpenCL_LIBRARIES} COMMENT "Copying OpenCL ICD Loader ...")
endif()

if(NOT TARGET OpenCL)
  add_library(OpenCL INTERFACE)
  if (TARGET OpenCL::OpenCL)
    # Since 3.8 CMake exports target for FindOpenCL
    target_link_libraries(OpenCL INTERFACE OpenCL::OpenCL)
  else()
    target_link_libraries(OpenCL INTERFACE ${OpenCL_LIBRARIES})
    target_include_directories(OpenCL INTERFACE ${OpenCL_INCLUDE_DIRS})
  endif()
endif()

set(OPENCL_INCLUDE "${OpenCL_INCLUDE_DIRS}")

# Configure SYCL version macro
set(sycl_inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)
string(TIMESTAMP __SYCL_COMPILER_VERSION "%Y%m%d")
set(version_header "${sycl_inc_dir}/CL/sycl/version.hpp")
configure_file("${version_header}.in" "${version_header}")

# Copy SYCL headers
add_custom_target(sycl-headers ALL
COMMAND ${CMAKE_COMMAND} -E copy_directory ${sycl_inc_dir}/CL ${dst_dir}/CL
COMMENT "Copying SYCL headers ...")

# SYCL runtime library
add_subdirectory(source)

# SYCL toolchain builds all components: compiler, libraries, headers, etc.
set(SYCL_TOOLCHAIN_DEPS
  sycl
  clang
  clang-offload-wrapper
  clang-offload-bundler
  llc
  llvm-as
  llvm-ar
  llvm-dis
  llvm-no-spir-kernel
  llvm-spirv
  llvm-link
  llvm-objcopy
  opt)
add_custom_target( sycl-toolchain
  DEPENDS ${SYCL_TOOLCHAIN_DEPS}
  COMMENT "Building SYCL compiler toolchain..."
)

add_subdirectory( test )
add_subdirectory( unittests )
add_subdirectory( tools )

# Configure SYCL toolchain installation targets
add_custom_target(deploy-sycl-toolchain)

set(SYCL_BUNDLE_DEPS ${SYCL_TOOLCHAIN_DEPS} opencl clang-resource-headers)

foreach(comp ${SYCL_BUNDLE_DEPS})
    add_custom_command(TARGET deploy-sycl-toolchain PRE_BUILD
        COMMAND ${CMAKE_COMMAND}
        ARGS -DCOMPONENT="${comp}" -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
endforeach()

set(LLVM_LIT_EXT "")
if(WIN32)
  set(LLVM_LIT_EXT ".py")
endif()
add_custom_command(TARGET deploy-sycl-toolchain PRE_BUILD
  COMMAND ${CMAKE_COMMAND}
  ARGS -E copy ${CMAKE_BINARY_DIR}/bin/llvm-lit${LLVM_LIT_EXT}
  ${CMAKE_INSTALL_PREFIX}/bin/llvm-lit${LLVM_LIT_EXT})

# For some reason clang-cl is being created on Linux. Remove it so users
# are not confused.
if (NOT WIN32)
  add_custom_command(TARGET deploy-sycl-toolchain POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E remove ${CMAKE_INSTALL_PREFIX}/bin/clang-cl)
endif()

if (SYCL_DEPLOY_OPENCL)
  install(TARGETS OpenCL EXPORT SYCL
      LIBRARY
        DESTINATION lib
        COMPONENT opencl
      ARCHIVE
        DESTINATION lib
        COMPONENT opencl
      RUNTIME
        DESTINATION bin
        COMPONENT opencl
      INCLUDES
      DESTINATION include
        COMPONENT opencl)

  # If OpenCL target is not built with SYCL, we need to copy libraries manually
  if(OpenCL_LIBRARIES)
    set (RESOLVED_LIBS "")
    foreach (file ${OpenCL_LIBRARIES})
      get_filename_component(resolved_lib "${file}" REALPATH)
      list (APPEND RESOLVED_LIBS "${resolved_lib}")
    endforeach()
    install(FILES ${RESOLVED_LIBS} DESTINATION lib COMPONENT opencl)
    if(WIN32)
      string(REPLACE ".lib" ".dll" DLL_LIBS "${RESOLVED_LIBS}")
      install(FILES ${DLL_LIBS} DESTINATION bin COMPONENT opencl)
    else(WIN32)
      if(RESOLVED_LIBS MATCHES "libOpenCL.so.1.0.0")
        # Workaround for some OpenCL ICD packages
        add_custom_command(TARGET deploy-sycl-toolchain POST_BUILD
            COMMAND ${CMAKE_COMMAND}
          ARGS -E rename ${CMAKE_INSTALL_PREFIX}/lib/libOpenCL.so.1.0.0
          ${CMAKE_INSTALL_PREFIX}/lib/libOpenCL.so)
      endif()
      add_custom_command(TARGET deploy-sycl-toolchain
        COMMAND ${CMAKE_COMMAND}
        ARGS -E create_symlink ${CMAKE_INSTALL_PREFIX}/lib/libOpenCL.so
        ${CMAKE_INSTALL_PREFIX}/lib/libOpenCL.so.1)
      add_custom_command(TARGET deploy-sycl-toolchain
        COMMAND ${CMAKE_COMMAND}
        ARGS -E create_symlink ${CMAKE_INSTALL_PREFIX}/lib/libOpenCL.so
        ${CMAKE_INSTALL_PREFIX}/lib/libOpenCL.so.1.2)
    endif()
  endif()
  if(OpenCL_INCLUDE_DIRS)
      install(DIRECTORY ${OpenCL_INCLUDE_DIRS}/CL
          DESTINATION include COMPONENT opencl)
  endif()
endif()

