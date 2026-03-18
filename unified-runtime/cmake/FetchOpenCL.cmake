# Finds or fetches OpenCL Headers and the ICD loader.
if(TARGET OpenCL)
  # If we already ran this module (so the OpenCL target exists),
  # everything is already set up, nothing to do.
  return()
endif()

# Repo URLs

set(OCL_HEADERS_REPO
  "https://github.com/KhronosGroup/OpenCL-Headers.git")
set(OCL_LOADER_REPO
  "https://github.com/KhronosGroup/OpenCL-ICD-Loader.git")

# Repo tags/hashes

set(OCL_HEADERS_TAG v2025.07.22)
set(OCL_LOADER_TAG v2025.07.22)

# We have a unique use case where even if we find a system install with the
# correct OpenCL version, it may not support the extension we need.
# We can't use find_package because that doesn't allow us to undefine targets if it
# turns out we can't use the system install, so instead use find_path and find_library
# which have validator functions (starting in version 3.25.0) that do exactly what we want.
find_path(OpenCL_INCLUDE_DIR
  NAMES
    CL/cl.h OpenCL/cl.h)
if(OpenCL_INCLUDE_DIR AND
    CMAKE_VERSION VERSION_GREATER_EQUAL "3.25.0")
  function(opencl_validator validator_result_var item)
    # The OpenCL-Headers CMake files don't provide granular info
    # on what is and isn't supposed, just the overall OpenCL version.
    # The current tag we are using happens to define an extension, so just check
    # if that extension exists to make sure the system install is not
    # too old.
    set(OPENCL_TEST_PROGRAM "#define CL_TARGET_OPENCL_VERSION 300
        #include <CL/cl_ext.h>
        #ifndef cl_khr_spirv_queries
        #error Unsupported header version
        #endif
        int main(int, char*[]) { return 0; }"
      )
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES ${OpenCL_INCLUDE_DIR})
    set(CMAKE_REQUIRED_LIBRARIES ${item})
    check_cxx_source_compiles("${OPENCL_TEST_PROGRAM}" OPENCL_HEADERS_VERSION_SUPPORTED)
    if(NOT OPENCL_HEADERS_VERSION_SUPPORTED)
      message(WARNING "Preinstalled OpenCL-Headers are not supported, "
	"use commit ${OCL_HEADERS_TAG} or later. Will fetch OpenCL.")
      set(${validator_result_var} FALSE PARENT_SCOPE)
    endif()
  endfunction()
  find_library(OpenCL_LIBRARY_CAND
    NAMES OpenCL
    VALIDATOR opencl_validator)
  if(OpenCL_LIBRARY_CAND)
    add_library(OpenCL INTERFACE)
    target_include_directories(OpenCL INTERFACE ${OpenCL_INCLUDE_DIR})
    target_link_libraries(OpenCL INTERFACE ${OpenCL_LIBRARY_CAND})
    # Signal to callers that we are using a system OpeNCL install so OpenCL
    # doesn't need to be installed as part of their install step.
    set(OpenCL_FOUND TRUE CACHE BOOL "" FORCE)
  endif()
endif()

# OpenCL Headers
if(NOT TARGET OpenCL)
  FetchContent_Declare(ocl-headers
      GIT_REPOSITORY    ${OCL_HEADERS_REPO}
      GIT_TAG           ${OCL_HEADERS_TAG}
  )
  FetchContent_GetProperties(ocl-headers)

  if(NOT ocl-headers_POPULATED)
    message(STATUS "Will fetch OpenCL headers from ${OCL_HEADERS_REPO}")
  endif()
  FetchContent_MakeAvailable(ocl-headers)
  set(OpenCL_INCLUDE_DIR ${ocl-headers_SOURCE_DIR} CACHE PATH "" FORCE)
else()
  message(STATUS "Using OpenCL headers at ${OpenCL_INCLUDE_DIR}")
endif()

# OpenCL Library (ICD Loader)

set(BUILD_SHARED_LIBS ON)

if(NOT TARGET OpenCL)
  FetchContent_Declare(ocl-icd
      GIT_REPOSITORY    ${OCL_LOADER_REPO}
      GIT_TAG           ${OCL_LOADER_TAG}
  )
  FetchContent_GetProperties(ocl-icd)
  if(NOT ocl-icd_POPULATED)
    message(STATUS "Will fetch OpenCL ICD Loader from ${OCL_LOADER_REPO}")
  endif()
  FetchContent_MakeAvailable(ocl-icd)
else()
  message(STATUS
    "Using OpenCL ICD Loader at ${OpenCL_LIBRARY}")
endif()

add_library(OpenCL-Headers INTERFACE)
target_include_directories(OpenCL-Headers INTERFACE ${OpenCL_INCLUDE_DIR})
target_compile_definitions(OpenCL-Headers INTERFACE -DCL_TARGET_OPENCL_VERSION=300 -DCL_USE_DEPRECATED_OPENCL_1_2_APIS=1)

set(OpenCL_LIBRARY OpenCL CACHE PATH "" FORCE)
