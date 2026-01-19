# Finds or fetches OpenCL Headers and the ICD loader.
if(TARGET OpenCL-Headers)
  # If we already ran this module (so the OpenCL-Headers target exists),
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

# Set NO_CMAKE_PACKAGE_REGISTRY so only system-wide installs are
# detected.
find_package(OpenCL 3.0 QUIET NO_CMAKE_PACKAGE_REGISTRY)

if(OpenCL_FOUND)
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
  set(CMAKE_REQUIRED_INCLUDES ${OpenCL_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${OpenCL_LIBRARY})
  check_cxx_source_compiles("${OPENCL_TEST_PROGRAM}" OPENCL_HEADERS_VERSION_SUPPORTED)
  if(NOT OPENCL_HEADERS_VERSION_SUPPORTED)
    message(WARNING "Preinstalled OpenCL-Headers are not supported, "
      "use commit ${OCL_HEADERS_TAG} or later. Will fetch OpenCL.")
    set(OpenCL_FOUND FALSE CACHE BOOL "" FORCE)
  endif()
endif()

# Ideally we could use the FIND_PACKAGE_ARGS argument to FetchContent_Declare to avoid
# the conditonals, but that was added in CMake 3.24 and the current minimum we require is
# 3.20.

# OpenCL Headers
if(NOT OpenCL_FOUND)
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

if(NOT OpenCL_FOUND)
  FetchContent_Declare(ocl-icd
      GIT_REPOSITORY    ${OCL_LOADER_REPO}
      GIT_TAG           ${OCL_LOADER_TAG}
  )
  FetchContent_GetProperties(ocl-icd)
  if(NOT ocl-icd_POPULATED)
    message(STATUS "Will fetch OpenCL ICD Loader from ${OCL_LOADER_REPO}")
  endif()
  FetchContent_MakeAvailable(ocl-icd)
  set(OpenCL_LIBRARY OpenCL::OpenCL CACHE PATH "" FORCE)
else()
  message(STATUS
    "Using OpenCL ICD Loader at ${OpenCL_LIBRARY}")
endif()

add_library(OpenCL-Headers INTERFACE)
target_include_directories(OpenCL-Headers INTERFACE ${OpenCL_INCLUDE_DIR})
target_compile_definitions(OpenCL-Headers INTERFACE -DCL_TARGET_OPENCL_VERSION=300 -DCL_USE_DEPRECATED_OPENCL_1_2_APIS=1)

set(OpenCL_FOUND ${OpenCL_FOUND} CACHE BOOL INTERNAL)
