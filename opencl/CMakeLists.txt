cmake_minimum_required(VERSION 3.14)

include(FetchContent)

# Suppress warnings related to dependencies
if (MSVC)
  add_definitions(
    -wd4133 # Suppress ''function': incompatible types - from 'char [9]' to 'LPCWSTR''
    -wd4152 # Suppress 'nonstandard extension, function/data pointer conversion in expression'
    -wd4201 # Suppress 'nonstandard extension used: nameless struct/union'
  )
endif()

# Repo URLs

set(OCL_HEADERS_REPO
  "https://github.com/KhronosGroup/OpenCL-Headers.git")
set(OCL_LOADER_REPO
  "https://github.com/KhronosGroup/OpenCL-ICD-Loader.git")

# Repo tags/hashes

set(OCL_HEADERS_TAG dcd5bede6859d26833cd85f0d6bbcee7382dc9b3)
set(OCL_LOADER_TAG 5f8249691ec8c25775789498951f8e9eb62c201d)

# OpenCL Headers
if(NOT OpenCL_HEADERS)
  message(STATUS "Will fetch OpenCL headers from ${OCL_HEADERS_REPO}")

  FetchContent_Declare(ocl-headers
    GIT_REPOSITORY    ${OCL_HEADERS_REPO}
    GIT_TAG           ${OCL_HEADERS_TAG}
  )
else()
  message(STATUS "OpenCL headers are added manually ${OpenCL_HEADERS}")

  FetchContent_Declare(ocl-headers
    URL               ${OpenCL_HEADERS}
  )
endif()

FetchContent_MakeAvailable(ocl-headers)
FetchContent_GetProperties(ocl-headers)
set(OpenCL_INCLUDE_DIR
  ${ocl-headers_SOURCE_DIR} CACHE PATH "Path to OpenCL Headers")

target_compile_definitions(Headers INTERFACE -DCL_TARGET_OPENCL_VERSION=220)
add_library(OpenCL-Headers ALIAS Headers)

# OpenCL Library (ICD Loader)

# Set OPENCL_ICD_LOADER_HEADERS_DIR, as prerequisite for ICD build
set(OPENCL_ICD_LOADER_HEADERS_DIR
  ${OpenCL_INCLUDE_DIR} CACHE PATH "Path to OpenCL Headers")

# LLVM build sets this OFF by default, but we need OpenCL to be built as shared
# library.
set(BUILD_SHARED_LIBS ON)

if(NOT OpenCL_LIBRARY_SRC)
  message(STATUS "Will fetch OpenCL ICD Loader from ${OCL_LOADER_REPO}")

  FetchContent_Declare(ocl-icd
    GIT_REPOSITORY    ${OCL_LOADER_REPO}
    GIT_TAG           ${OCL_LOADER_TAG}
  )
else()
  # TODO: add possibility to use prebuilt OpenCL library rather than building
  #       together with llvm.
  message(STATUS
    "OpenCL ICD Loader sources added manually ${OpenCL_LIBRARY_SRC}")

  FetchContent_Declare(ocl-icd
    URL               ${OpenCL_LIBRARY_SRC}
  )
endif()

FetchContent_MakeAvailable(ocl-icd)
add_library(OpenCL-ICD ALIAS OpenCL)

add_subdirectory(opencl-aot)
