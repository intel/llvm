cmake_minimum_required(VERSION 3.14)

include(FetchContent)

# OpenCL Headers
message(STATUS "Will fetch OpenCL headers from github.com")

FetchContent_Declare(ocl-headers
  GIT_REPOSITORY    https://github.com/KhronosGroup/OpenCL-Headers.git
  GIT_TAG           23710f1b99186065c1768fc3098ba681adc0f253
)

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

message(STATUS "Will fetch OpenCL ICD Loader from github.com")

FetchContent_Declare(ocl-icd
  GIT_REPOSITORY    https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
  GIT_TAG           5f8249691ec8c25775789498951f8e9eb62c201d
)

FetchContent_MakeAvailable(ocl-icd)
FetchContent_GetProperties(ocl-icd)
set(OpenCL_LIBRARY ${ocl-icd_SOURCE_DIR})

add_library(OpenCL-ICD ALIAS OpenCL)

