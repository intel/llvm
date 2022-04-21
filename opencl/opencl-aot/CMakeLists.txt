cmake_minimum_required(VERSION 3.14)

set(LLVM_LINK_COMPONENTS
        ${LLVM_TARGETS_TO_BUILD}
        Support
        )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB TARGET_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)

set(OPENCL_AOT_PROJECT_NAME opencl-aot)

add_llvm_tool(${OPENCL_AOT_PROJECT_NAME} ${TARGET_SOURCES})

if(NOT MSVC)
  # FIXME: when built with clang it produces a warning.
  target_compile_options(${OPENCL_AOT_PROJECT_NAME} PRIVATE "-Wno-unused-parameter")
endif()

target_link_libraries(${OPENCL_AOT_PROJECT_NAME}
  PRIVATE
    OpenCL-Headers
    OpenCL-ICD)
