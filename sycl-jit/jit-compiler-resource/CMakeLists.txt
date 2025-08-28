# We use C23/C++26's `#embed` to implement this resource creation, and "host"
# CXX compiler might not have support for it. As such, use freshly built
# `clang++` instead. That, in turn, requires dedicated directory for CMake to
# allow to override `CMAKE_CXX_COMPILER`.


set(SYCL_JIT_RESOURCE_CPP "${CMAKE_CURRENT_BINARY_DIR}/resource.cpp")
add_custom_command(
  OUTPUT ${SYCL_JIT_RESOURCE_CPP}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py --toolchain-dir ${CMAKE_BINARY_DIR} --output ${SYCL_JIT_RESOURCE_CPP} --prefix /sycl-jit-toolchain/
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS sycl-headers ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
)

set_source_files_properties(${SYCL_JIT_RESOURCE_CPP} PROPERTIES
  COMPILE_FLAGS -Wno-c23-extensions
)

add_llvm_library(sycl-jit-resource
  ${SYCL_JIT_RESOURCE_CPP}
  OBJECT
  DEPENDS
  clang
  # TODO: libdevice
)

set(CMAKE_CXX_COMPILER ${CMAKE_BINARY_DIR}/bin/clang++)
