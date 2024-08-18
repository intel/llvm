set(LLVM_LINK_COMPONENTS BitWriter BitReader Core Object Support TransformUtils TargetParser)

add_clang_tool(clang-offload-wrapper
  ClangOffloadWrapper.cpp
  SymPropReader.cpp

  DEPENDS
  intrinsics_gen
  )

set(CLANG_OFFLOAD_WRAPPER_LIB_DEPS
  clangBasic
  )

add_dependencies(clang-offload-wrapper sycl-compress)
include_directories(${CMAKE_BINARY_DIR}/include/sycl-compress/)

add_dependencies(clang clang-offload-wrapper)

#find_library(SYCL_COMPRESS_LIB sycl-compress PATHS ${CMAKE_BINARY_DIR}/lib NO_DEFAULT_PATH)
clang_target_link_libraries(clang-offload-wrapper
  PRIVATE
  ${CLANG_OFFLOAD_WRAPPER_LIB_DEPS}
  sycl-compress
  )
