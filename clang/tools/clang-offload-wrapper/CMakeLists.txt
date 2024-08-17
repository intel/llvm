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
message(CHECK_START "Finding sycl-compress")
set(SYCL_COMPRESS_LIB ${CMAKE_BINARY_DIR}/lib/libsycl-compress.a)
set(ZSTD_LIBRARY ${CMAKE_BINARY_DIR}/tools/sycl-compress/zstd/install/lib/libzstd.a)

add_dependencies(clang clang-offload-wrapper)

clang_target_link_libraries(clang-offload-wrapper
  PRIVATE
  ${CLANG_OFFLOAD_WRAPPER_LIB_DEPS}
  ${SYCL_COMPRESS_LIB}
  ${ZSTD_LIBRARY}
  )
