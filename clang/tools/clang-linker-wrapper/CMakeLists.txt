set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  BitWriter
  Core
  BinaryFormat
  MC
  Target
  TransformUtils
  Analysis
  Passes
  IRReader
  Object
  Option
  Support
  TargetParser
  CodeGen
  LTO
  FrontendOffloading
  SYCLLowerIR
  )

set(LLVM_TARGET_DEFINITIONS LinkerWrapperOpts.td)
tablegen(LLVM LinkerWrapperOpts.inc -gen-opt-parser-defs)
add_public_tablegen_target(LinkerWrapperOpts)

if(NOT CLANG_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen LinkerWrapperOpts)
endif()

add_clang_tool(clang-linker-wrapper
  ClangLinkerWrapper.cpp

  DEPENDS
  ${tablegen_deps}
  )

add_dependencies(clang-linker-wrapper sycl-compress)
include_directories(${CMAKE_BINARY_DIR}/include/sycl-compress/)
message(CHECK_START "Finding sycl-compress")
set(SYCL_COMPRESS_LIB ${CMAKE_BINARY_DIR}/lib/libsycl-compress.a)

set(CLANG_LINKER_WRAPPER_LIB_DEPS
  clangBasic
  )

target_link_libraries(clang-linker-wrapper
  PRIVATE
  ${CLANG_LINKER_WRAPPER_LIB_DEPS}
  ${SYCL_COMPRESS_LIB}
  )
