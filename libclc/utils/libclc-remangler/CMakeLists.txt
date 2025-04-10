# Setup libclc-remangler
set(LLVM_LINK_COMPONENTS
  BitWriter
  BitReader
  Core
  Demangle
  Support
  TransformUtils
  IRReader
  TargetParser
  )

add_clang_tool(libclc-remangler LibclcRemangler.cpp)

setup_host_tool( libclc-remangler LIBCLC_REMANGLER
  libclc-remangler_exe libclc-remangler_target )

target_include_directories(libclc-remangler PRIVATE
  ${CMAKE_SOURCE_DIR}/../clang/include
  ${CMAKE_BINARY_DIR}/tools/clang/include)

clang_target_link_libraries(libclc-remangler
  PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangTooling
  clangSerialization
  LLVMOption
  )
