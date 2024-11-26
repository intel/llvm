# Setup libclc-remangler
set(LLVM_LINK_COMPONENTS
  BitWriter
  BitReader
  Core
  Demangle
  Support
  TransformUtils
  IRReader
  )

add_clang_tool(libclc-remangler LibclcRemangler.cpp)

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

# If we've not already imported a libclc-remangler tool from an external build,
# set it up now.
if(NOT TARGET libclc-remangler_target)
  if ( CMAKE_CROSSCOMPILING )
    setup_host_tool(libclc-remangler LIBCLC_REMANGLER libclc-remangler_exe libclc-remangler_target)
  else()
    add_executable(libclc-remangler_target ALIAS libclc-remangler)
  endif()
endif()
