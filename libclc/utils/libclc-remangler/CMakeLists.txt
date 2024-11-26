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

if(CMAKE_CROSSCOMPILING)
  if(LLVM_NATIVE_TOOL_DIR AND NOT LIBCLC_REMANGLER_PATH)
    if(EXISTS "${LLVM_NATIVE_TOOL_DIR}/libclc-remangler${LLVM_HOST_EXECUTABLE_SUFFIX}")
      set(LLVM_CONFIG_PATH "${LLVM_NATIVE_TOOL_DIR}/libclc-remangler${LLVM_HOST_EXECUTABLE_SUFFIX}")
    endif()
  endif()

  if(NOT LIBCLC_REMANGLER_PATH)
    build_native_tool(libclc-remangler LIBCLC_REMANGLER_PATH)
    set(LIBCLC_REMANGLER_PATH "${LIBCLC_REMANGLER_PATH}" CACHE STRING "")

    add_custom_target(NativeLibCLCRemangler DEPENDS ${LIBCLC_REMANGLER_PATH})
    add_dependencies(libclc-remangler NativeLibCLCRemangler)
  endif()
else()
  set(LIBCLC_REMANGLER_PATH "${CMAKE_CURRENT_BINARY_DIR}/bin/libclc-remangler" CACHE STRING "")
endif()

# If we've not already imported a libclc-remangler tool from an external build,
# set it up now.
if(NOT TARGET libclc::libclc-remangler)
  add_executable(libclc::libclc-remangler ALIAS libclc-remangler)
endif()
