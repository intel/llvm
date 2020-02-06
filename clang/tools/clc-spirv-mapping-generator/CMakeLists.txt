set(LLVM_LINK_COMPONENTS support)

add_clang_executable(clc-spirv-mapping-generator ClcSpirvMappingGenerator.cpp)

set(LLVM_SPIRV_INCLUDE_DIRS "${LLVM_MAIN_SRC_DIR}/../llvm-spirv/include")
target_include_directories(clc-spirv-mapping-generator PRIVATE BEFORE ${LLVM_SPIRV_INCLUDE_DIRS})
target_link_libraries(clc-spirv-mapping-generator
  PRIVATE LLVMSPIRVLib clangCodeGen clangTooling clangBasic clangASTMatchers)
