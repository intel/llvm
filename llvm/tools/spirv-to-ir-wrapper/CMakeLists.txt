set(LLVM_LINK_COMPONENTS
  BinaryFormat
  Core
  Support
  SPIRVLib
  )

include_directories(
  ${LLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR}/include
  )

add_llvm_tool(spirv-to-ir-wrapper
  spirv-to-ir-wrapper.cpp
  )

# LLVMSPIRVLib is not part of libLLVM.so, so in LLVM_LINK_LLVM_DYLIB builds the
# SPIRVLib component listed above resolves to the dylib (which lacks its
# symbols). Link the static library explicitly, matching llvm-spirv.
if (LLVM_LINK_LLVM_DYLIB)
  target_link_libraries(spirv-to-ir-wrapper PRIVATE LLVMSPIRVLib)
endif()
