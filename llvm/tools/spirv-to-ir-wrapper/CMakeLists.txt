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
