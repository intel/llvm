set(LLVM_LINK_COMPONENTS
  Core
  Support
  SPIRVLib
  )

include_directories(
  ${LLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR}/include
  )

add_llvm_tool(spir-to-ir
  spir-to-ir.cpp
  )

#  DEPENDS
#  intrinsics_gen
