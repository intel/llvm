
if(LLVM_LINK_LLVM_DYLIB)
  set(LLVM_LINK_COMPONENTS
    BinaryFormat
    Core
    Support
  )
else()
  set(LLVM_LINK_COMPONENTS
    BinaryFormat
    Core
    Support
    SPIRVLib
  )
endif()


include_directories(
  ${LLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR}/include
  )

if(LLVM_LINK_LLVM_DYLIB)
  add_llvm_tool(spirv-to-ir-wrapper
    spirv-to-ir-wrapper.cpp
    DEPENDS
    LLVMSPIRVLib
  )
else()
  add_llvm_tool(spirv-to-ir-wrapper
    spirv-to-ir-wrapper.cpp
  )
endif()
  
target_link_libraries(spirv-to-ir-wrapper
	PRIVATE
	LLVMSPIRVLib
	)
