add_library(GenISAIntrinsics STATIC IMPORTED)
set_target_properties(GenISAIntrinsics
  PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libGenISAIntrinsics.a
)

add_mlir_conversion_library(MLIRGENToLLVM
  GENToLLVMPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/GENToLLVM

  DEPENDS
  GenISAIntrinsics  
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  GenISAIntrinsics
  MLIRGENDialect  
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSupport
)
