add_mlir_unittest(MLIRMemoryAccessAnalysisTests
  MatrixTest.cpp
)

target_link_libraries(MLIRMemoryAccessAnalysisTests
  PRIVATE MLIRPolygeistAnalysis
  MLIRArithDialect
  MLIRParser
  )
