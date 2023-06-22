if(MLIR_INCLUDE_TESTS)
  add_mlir_library(MLIRPolygeistTestAnalysis
    TestMemoryAccessAnalysis.cpp
    TestReachingDefinitionsAnalysis.cpp
    TestUniformityAnalysis.cpp

    EXCLUDE_FROM_LIBMLIR

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Polygeist
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRAnalysis
    MLIRIR
    MLIRPass
    MLIRPolygeistAnalysis
    MLIRSupport
    MLIRSYCLAnalysis
  )
endif()
