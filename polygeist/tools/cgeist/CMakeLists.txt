set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  CodeGen
  Core
  AggressiveInstCombine
  InstCombine
  Instrumentation
  MC
  IPO
  MCParser
  ObjCARCOpts
  Option
  ScalarOpts
  Support
  TransformUtils
  Vectorize
)

add_clang_executable(cgeist
  driver.cc
  "${LLVM_SOURCE_DIR}/../clang/tools/driver/cc1_main.cpp"
  "${LLVM_SOURCE_DIR}/../clang/tools/driver/cc1as_main.cpp"
  "${LLVM_SOURCE_DIR}/../clang/tools/driver/cc1gen_reproducer_main.cpp"

  Lib/AffineUtils.cc
  Lib/Attributes.cc
  Lib/ConstantFolder.cc
  Lib/CGCall.cc
  Lib/CGDecl.cc
  Lib/CGExpr.cc
  Lib/CGStmt.cc
  Lib/CodeGenTypes.cc  
  Lib/IfScope.cc
  Lib/pragmaHandler.cc
  Lib/TypeUtils.cc
  Lib/ValueCategory.cc
  Lib/utils.cc
)

option(CGEIST_DEFAULT_ALLOW_UNDEFINED_SYCL_TYPES
  "Whether to allow undefined SYCL types by default" OFF)
configure_file(CgeistConfig.h.in CgeistConfig.h)

install(TARGETS cgeist
EXPORT PolygeistTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT cgeist)

target_include_directories(cgeist PRIVATE
  "${PROJECT_BINARY_DIR}"
  "${LLVM_SOURCE_DIR}/../clang/include"
  "${CMAKE_BINARY_DIR}/tools/clang/include"
)

target_compile_definitions(cgeist PUBLIC -DLLVM_OBJ_ROOT="${LLVM_BINARY_DIR}")
target_link_libraries(cgeist PRIVATE
  MLIRSCFTransforms
  MLIRPolygeistDialect

  MLIRSupport
  MLIRIR
  MLIRAnalysis
  MLIRBuiltinToLLVMIRTranslation
  MLIRFuncInlinerExtension
  MLIRSPIRVDialect
  MLIRSYCLDialect
  MLIROpenMPDialect
  MLIRGPUDialect
  MLIRTransforms
  MLIRSCFToControlFlow
  MLIRFuncToLLVM
  MLIRAffineTransforms
  MLIRArithTransforms
  MLIRAffineToStandard
  MLIRMathToLLVM
  MLIRTargetLLVMIRImport
  MLIRPolygeistToLLVM
  MLIRPolygeistTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRReconcileUnrealizedCasts
  MLIRSCFToOpenMP
  MLIROpenMPToLLVM
  MLIROpenMPToLLVMIRTranslation
  MLIRSYCLToLLVM
  MLIRSYCLTransforms
  MLIRVectorDialect
  
  LLVMSYCLLowerIR
  LLVMPasses

  clangAST
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangLex
  clangSerialization
)
add_dependencies(cgeist MLIRPolygeistOpsIncGen MLIRPolygeistPassIncGen)
add_subdirectory(Test)
