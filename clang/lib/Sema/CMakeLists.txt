set(LLVM_LINK_COMPONENTS
  Core
  Demangle
  FrontendHLSL
  FrontendOpenMP
  MC
  Support
  TargetParser
  )

if (MSVC)
  set_source_files_properties(SemaSYCL.cpp PROPERTIES COMPILE_FLAGS /bigobj)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(is_clang TRUE)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(is_gcc TRUE)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
  set(is_icpx TRUE)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(is_msvc TRUE)
endif()

set(linux_opts "-Wall;-Wextra")
if (CMAKE_BUILD_TYPE MATCHES "Release")
  list(APPEND linux_opts "-Wconversion;-Wimplicit-fallthrough")
endif()

if (is_gcc OR is_clang OR (is_icpx AND NOT WIN32))
  set(opts_to_apply ${linux_opts})
elseif (is_msvc)
  set(opts_to_apply "/W4")
elseif (is_icpx AND WIN32)
  set(opts_to_apply "/Wall")
endif()

if (opts_to_apply)
  set_source_files_properties(
    SemaSYCL.cpp SemaSYCLDeclAttr.cpp
    PROPERTIES
      COMPILE_OPTIONS "${opts_to_apply}"
  )
endif()

clang_tablegen(OpenCLBuiltins.inc -gen-clang-opencl-builtins
  SOURCE OpenCLBuiltins.td
  TARGET ClangOpenCLBuiltinsImpl
  )

clang_tablegen(SPIRVBuiltins.inc -gen-clang-spirv-builtins
 SOURCE SPIRVBuiltins.td
 TARGET ClangSPIRVBuiltinsImpl
 )

add_clang_library(clangSema
  AnalysisBasedWarnings.cpp
  CheckExprLifetime.cpp
  CodeCompleteConsumer.cpp
  DeclSpec.cpp
  DelayedDiagnostic.cpp
  HeuristicResolver.cpp
  HLSLBuiltinTypeDeclBuilder.cpp
  HLSLExternalSemaSource.cpp
  IdentifierResolver.cpp
  JumpDiagnostics.cpp
  MultiplexExternalSemaSource.cpp
  ParsedAttr.cpp
  Scope.cpp
  ScopeInfo.cpp
  Sema.cpp
  SemaAMDGPU.cpp
  SemaARM.cpp
  SemaAVR.cpp
  SemaAccess.cpp
  SemaAttr.cpp
  SemaAPINotes.cpp
  SemaAvailability.cpp
  SemaBPF.cpp
  SemaBase.cpp
  SemaBoundsSafety.cpp
  SemaCXXScopeSpec.cpp
  SemaCast.cpp
  SemaChecking.cpp
  SemaCodeComplete.cpp
  SemaConcept.cpp
  SemaConsumer.cpp
  SemaCoroutine.cpp
  SemaCUDA.cpp
  SemaDirectX.cpp
  SemaDecl.cpp
  SemaDeclAttr.cpp
  SemaDeclCXX.cpp
  SemaDeclObjC.cpp
  SemaExceptionSpec.cpp
  SemaExpr.cpp
  SemaExprCXX.cpp
  SemaExprMember.cpp
  SemaExprObjC.cpp
  SemaFixItUtils.cpp
  SemaFunctionEffects.cpp
  SemaHLSL.cpp
  SemaHexagon.cpp
  SemaInit.cpp
  SemaLambda.cpp
  SemaLookup.cpp
  SemaLoongArch.cpp
  SemaM68k.cpp
  SemaMIPS.cpp
  SemaMSP430.cpp
  SemaModule.cpp
  SemaNVPTX.cpp
  SemaObjC.cpp
  SemaObjCProperty.cpp
  SemaOpenACC.cpp
  SemaOpenACCAtomic.cpp
  SemaOpenACCClause.cpp
  SemaOpenACCClauseAppertainment.cpp
  SemaOpenCL.cpp
  SemaOpenMP.cpp
  SemaOverload.cpp
  SemaPPC.cpp
  SemaPseudoObject.cpp
  SemaRISCV.cpp
  SemaStmt.cpp
  SemaStmtAsm.cpp
  SemaStmtAttr.cpp
  SemaSPIRV.cpp
  SemaSYCL.cpp
  SemaSYCLDeclAttr.cpp
  SemaSwift.cpp
  SemaSystemZ.cpp
  SemaTemplate.cpp
  SemaTemplateDeduction.cpp
  SemaTemplateDeductionGuide.cpp
  SemaTemplateInstantiate.cpp
  SemaTemplateInstantiateDecl.cpp
  SemaTemplateVariadic.cpp
  SemaType.cpp
  SemaTypeTraits.cpp
  SemaWasm.cpp
  SemaX86.cpp
  TypeLocBuilder.cpp

  DEPENDS
  ClangOpenCLBuiltinsImpl
  ClangSPIRVBuiltinsImpl
  omp_gen
  ClangDriverOptions
  acc_gen

  LINK_LIBS
  clangAPINotes
  clangAST
  clangAnalysis
  clangAnalysisLifetimeSafety
  clangBasic
  clangEdit
  clangLex
  clangSupport
  )
