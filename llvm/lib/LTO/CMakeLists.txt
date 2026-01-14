if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # We append -fno-lifetime-dse in HandleLLVMOptions.cmake
  # append("-fno-lifetime-dse" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  # But it is causing link failure with llvm::StdThreadPool::asyncEnqueue
  string(REPLACE "-fno-lifetime-dse" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif ()

add_llvm_component_library(LLVMLTO
  LTO.cpp
  LTOBackend.cpp
  LTOModule.cpp
  LTOCodeGenerator.cpp
  UpdateCompilerUsed.cpp
  ThinLTOCodeGenerator.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/LTO
  DEPENDS
  intrinsics_gen
  llvm_vcsrevision_h

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}

  LINK_COMPONENTS
  AggressiveInstCombine
  Analysis
  BinaryFormat
  BitReader
  BitWriter
  CGData
  CodeGen
  CodeGenTypes
  Core
  Extensions
  IPO
  InstCombine
  Instrumentation
  Linker
  MC
  ObjCARC
  Object
  Passes
  Remarks
  Scalar
  Support
  SYCLLowerIR
  Target
  TargetParser
  TransformUtils
  )
