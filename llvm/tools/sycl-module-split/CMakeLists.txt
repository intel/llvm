set(LLVM_LINK_COMPONENTS
  BitWriter
  Core
  IRPrinter
  IRReader
  Support
  TargetParser
  TransformUtils
  SYCLLowerIR
  Passes
  Analysis
  )

add_llvm_tool(sycl-module-split
  sycl-module-split.cpp
  )
