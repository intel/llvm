add_llvm_component_library(LLVMFrontendOffloading
  Utility.cpp
  OffloadWrapper.cpp
<<<<<<< HEAD
  SYCLOffloadWrapper.cpp

=======
  PropertySet.cpp
>>>>>>> 185a23e865c31145588276e7e0db2460e01eb703

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  BinaryFormat
  Object
  ObjectYAML
  Support
  TransformUtils
  TargetParser
)
