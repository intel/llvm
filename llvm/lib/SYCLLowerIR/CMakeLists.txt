# Lowering of SYCL ESIMD kernels depends on vc-intrinsics
# NOTE: could have been added earlier from llvm/projects
if (NOT TARGET LLVMGenXIntrinsics)
  find_package(LLVMGenXIntrinsics QUIET)

  if (NOT LLVMGenXIntrinsics_FOUND)
    set(LLVMGenXIntrinsics_GIT_REPO https://github.com/intel/vc-intrinsics.git)

    set(LLVMGenXIntrinsics_GIT_TAG 47fa68b3b5ca42fc59ff06fc93e7243a1d6d6fc9)
    if(NOT FETCHCONTENT_SOURCE_DIR_VC-INTRINSICS)
      message(STATUS "vc-intrinsics repo is missing. Will try to download "
        "${LLVMGenXIntrinsics_GIT_TAG} from ${LLVMGenXIntrinsics_GIT_REPO}")
    endif()
    include(FetchContent)
    FetchContent_Declare(vc-intrinsics
      GIT_REPOSITORY ${LLVMGenXIntrinsics_GIT_REPO}
      GIT_TAG        ${LLVMGenXIntrinsics_GIT_TAG}
      )
    FetchContent_MakeAvailable(vc-intrinsics)
    FetchContent_GetProperties(vc-intrinsics)
  else()
    message(STATUS "vc-intrinsics found in system at ${LLVMGenXIntrinsics_DIR}")
  endif()
endif (NOT TARGET LLVMGenXIntrinsics)

add_llvm_component_library(LLVMSYCLLowerIR
  ESIMD/ESIMDOptimizeVecArgCallConv.cpp
  ESIMD/ESIMDUtils.cpp
  ESIMD/ESIMDVerifier.cpp
  ESIMD/ESIMDRemoveHostCode.cpp
  ESIMD/ESIMDRemoveOptnoneNoinline.cpp
  ESIMD/LowerESIMD.cpp
  ESIMD/LowerESIMDKernelAttrs.cpp
  RecordSYCLAspectNames.cpp
  CleanupSYCLMetadata.cpp
  CompileTimePropertiesPass.cpp
  DeviceGlobals.cpp
  ESIMD/LowerESIMDVLoadVStore.cpp
  ESIMD/LowerESIMDSlmReservation.cpp
  LowerInvokeSimd.cpp
  LowerWGLocalMemory.cpp
  LowerWGScope.cpp
  MangleUtils.cpp
  MutatePrintfAddrspace.cpp
  SpecConstants.cpp
  SYCLAddOptLevelAttribute.cpp
  SYCLConditionalCallOnDevice.cpp
  SYCLCreateNVVMAnnotations.cpp
  SYCLDeviceLibBF16.cpp
  SYCLDeviceRequirements.cpp
  SYCLKernelParamOptInfo.cpp
  SYCLJointMatrixTransform.cpp
  SYCLOptimizeBarriers.cpp
  SYCLPropagateAspectsUsage.cpp
  SYCLPropagateJointMatrixUsage.cpp
  SYCLRemangleLibspirv.cpp
  SYCLVirtualFunctionsAnalysis.cpp
  SYCLUtils.cpp

  LocalAccessorToSharedMemory.cpp
  GlobalOffset.cpp
  TargetHelpers.cpp

  SanitizerPostOptimizer.cpp
  SanitizerUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SYCLLowerIR

  DEPENDS
  intrinsics_gen
  LLVMGenXIntrinsics
  LLVMDemangle
  LLVMTransformUtils
  DeviceConfigFile

  LINK_LIBS
  LLVMGenXIntrinsics
  LLVMDemangle
  LLVMTargetParser
  LLVMTransformUtils

  LINK_COMPONENTS
  Analysis
  BitWriter
  Core
  Demangle
  IRPrinter
  Support
  ipo
  )

target_link_libraries(LLVMSYCLLowerIR
  PUBLIC LLVMGenXIntrinsics
  )

# vc-intrinsics' add_compile_definitions(VC_INTR_LLVM_VERSION_MAJOR=...) in
# its top-level CMakeLists.txt is directory-scoped and does not propagate to
# LLVMSYCLLowerIR.
# FIXME: Remove this once https://github.com/intel/vc-intrinsics is fixed.
target_compile_definitions(LLVMSYCLLowerIR PUBLIC
  VC_INTR_LLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR})
