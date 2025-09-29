# Lowering of SYCL ESIMD kernels depends on vc-intrinsics
# NOTE: could have been added earlier from llvm/projects
if (NOT TARGET LLVMGenXIntrinsics)
  find_package(LLVMGenXIntrinsics QUIET)

  if (NOT LLVMGenXIntrinsics_FOUND)
    set(LLVMGenXIntrinsics_GIT_REPO https://github.com/intel/vc-intrinsics.git)

    # Date: May 29, 2025
    # Use OneNthEltsVecArgument instead of HalfVecArguments to fix build failure.
    set(LLVMGenXIntrinsics_GIT_TAG 60cea7590bd022d95f5cf336ee765033bd114d69)
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
  HostPipes.cpp
  LowerInvokeSimd.cpp
  LowerWGLocalMemory.cpp
  LowerWGScope.cpp
  MutatePrintfAddrspace.cpp
  SpecConstants.cpp
  SYCLAddOptLevelAttribute.cpp
  SYCLConditionalCallOnDevice.cpp
  SYCLCreateNVVMAnnotations.cpp
  SYCLDeviceLibReqMask.cpp
  SYCLDeviceRequirements.cpp
  SYCLKernelParamOptInfo.cpp
  SYCLJointMatrixTransform.cpp
  SYCLOptimizeBarriers.cpp
  SYCLPropagateAspectsUsage.cpp
  SYCLPropagateJointMatrixUsage.cpp
  SYCLVirtualFunctionsAnalysis.cpp
  SYCLUtils.cpp

  LocalAccessorToSharedMemory.cpp
  GlobalOffset.cpp
  TargetHelpers.cpp

  SanitizerPostOptimizer.cpp

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
