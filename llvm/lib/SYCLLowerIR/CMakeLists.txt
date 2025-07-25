# Lowering of SYCL ESIMD kernels depends on vc-intrinsics
# NOTE: could have been added earlier from llvm/projects
if (NOT TARGET LLVMGenXIntrinsics)
 if(NOT DEFINED LLVMGenXIntrinsics_SOURCE_DIR AND NOT LLVMGenXIntrinsics_INCLUDE_DIR)
   find_package(LLVMGenXIntrinsics QUIET)
 endif()

 if (NOT LLVMGenXIntrinsics_FOUND)
    if (NOT DEFINED LLVMGenXIntrinsics_SOURCE_DIR)
      find_path(LLVMGenXIntrinsics_INCLUDE_DIR GenXIntrinsics/include PATHS ${CMAKE_PREFIX_PATH})

      if (NOT LLVMGenXIntrinsics_INCLUDE_DIR)
        set(LLVMGenXIntrinsics_GIT_REPO https://github.com/intel/vc-intrinsics.git)

        # Date: May 29, 2025
        # Use OneNthEltsVecArgument instead of HalfVecArguments to fix build failure.
        set(LLVMGenXIntrinsics_GIT_TAG 60cea7590bd022d95f5cf336ee765033bd114d69)

        message(STATUS "vc-intrinsics repo is missing. Will try to download it from ${LLVMGenXIntrinsics_GIT_REPO}")
        include(FetchContent)
        FetchContent_Declare(vc-intrinsics
          GIT_REPOSITORY ${LLVMGenXIntrinsics_GIT_REPO}
          GIT_TAG        ${LLVMGenXIntrinsics_GIT_TAG}
        )
        FetchContent_MakeAvailable(vc-intrinsics)
        FetchContent_GetProperties(vc-intrinsics)

        set(LLVMGenXIntrinsics_SOURCE_DIR ${vc-intrinsics_SOURCE_DIR})
        set(LLVMGenXIntrinsics_BINARY_DIR ${vc-intrinsics_BINARY_DIR})
      else()
        message(STATUS "vc-intrinsics found in system at ${LLVMGenXIntrinsics_INCLUDE_DIR}")
        set(LLVMGenXIntrinsics_SOURCE_DIR ${LLVMGenXIntrinsics_INCLUDE_DIR}/..)
        set(LLVMGenXIntrinsics_BINARY_DIR ${CMAKE_BINARY_DIR}/vc-intrinsics-build)
        add_subdirectory(${LLVMGenXIntrinsics_SOURCE_DIR} ${LLVMGenXIntrinsics_BINARY_DIR})
      endif()
    else (NOT DEFINED LLVMGenXIntrinsics_SOURCE_DIR)
      # -DLLVMGenXIntrinsics_SOURCE_DIR is provided
      message(STATUS "vc-intrinsics are added manually ${LLVMGenXIntrinsics_SOURCE_DIR}")

      set(LLVMGenXIntrinsics_BINARY_DIR ${CMAKE_BINARY_DIR}/vc-intrinsics-build)
      add_subdirectory(${LLVMGenXIntrinsics_SOURCE_DIR} ${LLVMGenXIntrinsics_BINARY_DIR})
    endif (NOT DEFINED LLVMGenXIntrinsics_SOURCE_DIR)
    target_include_directories(LLVMGenXIntrinsics
      PUBLIC $<BUILD_INTERFACE:${LLVMGenXIntrinsics_SOURCE_DIR}/GenXIntrinsics/include>
      PUBLIC $<BUILD_INTERFACE:${LLVMGenXIntrinsics_BINARY_DIR}/GenXIntrinsics/include>
    )
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

  SanitizerKernelMetadata.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SYCLLowerIR
  ${LLVM_MAIN_SRC_DIR}/projects/vc-intrinsics/GenXIntrinsics/include
  ${LLVM_BINARY_DIR}/projects/vc-intrinsics/GenXIntrinsics/include

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

target_include_directories(LLVMSYCLLowerIR
  PUBLIC $<BUILD_INTERFACE:${LLVMGenXIntrinsics_SOURCE_DIR}/GenXIntrinsics/include/>
  PUBLIC $<BUILD_INTERFACE:${LLVMGenXIntrinsics_BINARY_DIR}>
  )
target_link_libraries(LLVMSYCLLowerIR
  PUBLIC LLVMGenXIntrinsics
  )
