if(${CMAKE_VERSION} VERSION_LESS 3.14)
    macro(FetchContent_MakeAvailable NAME)
        FetchContent_GetProperties(${NAME})
        if(NOT ${NAME}_POPULATED)
            FetchContent_Populate(${NAME})
            add_subdirectory(${${NAME}_SOURCE_DIR} ${${NAME}_BINARY_DIR})
        endif()
    endmacro()
endif()

# Lowering of SYCL ESIMD kernels depends on vc-intrinsics
# NOTE: could have been added earlier from llvm/projects
if (NOT TARGET LLVMGenXIntrinsics)
  if (NOT DEFINED LLVMGenXIntrinsics_SOURCE_DIR)
    set(LLVMGenXIntrinsics_GIT_REPO https://github.com/intel/vc-intrinsics.git)

    # Date: 13 Feb 2024
    # Add an intrinsic for named barrier arrive/signal operation
    set(LLVMGenXIntrinsics_GIT_TAG f9c34404d0ea9abad83875a10bd48d88cea90ebd)

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
endif (NOT TARGET LLVMGenXIntrinsics)

set_property(GLOBAL PROPERTY LLVMGenXIntrinsics_SOURCE_PROP ${LLVMGenXIntrinsics_SOURCE_DIR})
set_property(GLOBAL PROPERTY LLVMGenXIntrinsics_BINARY_PROP ${LLVMGenXIntrinsics_BINARY_DIR})

add_llvm_component_library(LLVMSYCLLowerIR
  ESIMD/ESIMDOptimizeVecArgCallConv.cpp
  ESIMD/ESIMDUtils.cpp
  ESIMD/ESIMDVerifier.cpp
  ESIMD/ESIMDRemoveHostCode.cpp
  ESIMD/ESIMDRemoveOptnoneNoinline.cpp
  ESIMD/LowerESIMD.cpp
  ESIMD/LowerESIMDKernelAttrs.cpp
  CleanupSYCLMetadata.cpp
  CompileTimePropertiesPass.cpp
  DeviceGlobals.cpp
  ESIMD/LowerESIMDVLoadVStore.cpp
  ESIMD/LowerESIMDSlmReservation.cpp
  HostPipes.cpp
  LowerInvokeSimd.cpp
  LowerWGLocalMemory.cpp
  LowerWGScope.cpp
  ModuleSplitter.cpp
  MutatePrintfAddrspace.cpp
  SYCLAddOptLevelAttribute.cpp
  SYCLPropagateAspectsUsage.cpp
  SYCLPropagateJointMatrixUsage.cpp
  SYCLUtils.cpp
  SanitizeDeviceGlobal.cpp

  LocalAccessorToSharedMemory.cpp
  GlobalOffset.cpp
  TargetHelpers.cpp
  PrepareSYCLNativeCPU.cpp
  RenameKernelSYCLNativeCPU.cpp
  ConvertToMuxBuiltinsSYCLNativeCPU.cpp
  PipelineSYCLNativeCPU.cpp

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
  Core
  Support
  ipo
  )

target_include_directories(LLVMSYCLLowerIR
  PRIVATE ${LLVM_MAIN_SRC_DIR}/projects/vc-intrinsics/GenXIntrinsics/include
  PRIVATE ${LLVM_BINARY_DIR}/projects/vc-intrinsics/GenXIntrinsics/include)
