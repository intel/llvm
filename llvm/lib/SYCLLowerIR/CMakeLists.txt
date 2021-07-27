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
    message(STATUS "vc-intrinsics are missing. Will try to download them from github.com")

    set( LLVMGenXIntrinsics_GIT_TAG 19e6ddafc545bc96ceaacbf7c81cacbff59a39b6 )

    include(FetchContent)
    FetchContent_Declare(vc-intrinsics
      GIT_REPOSITORY https://github.com/intel/vc-intrinsics.git
      GIT_TAG        ${LLVMGenXIntrinsics_GIT_TAG}
    )
    FetchContent_MakeAvailable(vc-intrinsics)
    FetchContent_GetProperties(vc-intrinsics)

    set(LLVMGenXIntrinsics_SOURCE_DIR ${vc-intrinsics_SOURCE_DIR})
    set(LLVMGenXIntrinsics_BINARY_DIR ${vc-intrinsics_BINARY_DIR})
  else()
    # -DLLVMGenXIntrinsics_SOURCE_DIR is provided
    message(STATUS "vc-intrinsics are added manually ${LLVMGenXIntrinsics_SOURCE_DIR}")

    set(LLVMGenXIntrinsics_BINARY_DIR ${CMAKE_BINARY_DIR}/vc-intrinsics-build)
    add_subdirectory(${LLVMGenXIntrinsics_SOURCE_DIR} ${LLVMGenXIntrinsics_BINARY_DIR})
  endif()

  target_include_directories(LLVMGenXIntrinsics
    PUBLIC $<BUILD_INTERFACE:${LLVMGenXIntrinsics_SOURCE_DIR}/GenXIntrinsics/include>
    PUBLIC $<BUILD_INTERFACE:${LLVMGenXIntrinsics_BINARY_DIR}/GenXIntrinsics/include>
  )
endif()

set_property(GLOBAL PROPERTY LLVMGenXIntrinsics_SOURCE_PROP ${LLVMGenXIntrinsics_SOURCE_DIR})
set_property(GLOBAL PROPERTY LLVMGenXIntrinsics_BINARY_PROP ${LLVMGenXIntrinsics_BINARY_DIR})

add_llvm_component_library(LLVMSYCLLowerIR
  LowerWGScope.cpp
  LowerESIMD.cpp
  LowerESIMDVLoadVStore.cpp
  LowerESIMDVecArg.cpp
  LowerWGLocalMemory.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SYCLLowerIR
  ${LLVM_MAIN_SRC_DIR}/projects/vc-intrinsics/GenXIntrinsics/include
  ${LLVM_BINARY_DIR}/projects/vc-intrinsics/GenXIntrinsics/include

  DEPENDS
  intrinsics_gen
  LLVMGenXIntrinsics
  LLVMDemangle
  LLVMTransformUtils

  LINK_LIBS
  LLVMGenXIntrinsics
  LLVMDemangle
  LLVMTransformUtils
  
  LINK_COMPONENTS
  Core
  Support
  )

target_include_directories(LLVMSYCLLowerIR
  PRIVATE ${LLVM_MAIN_SRC_DIR}/projects/vc-intrinsics/GenXIntrinsics/include
  PRIVATE ${LLVM_BINARY_DIR}/projects/vc-intrinsics/GenXIntrinsics/include)
