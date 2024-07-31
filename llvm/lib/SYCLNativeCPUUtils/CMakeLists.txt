add_llvm_component_library(LLVMSYCLNativeCPUUtils
  PipelineSYCLNativeCPU.cpp
  PrepareSYCLNativeCPU.cpp
  RenameKernelSYCLNativeCPU.cpp
  ConvertToMuxBuiltinsSYCLNativeCPU.cpp


  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SYCLLowerIR
  
  LINK_COMPONENTS
  Analysis
  Core
  Support
  Passes
  Target
  TargetParser
  TransformUtils
  ipo
  )

set(OCK_SOURCE_DIR "" CACHE PATH "Root of the local checkout of the oneAPI Construction Kit")
set(OCK_GIT_REPO "" CACHE STRING "Git repository for the oneAPI Construction Kit FetchContent")
set(OCK_GIT_TAG "" CACHE STRING "Git tag for the oneAPI Construction Kit FetchContent")
option(NATIVECPU_OCK_USE_FETCHCONTENT "Use FetchContent to acquire oneAPI Construction Kit source code" On)
option(NATIVECPU_USE_OCK "Use the oneAPI Construction Kit for Native CPU" ON)

# Don't fetch OCK if Native CPU is not enabled.
if(NOT "native_cpu" IN_LIST SYCL_ENABLE_PLUGINS)
  set(NATIVECPU_USE_OCK Off CACHE BOOL "Use the oneAPI Construction Kit for Native CPU" FORCE)
endif()

if(NATIVECPU_USE_OCK)
  if(NATIVECPU_OCK_USE_FETCHCONTENT)
    set(OCK_GIT_INTERNAL_REPO "https://github.com/codeplaysoftware/oneapi-construction-kit.git")
    # commit 0e913951b61b709ce5c5c136f7557dd88d6e200c
    # Merge: 0cf91055 85624e50
    # Author: Colin Davidson <colin.davidson@codeplay.com>
    # Date:   Mon Jul 1 15:39:36 2024 +0100
    #     Merge pull request #485 from coldav/colin/fix_memmove_bug_finegrained
    #     Fix risc-v memmove creating memcpy instrinsic
    set(OCK_GIT_INTERNAL_TAG 0e913951b61b709ce5c5c136f7557dd88d6e200c)

    # Overwrite OCK_GIT_INTERNAL_REPO/OCK_GIT_INTERNAL_TAG if the corresponding options are set
    if(OCK_GIT_REPO)
      set(OCK_GIT_INTERNAL_REPO "${OCK_GIT_REPO}")
    endif()
    if(OCK_GIT_TAG)
      set(OCK_GIT_INTERNAL_TAG "${OCK_GIT_TAG}")
    endif()
    include(FetchContent)
    FetchContent_Declare(oneapi-ck
     GIT_REPOSITORY "${OCK_GIT_INTERNAL_REPO}"
     GIT_TAG "${OCK_GIT_INTERNAL_TAG}"
    )
    FetchContent_GetProperties(oneapi-ck)
    if(NOT oneapi-ck_POPULATED)
      message(STATUS "Cloning oneAPI Construction Kit from ${OCK_GIT_INTERNAL_REPO}, tag ${OCK_GIT_INTERNAL_TAG}")
      FetchContent_Populate(oneapi-ck)
      message(STATUS "oneAPI Construction Kit cloned in ${oneapi-ck_SOURCE_DIR}")
      set(OCK_SOURCE_DIR_INTERNAL ${oneapi-ck_SOURCE_DIR}/compiler_passes)
      set(OCK_BINARY_DIR_INTERNAL ${oneapi-ck_BINARY_DIR})
    endif()
  elseif(OCK_SOURCE_DIR)
    set(OCK_SOURCE_DIR_INTERNAL "${OCK_SOURCE_DIR}/compiler_passes")
    set(OCK_BINARY_DIR_INTERNAL "${CMAKE_CURRENT_BINARY_DIR}/oneapi-construction-kit")
  else()
    message(FATAL_ERROR "NATIVECPU_OCK_USE_FETCHCONTENT is Off and OCK_SOURCE_DIR not set")
  endif()

  set(CA_ENABLE_API "cl" CACHE STRING "" FORCE)
  add_subdirectory(
    ${OCK_SOURCE_DIR_INTERNAL}
    ${OCK_BINARY_DIR_INTERNAL} EXCLUDE_FROM_ALL)

  install(TARGETS compiler-pipeline
  EXPORT;LLVMExports
          LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT compiler-pipeline
          ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT compiler-pipeline
          RUNTIME DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT compiler-pipeline)
  set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS compiler-pipeline)
  install(TARGETS vecz
  EXPORT;LLVMExports
          LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT vecz
          ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT vecz
          RUNTIME DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT vecz)
  set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS vecz)
  install(TARGETS multi_llvm EXPORT;LLVMExports)
  set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS multi_llvm)
  target_compile_definitions(LLVMSYCLNativeCPUUtils PRIVATE  NATIVECPU_USE_OCK)
  target_include_directories(LLVMSYCLNativeCPUUtils PRIVATE 
    ${oneapi-ck_SOURCE_DIR}/modules/compiler/multi_llvm/include
    ${oneapi-ck_SOURCE_DIR}/modules/cargo/include
    ${oneapi-ck_SOURCE_DIR}/modules/compiler/vecz/include
    ${oneapi-ck_SOURCE_DIR}/modules/compiler/utils/include)
  target_link_libraries(LLVMSYCLNativeCPUUtils PRIVATE compiler-pipeline vecz)

endif()
