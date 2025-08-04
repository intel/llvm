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
  SYCLLowerIR
  Target
  TargetParser
  TransformUtils
  ipo
  )

option(NATIVECPU_USE_OCK "Use the oneAPI Construction Kit for Native CPU" ON)

# Don't fetch OCK if Native CPU is not enabled.
if(NOT "native_cpu" IN_LIST SYCL_ENABLE_BACKENDS)
  set(NATIVECPU_USE_OCK Off CACHE BOOL "Use the oneAPI Construction Kit for Native CPU" FORCE)
endif()

if(NATIVECPU_USE_OCK)
  set(OCK_SEARCH_LOC "oneapi-construction-kit/compiler_passes")
  if(NOT FETCHCONTENT_SOURCE_DIR_ONEAPI-CK)
    find_path(OCK_SOURCE_DIR ${OCK_SEARCH_LOC} PATHS ${CMAKE_PREFIX_PATH})
  endif()
  if(OCK_SOURCE_DIR)
    message(STATUS "Found system source location of oneAPI Construction Kit in ${OCK_SOURCE_DIR}")
    set(OCK_SOURCE_DIR "${OCK_SOURCE_DIR}/${OCK_SEARCH_LOC}")
    set(OCK_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/oneapi-construction-kit")
  else()
    set(OCK_GIT_REPO "https://github.com/uxlfoundation/oneapi-construction-kit.git")
    # commit d0a32d701e34b3285de7ce776ea36abfec673df7
    # Merge: a9f848e0e8 56473a8c25
    # Author: Harald van Dijk <harald.vandijk@codeplay.com>
    # Date:   Mon Jun 30 12:24:46 2025 +0100
    # 
    #     Merge pull request #878 from hvdijk/specify-fuse-ld-lld
    #     
    #     [RefSi] Explicitly specify -fuse-ld=lld.
    set(OCK_GIT_TAG d0a32d701e34b3285de7ce776ea36abfec673df7)
   
    include(FetchContent)
    FetchContent_Declare(oneapi-ck
     GIT_REPOSITORY "${OCK_GIT_REPO}"
     GIT_TAG "${OCK_GIT_TAG}"
    )
    FetchContent_GetProperties(oneapi-ck)
    if(NOT oneapi-ck_POPULATED)
      if(FETCHCONTENT_SOURCE_DIR_ONEAPI-CK)
        message(STATUS "Using specified oneAPI Consutrction Kit repo location at ${FETCHCONTENT_SOURCE_DIR_ONEAPI-CK}")
      else()
        message(STATUS "Cloning oneAPI Construction Kit from ${OCK_GIT_REPO}, tag ${OCK_GIT_TAG}")
      endif()
      FetchContent_Populate(oneapi-ck)
      message(STATUS "oneAPI Construction Kit cloned in ${oneapi-ck_SOURCE_DIR}")
      set(OCK_SOURCE_DIR ${oneapi-ck_SOURCE_DIR}/compiler_passes)
      set(OCK_BINARY_DIR ${oneapi-ck_BINARY_DIR})
    endif()
  endif()

  set(CA_ENABLE_API "cl" CACHE STRING "" FORCE)
  add_subdirectory(
    ${OCK_SOURCE_DIR}
    ${OCK_BINARY_DIR} EXCLUDE_FROM_ALL)

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
