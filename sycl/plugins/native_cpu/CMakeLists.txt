# Plugin for SYCL Native CPU
# Create shared library for libpi_nativecpu.so

# Get the Native CPU adapter sources so they can be shared with the Native CPU PI plugin
get_target_property(UR_NATIVE_CPU_ADAPTER_SOURCES ur_adapter_native_cpu SOURCES)

add_sycl_plugin(native_cpu
  SOURCES
    ${UR_NATIVE_CPU_ADAPTER_SOURCES}
    # Some code is shared with the UR adapter
    "../unified_runtime/pi2ur.hpp"
    "${sycl_inc_dir}/sycl/detail/pi.h"
    "${sycl_inc_dir}/sycl/detail/pi.hpp"
    "pi_native_cpu.cpp"
    "pi_native_cpu.hpp"
  INCLUDE_DIRS
    ${sycl_inc_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}/../unified_runtime # for Unified Runtime
    ${UNIFIED_RUNTIME_SOURCE_DIR}/source/ # for adapters/native_cpu
  LIBRARIES
    sycl
    UnifiedRuntime-Headers
    UnifiedRuntimeCommon
)

set(OCK_SOURCE_DIR "" CACHE PATH "Root of the local checkout of the oneAPI Construction Kit")
set(OCK_GIT_REPO "" CACHE STRING "Git repository for the oneAPI Construction Kit FetchContent")
set(OCK_GIT_TAG "" CACHE STRING "Git tag for the oneAPI Construction Kit FetchContent")
option(NATIVECPU_OCK_USE_FETCHCONTENT "Use FetchContent to acquire oneAPI Construction Kit source code" On)

if(NATIVECPU_USE_OCK)
  if(NATIVECPU_OCK_USE_FETCHCONTENT)
    set(OCK_GIT_INTERNAL_REPO "https://github.com/codeplaysoftware/oneapi-construction-kit.git")
    # commit d27e6cfb836f306d2dcefbc418c73c16877ef4eb
    # Merge: f02b80de 0fa75c83
    # Author: Harald van Dijk <harald.vandijk@codeplay.com>
    # Date:   Mon Mar 18 17:44:12 2024 +0000
    # Merge pull request #406 from hvdijk/fix-vkeiep
    # Fix vkEnumerateInstanceExtensionProperties.
    set(OCK_GIT_INTERNAL_TAG  d27e6cfb836f306d2dcefbc418c73c16877ef4eb)

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
      set(OCK_SOURCE_DIR_INTERNAL ${oneapi-ck_SOURCE_DIR})
      set(OCK_BINARY_DIR_INTERNAL ${oneapi-ck_BINARY_DIR})
    endif()
  elseif(OCK_SOURCE_DIR)
    set(OCK_SOURCE_DIR_INTERNAL "${OCK_SOURCE_DIR}")
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
  target_compile_definitions(LLVMSYCLLowerIR PRIVATE  NATIVECPU_USE_OCK)
  target_include_directories(LLVMSYCLLowerIR PRIVATE 
    ${oneapi-ck_SOURCE_DIR}/modules/compiler/multi_llvm/include
    ${oneapi-ck_SOURCE_DIR}/modules/cargo/include
    ${oneapi-ck_SOURCE_DIR}/modules/compiler/vecz/include
    ${oneapi-ck_SOURCE_DIR}/modules/compiler/utils/include)
  target_link_libraries(LLVMSYCLLowerIR PRIVATE compiler-pipeline vecz)
  target_compile_definitions(pi_native_cpu PRIVATE NATIVECPU_USE_OCK)

endif()
set_target_properties(pi_native_cpu PROPERTIES LINKER_LANGUAGE CXX)
