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


if(NATIVECPU_USE_OCK)
  include(FetchContent)
  FetchContent_Declare(oneapi-ck
   GIT_REPOSITORY https://github.com/codeplaysoftware/oneapi-construction-kit.git
   # commit 62090f9d469a30e0595f445ff664ea2ecfe77b44
   # Merge: 6151546a af33ddfb
   # Author: PietroGhg <pietro.ghiglio@codeplay.com>
   # Date:   Thu Feb 29 12:46:49 2024 +0100
   # Merge pull request #385 from PietroGhg/pietro/multi_llvm_include
   # [NATIVECPU] Avoid specifying include path manually for multi_llvm and Native CPU
   GIT_TAG  62090f9d469a30e0595f445ff664ea2ecfe77b44
  )
  FetchContent_GetProperties(oneapi-ck)
  if(NOT oneapi-ck_POPULATED)
    message(STATUS "Cloning oneAPI Construction Kit")
    FetchContent_Populate(oneapi-ck)
    message(STATUS "oneAPI Construction Kit cloned in ${oneapi-ck_SOURCE_DIR}")
    set(CA_NATIVE_CPU 1)
    add_subdirectory(${oneapi-ck_SOURCE_DIR} ${oneapi-ck_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
  install(TARGETS multi_llvm
  EXPORT;LLVMExports
        LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT multi_llvm
        ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT multi_llvm
        RUNTIME DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT multi_llvm)
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
