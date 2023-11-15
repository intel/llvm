# Plugin for SYCL Native CPU
# Create shared library for libpi_nativecpu.so

# Get the Native CPU adapter sources so they can be shared with the Native CPU PI plugin
get_target_property(UR_NATIVE_CPU_ADAPTER_SOURCES ur_adapter_native_cpu SOURCES)

add_sycl_plugin(native_cpu
  SOURCES
    ${UR_NATIVE_CPU_ADAPTER_SOURCES}
    # Some code is shared with the UR adapter
    "../unified_runtime/pi2ur.hpp"
    "../unified_runtime/pi2ur.cpp"
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
  # TODO: I'm not sure why we need this here, but we get errors when configuring without it 
  if(NOT CMAKE_CLC_COMPILE_OBJECT)
    set(CMAKE_CLC_COMPILE_OBJECT
      "<CMAKE_CLC_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE> -emit-llvm")
  endif()
  if(NOT CMAKE_CLC_CREATE_STATIC_LIBRARY)
    set(CMAKE_CLC_CREATE_STATIC_LIBRARY
      "<CMAKE_CLC_ARCHIVE> -o <TARGET> <OBJECTS>")
  endif()
  set(CMAKE_INCLUDE_FLAG_CLC "-I")

  include(FetchContent)
  FetchContent_Declare(oneapi-ck
   GIT_REPOSITORY https://github.com/PietroGhg/oneapi-construction-kit.git
   GIT_TAG        pietro/vecz
  )
  FetchContent_GetProperties(oneapi-ck)
  if(NOT oneapi-ck_POPULATED)
    message(STATUS "Cloning oneAPI Construction Kit")
    FetchContent_Populate(oneapi-ck)
    message(STATUS "oneAPI Construction Kit cloned in ${oneapi-ck_SOURCE_DIR}")
    set(CA_NATIVE_CPU 1)
    add_subdirectory(${oneapi-ck_SOURCE_DIR} ${oneapi-ck_BINARY_DIR})
  endif()
  target_compile_definitions(LLVMSYCLLowerIR PRIVATE  NATIVECPU_USE_OCK)
  target_include_directories(LLVMSYCLLowerIR PRIVATE 
    ${oneapi-ck_SOURCE_DIR}/modules/compiler/multi_llvm/include
    ${oneapi-ck_SOURCE_DIR}/modules/cargo/include
    ${oneapi-ck_SOURCE_DIR}/modules/compiler/vecz/include
    ${oneapi-ck_SOURCE_DIR}/modules/compiler/utils/include)
  target_link_libraries(LLVMSYCLLowerIR PRIVATE compiler-utils vecz)
  target_compile_definitions(pi_native_cpu PRIVATE NATIVECPU_USE_OCK)

endif()
set_target_properties(pi_native_cpu PROPERTIES LINKER_LANGUAGE CXX)
