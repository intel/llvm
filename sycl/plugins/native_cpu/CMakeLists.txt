add_sycl_plugin(native_cpu
  SOURCES
    "pi_native_cpu.cpp"
    "../unified_runtime/pi2ur.hpp"
    "../unified_runtime/pi2ur.cpp"
    "../unified_runtime/ur/ur.hpp"
    "../unified_runtime/ur/ur.cpp"
    "../unified_runtime/ur/adapters/native_cpu/adapter.cpp"
    "../unified_runtime/ur/adapters/native_cpu/command_buffer.cpp"
    "../unified_runtime/ur/adapters/native_cpu/common.cpp"
    "../unified_runtime/ur/adapters/native_cpu/common.hpp"
    "../unified_runtime/ur/adapters/native_cpu/context.cpp"
    "../unified_runtime/ur/adapters/native_cpu/context.hpp"
    "../unified_runtime/ur/adapters/native_cpu/device.cpp"
    "../unified_runtime/ur/adapters/native_cpu/device.hpp"
    "../unified_runtime/ur/adapters/native_cpu/enqueue.cpp"
    "../unified_runtime/ur/adapters/native_cpu/event.cpp"
    "../unified_runtime/ur/adapters/native_cpu/image.cpp"
    "../unified_runtime/ur/adapters/native_cpu/kernel.cpp"
    "../unified_runtime/ur/adapters/native_cpu/kernel.hpp"
    "../unified_runtime/ur/adapters/native_cpu/memory.cpp"
    "../unified_runtime/ur/adapters/native_cpu/memory.hpp"
    "../unified_runtime/ur/adapters/native_cpu/platform.cpp"
    "../unified_runtime/ur/adapters/native_cpu/platform.hpp"
    "../unified_runtime/ur/adapters/native_cpu/program.cpp"
    "../unified_runtime/ur/adapters/native_cpu/program.hpp"
    "../unified_runtime/ur/adapters/native_cpu/queue.cpp"
    "../unified_runtime/ur/adapters/native_cpu/queue.hpp"
    "../unified_runtime/ur/adapters/native_cpu/sampler.cpp"
    "../unified_runtime/ur/adapters/native_cpu/ur_interface_loader.cpp"
    "../unified_runtime/ur/adapters/native_cpu/usm.cpp"
    "../unified_runtime/ur/adapters/native_cpu/usm_p2p.cpp"
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../unified_runtime
  LIBRARIES
    sycl
    UnifiedRuntime-Headers
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
   GIT_REPOSITORY https://github.com/codeplaysoftware/oneapi-construction-kit.git
   GIT_TAG        sycl_native_experimental
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
    ${oneapi-ck_SOURCE_DIR}/modules/compiler/utils/include)
  target_link_libraries(LLVMSYCLLowerIR PRIVATE compiler-utils)
  target_compile_definitions(pi_native_cpu PRIVATE NATIVECPU_USE_OCK)

endif()
