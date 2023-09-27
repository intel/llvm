# PI Unified Runtime plugin library.
#
if (NOT DEFINED UNIFIED_RUNTIME_LIBRARY OR NOT DEFINED UNIFIED_RUNTIME_INCLUDE_DIR)
  include(FetchContent)

  # The UR tag should be from the 'adapters' branch
  set(UNIFIED_RUNTIME_REPO "https://github.com/oneapi-src/unified-runtime.git")
  set(UNIFIED_RUNTIME_TAG c791b8bba63af1c1880ae278e9d6df90021636dd)

  set(UR_BUILD_ADAPTER_L0 ON)
  set(UMF_ENABLE_POOL_TRACKING ON)
  message(STATUS "Will fetch Unified Runtime from ${UNIFIED_RUNTIME_REPO}")
  FetchContent_Declare(unified-runtime
    GIT_REPOSITORY    ${UNIFIED_RUNTIME_REPO}
    GIT_TAG           ${UNIFIED_RUNTIME_TAG}
  )

  # Disable errors from warnings while building the UR.
  # And remember origin flags before doing that.
  set(CMAKE_CXX_FLAGS_BAK "${CMAKE_CXX_FLAGS}")
  if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX-")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /WX-")
    # FIXME: Unified runtime build fails with /DUNICODE
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /UUNICODE")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /UUNICODE")
    # USE_Z7 forces use of /Z7 instead of /Zi which is broken with sccache
    set(USE_Z7 ON)
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error")
  endif()

  # No need to build tests from unified-runtime
  set(UR_BUILD_TESTS "0" CACHE STRING "0")

  FetchContent_GetProperties(unified-runtime)
  FetchContent_MakeAvailable(unified-runtime)

  # Restore original flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BAK}")

  add_library(UnifiedRuntimeLoader ALIAS ur_loader)
  add_library(UnifiedRuntimeCommon ALIAS ur_common)
  add_library(UnifiedMallocFramework ALIAS unified_malloc_framework)

  set(UNIFIED_RUNTIME_SOURCE_DIR
    ${unified-runtime_SOURCE_DIR} CACHE PATH "Path to Unified Runtime Headers")
  set(UNIFIED_RUNTIME_INCLUDE_DIR "${UNIFIED_RUNTIME_SOURCE_DIR}/include")
endif()

add_library(UnifiedRuntime-Headers INTERFACE)

target_include_directories(UnifiedRuntime-Headers
  INTERFACE
    "${UNIFIED_RUNTIME_INCLUDE_DIR}"
)

find_package(Threads REQUIRED)

add_sycl_plugin(unified_runtime
  SOURCES
    # These are short-term shared with Unified Runtime
    # The two plugins define a few things differrently so must
    # be built separately. This difference is spelled in
    # their "ur_bindings.hpp" files.
    #
    "ur_bindings.hpp"
    "pi2ur.hpp"
    "pi2ur.cpp"
    # These below belong to Unified Runtime PI Plugin only
    "pi_unified_runtime.hpp"
    "pi_unified_runtime.cpp"
    LIBRARIES
    Threads::Threads
    UnifiedRuntimeLoader
    UnifiedRuntime-Headers
    LevelZeroLoader-Headers # we need for #include <ze_api.h> in common.h
)

add_dependencies(sycl-runtime-libraries ur_adapter_level_zero)

if ("cuda" IN_LIST SYCL_ENABLE_PLUGINS)
  # Build CUDA adapter
  add_sycl_library("ur_adapter_cuda" SHARED
    SOURCES
      "ur/ur.hpp"
      "ur/ur.cpp"
      "ur/adapters/cuda/adapter.cpp"
      "ur/adapters/cuda/adapter.hpp"
      "ur/adapters/cuda/command_buffer.cpp"
      "ur/adapters/cuda/command_buffer.hpp"
      "ur/adapters/cuda/common.cpp"
      "ur/adapters/cuda/common.hpp"
      "ur/adapters/cuda/context.cpp"
      "ur/adapters/cuda/context.hpp"
      "ur/adapters/cuda/device.cpp"
      "ur/adapters/cuda/device.hpp"
      "ur/adapters/cuda/enqueue.cpp"
      "ur/adapters/cuda/event.cpp"
      "ur/adapters/cuda/event.hpp"
      "ur/adapters/cuda/image.cpp"
      "ur/adapters/cuda/image.hpp"
      "ur/adapters/cuda/kernel.cpp"
      "ur/adapters/cuda/kernel.hpp"
      "ur/adapters/cuda/memory.cpp"
      "ur/adapters/cuda/memory.hpp"
      "ur/adapters/cuda/platform.cpp"
      "ur/adapters/cuda/platform.hpp"
      "ur/adapters/cuda/program.cpp"
      "ur/adapters/cuda/program.hpp"
      "ur/adapters/cuda/queue.cpp"
      "ur/adapters/cuda/queue.hpp"
      "ur/adapters/cuda/sampler.cpp"
      "ur/adapters/cuda/sampler.hpp"
      "ur/adapters/cuda/tracing.cpp"
      "ur/adapters/cuda/ur_interface_loader.cpp"
      "ur/adapters/cuda/usm.cpp"
      "ur/adapters/cuda/usm.hpp"
      "ur/adapters/cuda/usm_p2p.cpp"
    INCLUDE_DIRS
      ${sycl_inc_dir}
    LIBRARIES
      UnifiedRuntime-Headers
      UnifiedRuntimeCommon
      Threads::Threads
      cudadrv
  )

  set_target_properties("ur_adapter_cuda" PROPERTIES
    VERSION "0.0.0"
    SOVERSION "0"
  )

  if(UMF_ENABLE_POOL_TRACKING)
    target_compile_definitions("ur_adapter_cuda" PRIVATE
      UMF_ENABLE_POOL_TRACKING)
  else()
    message(WARNING "CUDA adapter USM pools are disabled, set UMF_ENABLE_POOL_TRACKING to enable them")
  endif()
endif()

if ("hip" IN_LIST SYCL_ENABLE_PLUGINS)
  # Build HIP adapter
  add_sycl_library("ur_adapter_hip" SHARED
    SOURCES
      "ur/ur.hpp"
      "ur/ur.cpp"
      "ur/adapters/hip/adapter.cpp"
      "ur/adapters/hip/adapter.hpp"
      "ur/adapters/hip/command_buffer.cpp"
      "ur/adapters/hip/command_buffer.hpp"
      "ur/adapters/hip/common.cpp"
      "ur/adapters/hip/common.hpp"
      "ur/adapters/hip/context.cpp"
      "ur/adapters/hip/context.hpp"
      "ur/adapters/hip/device.cpp"
      "ur/adapters/hip/device.hpp"
      "ur/adapters/hip/enqueue.cpp"
      "ur/adapters/hip/event.cpp"
      "ur/adapters/hip/event.hpp"
      "ur/adapters/hip/image.cpp"
      "ur/adapters/hip/kernel.cpp"
      "ur/adapters/hip/kernel.hpp"
      "ur/adapters/hip/memory.cpp"
      "ur/adapters/hip/memory.hpp"
      "ur/adapters/hip/platform.cpp"
      "ur/adapters/hip/platform.hpp"
      "ur/adapters/hip/program.cpp"
      "ur/adapters/hip/program.hpp"
      "ur/adapters/hip/queue.cpp"
      "ur/adapters/hip/queue.hpp"
      "ur/adapters/hip/sampler.cpp"
      "ur/adapters/hip/sampler.hpp"
      "ur/adapters/hip/ur_interface_loader.cpp"
      "ur/adapters/hip/usm.cpp"
      "ur/adapters/hip/usm.hpp"
      "ur/adapters/hip/usm_p2p.cpp"
    INCLUDE_DIRS
      ${sycl_inc_dir}
    LIBRARIES
      UnifiedRuntime-Headers
      UnifiedRuntimeCommon
      Threads::Threads
  )

  set_target_properties("ur_adapter_hip" PROPERTIES
    VERSION "0.0.0"
    SOVERSION "0"
  )

  if(UMF_ENABLE_POOL_TRACKING)
    target_compile_definitions("ur_adapter_hip" PRIVATE
      UMF_ENABLE_POOL_TRACKING)
  else()
    message(WARNING "HIP adapter USM pools are disabled, set UMF_ENABLE_POOL_TRACKING to enable them")
  endif()

  if("${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "AMD")
    target_link_libraries(ur_adapter_hip PUBLIC rocmdrv)
    # Set HIP define to select AMD platform
    target_compile_definitions(ur_adapter_hip PRIVATE __HIP_PLATFORM_AMD__)
  elseif("${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "NVIDIA")
    target_link_libraries(ur_adapter_hip PUBLIC cudadrv cudart)
    # Set HIP define to select NVIDIA platform
    target_compile_definitions(ur_adapter_hip PRIVATE __HIP_PLATFORM_NVIDIA__)
  else()
    message(FATAL_ERROR "Unspecified PI HIP platform please set SYCL_BUILD_PI_HIP_PLATFORM to 'AMD' or 'NVIDIA'")
  endif()
endif()

if("native_cpu" IN_LIST SYCL_ENABLE_PLUGINS)
  add_sycl_library("ur_adapter_native_cpu" SHARED
    SOURCES
      "ur/ur.cpp"
      "ur/ur.hpp"
      "ur/adapters/native_cpu/adapter.cpp"
      "ur/adapters/native_cpu/command_buffer.cpp"
      "ur/adapters/native_cpu/common.cpp"
      "ur/adapters/native_cpu/common.hpp"
      "ur/adapters/native_cpu/context.cpp"
      "ur/adapters/native_cpu/context.hpp"
      "ur/adapters/native_cpu/device.cpp"
      "ur/adapters/native_cpu/device.hpp"
      "ur/adapters/native_cpu/enqueue.cpp"
      "ur/adapters/native_cpu/event.cpp"
      "ur/adapters/native_cpu/image.cpp"
      "ur/adapters/native_cpu/kernel.cpp"
      "ur/adapters/native_cpu/kernel.hpp"
      "ur/adapters/native_cpu/memory.cpp"
      "ur/adapters/native_cpu/memory.hpp"
      "ur/adapters/native_cpu/platform.cpp"
      "ur/adapters/native_cpu/platform.hpp"
      "ur/adapters/native_cpu/program.cpp"
      "ur/adapters/native_cpu/program.hpp"
      "ur/adapters/native_cpu/queue.cpp"
      "ur/adapters/native_cpu/queue.hpp"
      "ur/adapters/native_cpu/sampler.cpp"
      "ur/adapters/native_cpu/ur_interface_loader.cpp"
      "ur/adapters/native_cpu/usm.cpp"
      "ur/adapters/native_cpu/usm_p2p.cpp"
    INCLUDE_DIRS
      ${sycl_inc_dir}
    LIBRARIES
      UnifiedRuntime-Headers
      Threads::Threads
      sycl
      OpenCL-Headers
  )

  set_target_properties("ur_adapter_native_cpu" PROPERTIES
    VERSION "0.0.0"
    SOVERSION "0"
  )
endif()


if (TARGET UnifiedRuntimeLoader)
  set_target_properties(hello_world PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
  # Install the UR loader.
  # TODO: this is piggy-backing on the existing target component level-zero-sycl-dev
  # When UR is moved to its separate repo perhaps we should introduce new component,
  # e.g. unified-runtime-sycl-dev.
  install(TARGETS ur_loader
    LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT level-zero-sycl-dev
    ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT level-zero-sycl-dev
    RUNTIME DESTINATION "bin" COMPONENT level-zero-sycl-dev
  )
endif()

# Install the UR adapters too
install(TARGETS ur_adapter_level_zero
  LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT level-zero-sycl-dev
  ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT level-zero-sycl-dev
  RUNTIME DESTINATION "bin" COMPONENT level-zero-sycl-dev
)
