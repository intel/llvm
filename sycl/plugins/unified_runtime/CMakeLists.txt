# PI Unified Runtime plugin library.
#

# Options to override the default behaviour of the FetchContent to include UR
# source code.
set(SYCL_PI_UR_OVERRIDE_FETCH_CONTENT_REPO
  "" CACHE STRING "Override the Unified Runtime FetchContent repository")
set(SYCL_PI_UR_OVERRIDE_FETCH_CONTENT_TAG
  "" CACHE STRING "Override the Unified Runtime FetchContent tag")

# Options to disable use of FetchContent to include Unified Runtime source code
# to improve developer workflow.
option(SYCL_PI_UR_USE_FETCH_CONTENT
  "Use FetchContent to acquire the Unified Runtime source code" ON)
set(SYCL_PI_UR_SOURCE_DIR
  "" CACHE PATH "Path to root of Unified Runtime repository")

# Override default to enable building tests from unified-runtime
set(UR_BUILD_TESTS OFF CACHE BOOL "Build unit tests.")
set(UMF_ENABLE_POOL_TRACKING ON)

if("level_zero" IN_LIST SYCL_ENABLE_PLUGINS)
  set(UR_BUILD_ADAPTER_L0 ON)
endif()
if("cuda" IN_LIST SYCL_ENABLE_PLUGINS)
  set(UR_BUILD_ADAPTER_CUDA ON)
endif()
if("hip" IN_LIST SYCL_ENABLE_PLUGINS)
  set(UR_BUILD_ADAPTER_HIP ON)
endif()
if("opencl" IN_LIST SYCL_ENABLE_PLUGINS)
  set(UR_BUILD_ADAPTER_OPENCL ON)
  set(UR_OPENCL_ICD_LOADER_LIBRARY OpenCL-ICD)
endif()
if("native_cpu" IN_LIST SYCL_ENABLE_PLUGINS)
  set(UR_BUILD_ADAPTER_NATIVE_CPU ON)
endif()

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

if(SYCL_PI_UR_USE_FETCH_CONTENT)
  include(FetchContent)

  set(UNIFIED_RUNTIME_REPO "https://github.com/steffenlarsen/unified-runtime.git")
  set(UNIFIED_RUNTIME_TAG 5f680fb9688e9d9a4aab172c8ade461c70401460)

  if(SYCL_PI_UR_OVERRIDE_FETCH_CONTENT_REPO)
    set(UNIFIED_RUNTIME_REPO "${SYCL_PI_UR_OVERRIDE_FETCH_CONTENT_REPO}")
  endif()
  if(SYCL_PI_UR_OVERRIDE_FETCH_CONTENT_TAG)
    set(UNIFIED_RUNTIME_TAG "${SYCL_PI_UR_OVERRIDE_FETCH_CONTENT_TAG}")
  endif()

  message(STATUS "Will fetch Unified Runtime from ${UNIFIED_RUNTIME_REPO}")
  FetchContent_Declare(unified-runtime
    GIT_REPOSITORY    ${UNIFIED_RUNTIME_REPO}
    GIT_TAG           ${UNIFIED_RUNTIME_TAG}
  )

  FetchContent_GetProperties(unified-runtime)
  FetchContent_MakeAvailable(unified-runtime)

  set(UNIFIED_RUNTIME_SOURCE_DIR
    "${unified-runtime_SOURCE_DIR}" CACHE PATH
    "Path to Unified Runtime Headers" FORCE)
elseif(SYCL_PI_UR_SOURCE_DIR)
  # SYCL_PI_UR_USE_FETCH_CONTENT is OFF and SYCL_PI_UR_SOURCE_DIR has been set,
  # use the external Unified Runtime source directory.
  set(UNIFIED_RUNTIME_SOURCE_DIR
    "${SYCL_PI_UR_SOURCE_DIR}" CACHE PATH
    "Path to Unified Runtime Headers" FORCE)
  add_subdirectory(
    ${UNIFIED_RUNTIME_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/unified-runtime)
else()
  # SYCL_PI_UR_USE_FETCH_CONTENT is OFF and SYCL_PI_UR_SOURCE_DIR has not been
  # set, check if the fallback local directory exists.
  if(NOT ${CMAKE_CURRENT_SOURCE_DIR}/unified-runtime)
    message(FATAL_ERROR
      "SYCL_PI_UR_USE_FETCH_CONTENT is disabled but no alternative Unified
      Runtime source directory has be proivided, either:

      * Set -DSYCL_PI_UR_SOURCE_DIR=/path/to/unified-runtime
      * Clone the UR repo in ${CMAKE_CURRENT_SOURCE_DIR}/unified-runtime")
  endif()
  # The fallback local directory for the Unified Runtime repository has been
  # found, use it.
  set(UNIFIED_RUNTIME_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/unified-runtime" CACHE PATH
    "Path to Unified Runtime Headers" FORCE)
  add_subdirectory(${UNIFIED_RUNTIME_SOURCE_DIR})
endif()

# Restore original flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BAK}")

message(STATUS
  "Using Unified Runtime source directory: ${UNIFIED_RUNTIME_SOURCE_DIR}")

set(UNIFIED_RUNTIME_INCLUDE_DIR "${UNIFIED_RUNTIME_SOURCE_DIR}/include")

add_library(UnifiedRuntimeLoader ALIAS ur_loader)
add_library(UnifiedRuntimeCommon ALIAS ur_common)
add_library(UnifiedMallocFramework ALIAS unified_malloc_framework)

add_library(UnifiedRuntime-Headers INTERFACE)

target_include_directories(UnifiedRuntime-Headers
  INTERFACE
    "${UNIFIED_RUNTIME_INCLUDE_DIR}"
)

find_package(Threads REQUIRED)

set(UNIFIED_RUNTIME_PLUGIN_ARGS
  SOURCES
    # These are short-term shared with Unified Runtime
    # The two plugins define a few things differently so must
    # be built separately. This difference is spelled in
    # their "ur_bindings.hpp" files.
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
)

# We need for #include <ze_api.h> in common.h
if("level_zero" IN_LIST SYCL_ENABLE_PLUGINS)
  list(APPEND UNIFIED_RUNTIME_PLUGIN_ARGS LevelZeroLoader-Headers)
endif()

<<<<<<< HEAD
if ("opencl" IN_LIST SYCL_ENABLE_PLUGINS)
  # Build OpenCL adapter
  add_sycl_library("ur_adapter_opencl" SHARED
    SOURCES
      "ur/ur.hpp"
      "ur/ur.cpp"
      "ur/adapters/opencl/adapter.cpp"
      "ur/adapters/opencl/adapter.hpp"
      "ur/adapters/opencl/command_buffer.hpp"
      "ur/adapters/opencl/command_buffer.cpp"
      "ur/adapters/opencl/common.cpp"
      "ur/adapters/opencl/common.hpp"
      "ur/adapters/opencl/context.cpp"
      "ur/adapters/opencl/context.hpp"
      "ur/adapters/opencl/device.cpp"
      "ur/adapters/opencl/device.hpp"
      "ur/adapters/opencl/enqueue.cpp"
      "ur/adapters/opencl/event.cpp"
      "ur/adapters/opencl/image.cpp"
      "ur/adapters/opencl/kernel.cpp"
      "ur/adapters/opencl/memory.cpp"
      "ur/adapters/opencl/physical_mem.cpp"
      "ur/adapters/opencl/platform.cpp"
      "ur/adapters/opencl/platform.hpp"
      "ur/adapters/opencl/program.cpp"
      "ur/adapters/opencl/queue.cpp"
      "ur/adapters/opencl/sampler.cpp"
      "ur/adapters/opencl/ur_interface_loader.cpp"
      "ur/adapters/opencl/usm.cpp"
      "ur/adapters/opencl/usm_p2p.cpp"
      "ur/adapters/opencl/virtual_mem.cpp"
    INCLUDE_DIRS
      ${sycl_inc_dir}
    LIBRARIES
      UnifiedRuntime-Headers
      Threads::Threads
      OpenCL-ICD
    )

  # Suppress a compiler message about undefined CL_TARGET_OPENCL_VERSION.
  # Define all symbols up to OpenCL 3.0.
  target_compile_definitions(ur_adapter_opencl PRIVATE CL_TARGET_OPENCL_VERSION=300)

  set_target_properties("ur_adapter_opencl" PROPERTIES
    VERSION "0.0.0"
    SOVERSION "0"
  )
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
=======
if("opencl" IN_LIST SYCL_ENABLE_PLUGINS)
  list(APPEND UNIFIED_RUNTIME_PLUGIN_ARGS OpenCL-ICD)
>>>>>>> intel/sycl
endif()

add_sycl_plugin(unified_runtime ${UNIFIED_RUNTIME_PLUGIN_ARGS})

if(TARGET UnifiedRuntimeLoader)
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
if("level_zero" IN_LIST SYCL_ENABLE_PLUGINS)
  add_dependencies(sycl-runtime-libraries ur_adapter_level_zero)

  # Install the UR adapters too
  install(TARGETS ur_adapter_level_zero
    LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT level-zero-sycl-dev
    ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT level-zero-sycl-dev
    RUNTIME DESTINATION "bin" COMPONENT level-zero-sycl-dev
  )
endif()

if("cuda" IN_LIST SYCL_ENABLE_PLUGINS)
  add_dependencies(sycl-runtime-libraries ur_adapter_cuda)
endif()

if("hip" IN_LIST SYCL_ENABLE_PLUGINS)
  add_dependencies(sycl-runtime-libraries ur_adapter_hip)
endif()

if ("opencl" IN_LIST SYCL_ENABLE_PLUGINS)
    add_dependencies(sycl-runtime-libraries ur_adapter_opencl)
endif()

if ("native_cpu" IN_LIST SYCL_ENABLE_PLUGINS)
    add_dependencies(sycl-runtime-libraries ur_adapter_native_cpu)
endif()
