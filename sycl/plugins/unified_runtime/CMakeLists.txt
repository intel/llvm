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
  set(UR_OPENCL_ICD_LOADER_LIBRARY OpenCL-ICD CACHE FILEPATH
    "Path of the OpenCL ICD Loader library" FORCE)
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

  set(UNIFIED_RUNTIME_REPO "https://github.com/oneapi-src/unified-runtime.git")
  # commit ceba4a5f8952c7045c7818fb3bef4b439aa47df3
  # Merge: b76d9078 81b75993
  # Author: aarongreig <aaron.greig@codeplay.com>
  # Date:   Wed Feb 7 14:55:22 2024 +0000
  #    Merge pull request #1320 from aarongreig/aaron/hotfix1192
  #    Fix CI regression by generating missing bits in nullddi and ldrddi
  set(UNIFIED_RUNTIME_TAG ceba4a5f8952c7045c7818fb3bef4b439aa47df3)

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
  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/unified-runtime)
    message(FATAL_ERROR
      "SYCL_PI_UR_USE_FETCH_CONTENT is disabled but no alternative Unified \
      Runtime source directory has been provided, either:

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
set(UNIFIED_RUNTIME_SRC_INCLUDE_DIR "${UNIFIED_RUNTIME_SOURCE_DIR}/source")
set(UNIFIED_RUNTIME_COMMON_INCLUDE_DIR "${UNIFIED_RUNTIME_SOURCE_DIR}/source/common")

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
    # These below belong to Unified Runtime PI Plugin only
    "pi_unified_runtime.hpp"
    "pi_unified_runtime.cpp"
  LIBRARIES
    Threads::Threads
    UnifiedRuntimeLoader
    UnifiedRuntime-Headers
    UnifiedRuntimeCommon
  INCLUDE_DIRS
    "${UNIFIED_RUNTIME_SRC_INCLUDE_DIR}"
    "${UNIFIED_RUNTIME_COMMON_INCLUDE_DIR}"
)

# We need for #include <ze_api.h> in common.h
if("level_zero" IN_LIST SYCL_ENABLE_PLUGINS)
  list(APPEND UNIFIED_RUNTIME_PLUGIN_ARGS LevelZeroLoader-Headers)
endif()

if("opencl" IN_LIST SYCL_ENABLE_PLUGINS)
  list(APPEND UNIFIED_RUNTIME_PLUGIN_ARGS OpenCL-ICD)
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

    # Install the UR adapters too
    install(TARGETS ur_adapter_opencl
      LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT level-zero-sycl-dev
      ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT level-zero-sycl-dev
      RUNTIME DESTINATION "bin" COMPONENT level-zero-sycl-dev
    )
endif()

if ("native_cpu" IN_LIST SYCL_ENABLE_PLUGINS)
    add_dependencies(sycl-runtime-libraries ur_adapter_native_cpu)

  option(NATIVECPU_USE_OCK "Use the oneAPI Construction Kit for Native CPU" ON)

  if(NATIVECPU_USE_OCK)
    message(STATUS "Compiling Native CPU adapter with OCK support.")
    target_compile_definitions(ur_adapter_native_cpu PRIVATE NATIVECPU_USE_OCK)
  else()
    message(WARNING "Compiling Native CPU adapter without OCK support.
    Some valid SYCL programs may not build or may have low performance.")
  endif()
endif()
