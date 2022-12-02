# PI Unified Runtime plugin library
#

include(FetchContent)

set(UNIFIED_RUNTIME_REPO "https://github.com/oneapi-src/unified-runtime.git")
set(UNIFIED_RUNTIME_TAG fd711c920acc4434cb52ff18b078c082d9d7f44d)

message(STATUS "Will fetch Unified Runtime from ${UNIFIED_RUNTIME_REPO}")
FetchContent_Declare(unified-runtime
  GIT_REPOSITORY    ${UNIFIED_RUNTIME_REPO}
  GIT_TAG           ${UNIFIED_RUNTIME_TAG}
)

FetchContent_MakeAvailable(unified-runtime)
FetchContent_GetProperties(unified-runtime)

set(UNIFIED_RUNTIME_SOURCE_DIR
  ${unified-runtime_SOURCE_DIR} CACHE PATH "Path to Unified Runtime Headers")
set(UNIFIED_RUNTIME_INCLUDE_DIR "${UNIFIED_RUNTIME_SOURCE_DIR}/include")


add_library (UnifiedRuntime-Headers INTERFACE)
target_include_directories(UnifiedRuntime-Headers
  INTERFACE
    "${UNIFIED_RUNTIME_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS unified-runtime)

find_package(Threads REQUIRED)

#
# Build a helper static library to carry pieces shared between
# this Unified Runtime plugin and Level Zero plugin. We cannot
# use dynamic plugin library as that only exports pi* symbols.
#
add_library(unified_runtime_static STATIC
  "${sycl_inc_dir}/sycl/detail/pi.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/ur.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/ur.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/pi2ur.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/adapters/level_zero/ur_level_zero.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/adapters/level_zero/ur_level_zero.cpp"
)

target_include_directories(unified_runtime_static
  INTERFACE
    "${UNIFIED_RUNTIME_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  PRIVATE
    # For include <sycl/detail/pi.h>
    # TODO: how to get rid of this?
    "${CMAKE_SOURCE_DIR}/../sycl/include"
)

target_link_libraries(unified_runtime_static PRIVATE
  UnifiedRuntime-Headers
  LevelZeroLoader-Headers
)

#
# NOTE: the Unified Runtime doesn't have the loader [yet].
# So what we really build is the Unified Runtime with Level Zero Adapter
# together.
#
add_sycl_plugin(unified_runtime
  SOURCES
    # Put here anything that belongs exclusively to Unified Runtime
    # and should not be shared with the Level Zero plugin
    "${CMAKE_CURRENT_SOURCE_DIR}/pi2ur.cpp"
  INCLUDE_DIRS
    "${UNIFIED_RUNTIME_INCLUDE_DIR}"
  LIBRARIES
    unified_runtime_static
    Threads::Threads
    UnifiedRuntime-Headers
    LevelZeroLoader-Headers
    ze_loader
)
