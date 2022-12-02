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
)


#
# Build a helper static library to carry pieces shared between
# this Unified Runtime plugin and Level Zero plugin. We cannot
# use dynamic plugin library as that only exports pi* symbols.
#
add_library(unified_runtime_static STATIC
  "ur.hpp"
  "ur.cpp"
  "pi2ur.hpp"
  "adapters/level_zero/ur_level_zero.hpp"
  "adapters/level_zero/ur_level_zero.cpp"
)

target_include_directories(unified_runtime_static
  INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
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
find_package(Threads REQUIRED)

add_sycl_plugin(unified_runtime
  SOURCES
    # Put here anything that belongs exclusively to Unified Runtime
    # and should not be shared with the Level Zero plugin
    "pi2ur.cpp"
  LIBRARIES
    unified_runtime_static
    Threads::Threads
    UnifiedRuntime-Headers
    LevelZeroLoader-Headers
    ze_loader
)
