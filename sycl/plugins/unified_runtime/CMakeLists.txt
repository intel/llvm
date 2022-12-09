# PI Unified Runtime plugin library
#
if (NOT DEFINED UNIFIED_RUNTIME_INCLUDE_DIR)
  include(FetchContent)

  set(UNIFIED_RUNTIME_REPO "https://github.com/oneapi-src/unified-runtime.git")
  set(UNIFIED_RUNTIME_TAG 7ef04f96c1dbf161b7731d0d444183d27e603c57)

  message(STATUS "Will fetch Unified Runtime from ${UNIFIED_RUNTIME_REPO}")
  FetchContent_Declare(unified-runtime
    GIT_REPOSITORY    ${UNIFIED_RUNTIME_REPO}
    GIT_TAG           ${UNIFIED_RUNTIME_TAG}
  )

  FetchContent_MakeAvailable(unified-runtime)
  FetchContent_GetProperties(unified-runtime)

  set(UNIFIED_RUNTIME_LIBRARY unified-runtime::loader)
  set(UNIFIED_RUNTIME_SOURCE_DIR
    ${unified-runtime_SOURCE_DIR} CACHE PATH "Path to Unified Runtime Headers")
  set(UNIFIED_RUNTIME_INCLUDE_DIR "${UNIFIED_RUNTIME_SOURCE_DIR}/include")
endif()

add_library (UnifiedRuntimeLoader INTERFACE)
target_link_libraries(UnifiedRuntimeLoader
  INTERFACE "${UNIFIED_RUNTIME_LIBRARY}"
)

add_library (UnifiedRuntime-Headers INTERFACE)
target_include_directories(UnifiedRuntime-Headers
  INTERFACE
    "${UNIFIED_RUNTIME_INCLUDE_DIR}"
)

find_package(Threads REQUIRED)

# Build Unified Runtime Level Zero Adapter
add_subdirectory(ur/adapters/level_zero)

add_sycl_plugin(unified_runtime
  SOURCES
    "pi_unified_runtime.hpp"
    "pi_unified_runtime.cpp"
    LIBRARIES
    Threads::Threads
    UnifiedRuntimeLoader
    UnifiedRuntime-Headers
)
