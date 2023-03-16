# PI Unified Runtime plugin library
#
if (NOT DEFINED UNIFIED_RUNTIME_LIBRARY OR NOT DEFINED UNIFIED_RUNTIME_INCLUDE_DIR)
  include(FetchContent)

  set(UNIFIED_RUNTIME_REPO "https://github.com/oneapi-src/unified-runtime.git")
  set(UNIFIED_RUNTIME_TAG 6fb8e2620f1474428a539ef46d6dd47043c7d59b)

  message(STATUS "Will fetch Unified Runtime from ${UNIFIED_RUNTIME_REPO}")
  FetchContent_Declare(unified-runtime
    GIT_REPOSITORY    ${UNIFIED_RUNTIME_REPO}
    GIT_TAG           ${UNIFIED_RUNTIME_TAG}
  )

  # Disable errors from warnings while building the UR.
  # And remember origin flags before doing that.
  set(CMAKE_CXX_FLAGS_BAK "${CMAKE_CXX_FLAGS}")
  if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")
  endif()

  # No need to build tests from unified-runtime
  set(UR_BUILD_TESTS "0" CACHE STRING "0")

  FetchContent_GetProperties(unified-runtime)
  FetchContent_MakeAvailable(unified-runtime)

  # Restore original flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BAK}")

  add_library(${PROJECT_NAME}::ur_loader ALIAS loader)

  set(UNIFIED_RUNTIME_LIBRARY ${PROJECT_NAME}::ur_loader)

  set(UNIFIED_RUNTIME_SOURCE_DIR
    ${unified-runtime_SOURCE_DIR} CACHE PATH "Path to Unified Runtime Headers")
  set(UNIFIED_RUNTIME_INCLUDE_DIR "${UNIFIED_RUNTIME_SOURCE_DIR}/include")
endif()

add_library (UnifiedRuntime-Headers INTERFACE)
target_include_directories(UnifiedRuntime-Headers
  INTERFACE
    "${UNIFIED_RUNTIME_INCLUDE_DIR}"
)

find_package(Threads REQUIRED)

#
# NOTE: the Unified Runtime doesn't have the loader [yet].
# So what we really build is the Unified Runtime with Level Zero Adapter
# together.
#
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
    "ur/ur.hpp"
    "ur/ur.cpp"
    "ur/adapters/level_zero/ur_level_zero.hpp"
    "ur/adapters/level_zero/ur_level_zero_common.hpp"
    "ur/adapters/level_zero/ur_level_zero_context.hpp"
    "ur/adapters/level_zero/ur_level_zero_device.hpp"
    "ur/adapters/level_zero/ur_level_zero_event.hpp"
    "ur/adapters/level_zero/ur_level_zero_mem.hpp"
    "ur/adapters/level_zero/ur_level_zero_module.hpp"
    "ur/adapters/level_zero/ur_level_zero_platform.hpp"
    "ur/adapters/level_zero/ur_level_zero_program.hpp"
    "ur/adapters/level_zero/ur_level_zero_queue.hpp"
    "ur/adapters/level_zero/ur_level_zero_sampler.hpp"
    "ur/adapters/level_zero/ur_level_zero.cpp"
    "ur/adapters/level_zero/ur_level_zero_common.cpp"
    "ur/adapters/level_zero/ur_level_zero_context.cpp"
    "ur/adapters/level_zero/ur_level_zero_device.cpp"
    "ur/adapters/level_zero/ur_level_zero_event.cpp"
    "ur/adapters/level_zero/ur_level_zero_mem.cpp"
    "ur/adapters/level_zero/ur_level_zero_module.cpp"
    "ur/adapters/level_zero/ur_level_zero_platform.cpp"
    "ur/adapters/level_zero/ur_level_zero_program.cpp"
    "ur/adapters/level_zero/ur_level_zero_queue.cpp"
    "ur/adapters/level_zero/ur_level_zero_sampler.cpp"
    "ur/usm_allocator.hpp"
    "ur/usm_allocator.cpp"
    # These below belong to Unified Runtime PI Plugin only
    "pi_unified_runtime.hpp"
    "pi_unified_runtime.cpp"
    LIBRARIES
    Threads::Threads
    UnifiedRuntime-Headers
    LevelZeroLoader-Headers
    LevelZeroLoader
)

if (TARGET ${PROJECT_NAME}::ur_loader)
  set_target_properties(hello_world PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
endif()
