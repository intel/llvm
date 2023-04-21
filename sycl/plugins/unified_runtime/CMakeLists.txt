# PI Unified Runtime plugin library
#
if (NOT DEFINED UNIFIED_RUNTIME_LIBRARY OR NOT DEFINED UNIFIED_RUNTIME_INCLUDE_DIR)
  include(FetchContent)

  set(UNIFIED_RUNTIME_REPO "https://github.com/oneapi-src/unified-runtime.git")
  set(UNIFIED_RUNTIME_TAG 74843ea0800e6fb7ce0f82e0ef991fc258f4b9bd)

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

  add_library(UnifiedRuntimeLoader ALIAS loader)

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
)

# Build level zero adapter
add_sycl_library("ur_adapter_level_zero" SHARED
  SOURCES
    "ur/ur.hpp"
    "ur/ur.cpp"
    "ur/usm_allocator.hpp"
    "ur/usm_allocator.cpp"
    "ur/usm_allocator_config.cpp"
    "ur/usm_allocator_config.hpp"
    "ur/adapters/level_zero/ur_level_zero.hpp"
    "ur/adapters/level_zero/ur_level_zero.cpp"
    "ur/adapters/level_zero/ur_loader_interface.cpp"
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
  INCLUDE_DIRS
    ${sycl_inc_dir}
  LIBRARIES
    UnifiedRuntime-Headers
    LevelZeroLoader-Headers
    LevelZeroLoader
    Threads::Threads
)

set_target_properties("ur_adapter_level_zero" PROPERTIES
    VERSION "0.0.0"
    SOVERSION "0"
)


if (TARGET UnifiedRuntimeLoader)
  set_target_properties(hello_world PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
endif()
