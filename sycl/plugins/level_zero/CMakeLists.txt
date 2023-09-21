# PI Level Zero plugin library

if (NOT DEFINED LEVEL_ZERO_LIBRARY OR NOT DEFINED LEVEL_ZERO_INCLUDE_DIR)
  message(STATUS "Download Level Zero loader and headers from github.com")

  set(LEVEL_ZERO_LOADER_REPO "https://github.com/oneapi-src/level-zero.git")
  set(LEVEL_ZERO_LOADER_TAG v1.11.0)

  # Disable due to a bug https://github.com/oneapi-src/level-zero/issues/104
  set(CMAKE_INCLUDE_CURRENT_DIR OFF)

  message(STATUS "Will fetch Level Zero Loader from ${LEVEL_ZERO_LOADER_REPO}")
  include(FetchContent)
  FetchContent_Declare(level-zero-loader
    GIT_REPOSITORY    ${LEVEL_ZERO_LOADER_REPO}
    GIT_TAG           ${LEVEL_ZERO_LOADER_TAG}
  )

  # Workaround warnings/errors for Level Zero build
  set(CMAKE_CXX_FLAGS_BAK "${CMAKE_CXX_FLAGS}")
  if (WIN32)
    # FIXME: Level Zero build fails with /DUNICODE
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /UUNICODE")
    # USE_Z7 forces use of /Z7 instead of /Zi which is broken with sccache
    set(USE_Z7 ON)
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-truncation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat-extra-semi")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning-option")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-non-virtual-dtor")
  endif()

  FetchContent_MakeAvailable(level-zero-loader)
  FetchContent_GetProperties(level-zero-loader)

  # Restore original flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BAK}")

  set(LEVEL_ZERO_LIBRARY ze_loader)
  set(LEVEL_ZERO_INCLUDE_DIR
    ${level-zero-loader_SOURCE_DIR}/include CACHE PATH "Path to Level Zero Headers")
endif()

if (SYCL_ENABLE_XPTI_TRACING)
  set(XPTI_PROXY_SRC "${CMAKE_SOURCE_DIR}/../xpti/src/xpti_proxy.cpp")
  set(XPTI_INCLUDE "${CMAKE_SOURCE_DIR}/../xpti/include")
  set(XPTI_LIBS "${CMAKE_DL_LIBS}")
endif()

find_package(Python3 REQUIRED)

add_custom_target(ze-api DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ze_api.def)
add_custom_command(
  OUTPUT
  ${CMAKE_CURRENT_BINARY_DIR}/ze_api.def
  COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/ze_api_generator.py
  ${LEVEL_ZERO_INCLUDE_DIR}/ze_api.h
  DEPENDS
  ${LEVEL_ZERO_INCLUDE_DIR}/ze_api.h
  )

find_package(Threads REQUIRED)

add_sycl_plugin(level_zero
  SOURCES
    # These are short-term shared with Unified Runtime
    # The two plugins define a few things differrently so must
    # be built separately. This difference is spelled in
    # their "ur_bindings.hpp" files.
    #
    "ur_bindings.hpp"
    "../unified_runtime/pi2ur.hpp"
    "../unified_runtime/pi2ur.cpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/ur/ur.hpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/ur/ur.cpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/ur_level_zero.hpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/command_buffer.hpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/adapter.hpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/common.hpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/context.hpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/device.hpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/event.hpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/image.hpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/kernel.hpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/memory.hpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/platform.hpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/program.hpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/queue.hpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/sampler.hpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/usm.hpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/ur_level_zero.cpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/adapter.cpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/command_buffer.cpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/common.cpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/context.cpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/device.cpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/event.cpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/image.cpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/kernel.cpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/memory.cpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/platform.cpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/program.cpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/queue.cpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/sampler.cpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/usm.cpp"
    "${UNIFIED_RUNTIME_SOURCE_DIR}/source/adapters/level_zero/usm_p2p.cpp"
    # Following are the PI Level-Zero Plugin only codes.
    "pi_level_zero.cpp"
    "pi_level_zero.hpp"
    "tracing.cpp"
    ${XPTI_PROXY_SRC}
  INCLUDE_DIRS
    ${CMAKE_CURRENT_BINARY_DIR} # for ze_api.def
    ${CMAKE_CURRENT_SOURCE_DIR} # for Level-Zero Plugin "ur_bindings.hpp"
    ${CMAKE_CURRENT_SOURCE_DIR}/../unified_runtime # for Unified Runtime
    ${UNIFIED_RUNTIME_SOURCE_DIR}/source/ # for adapters/level_zero
    ${XPTI_INCLUDE}
  LIBRARIES
    LevelZeroLoader-Headers
    UnifiedRuntime-Headers
    UnifiedRuntimeCommon
    UnifiedMallocFramework
    LevelZeroLoader
    Threads::Threads
    ${XPTI_LIBS}
)

add_dependencies(pi_level_zero ze-api)

if (SYCL_ENABLE_XPTI_TRACING)
  target_compile_definitions(pi_level_zero PRIVATE
    XPTI_ENABLE_INSTRUMENTATION
    XPTI_STATIC_LIBRARY
  )
endif()
