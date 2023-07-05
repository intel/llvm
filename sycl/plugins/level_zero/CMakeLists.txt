# PI Level Zero plugin library

if (NOT DEFINED LEVEL_ZERO_LIBRARY OR NOT DEFINED LEVEL_ZERO_INCLUDE_DIR)
  message(STATUS "Download Level Zero loader and headers from github.com")

  set(LEVEL_ZERO_LOADER_REPO "https://github.com/oneapi-src/level-zero.git")
  set(LEVEL_ZERO_LOADER_TAG v1.11.0)

  # Disable due to a bug https://github.com/oneapi-src/level-zero/issues/104
  set(CMAKE_INCLUDE_CURRENT_DIR OFF)
  # Prevent L0 loader from exporting extra symbols
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)

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

add_library (LevelZeroLoader INTERFACE)
# The MSVC linker does not like / at the start of a path, so to work around this
# we split it into a link library and a library path, where the path is allowed
# to have leading /.
get_filename_component(LEVEL_ZERO_LIBRARY_SRC "${LEVEL_ZERO_LIBRARY}" DIRECTORY)
get_filename_component(LEVEL_ZERO_LIB_NAME "${LEVEL_ZERO_LIBRARY}" NAME)
target_link_directories(LevelZeroLoader
  INTERFACE "${LEVEL_ZERO_LIBRARY_SRC}"
)
target_link_libraries(LevelZeroLoader
  INTERFACE "${LEVEL_ZERO_LIB_NAME}"
)

add_library (LevelZeroLoader-Headers INTERFACE)
target_include_directories(LevelZeroLoader-Headers
  INTERFACE "${LEVEL_ZERO_INCLUDE_DIR}"
)

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
    "../unified_runtime/ur/ur.hpp"
    "../unified_runtime/ur/ur.cpp"
    "../unified_runtime/ur/usm_allocator.cpp"
    "../unified_runtime/ur/usm_allocator.hpp"
    "../unified_runtime/ur/usm_allocator_config.cpp"
    "../unified_runtime/ur/usm_allocator_config.hpp"
    "../unified_runtime/ur/adapters/level_zero/ur_level_zero.hpp"
    "../unified_runtime/ur/adapters/level_zero/command_buffer.hpp"
    "../unified_runtime/ur/adapters/level_zero/common.hpp"
    "../unified_runtime/ur/adapters/level_zero/context.hpp"
    "../unified_runtime/ur/adapters/level_zero/device.hpp"
    "../unified_runtime/ur/adapters/level_zero/event.hpp"
    "../unified_runtime/ur/adapters/level_zero/memory.hpp"
    "../unified_runtime/ur/adapters/level_zero/kernel.hpp"
    "../unified_runtime/ur/adapters/level_zero/platform.hpp"
    "../unified_runtime/ur/adapters/level_zero/program.hpp"
    "../unified_runtime/ur/adapters/level_zero/queue.hpp"
    "../unified_runtime/ur/adapters/level_zero/sampler.hpp"
    "../unified_runtime/ur/adapters/level_zero/usm.hpp"
    "../unified_runtime/ur/adapters/level_zero/ur_level_zero.cpp"
    "../unified_runtime/ur/adapters/level_zero/command_buffer.cpp"
    "../unified_runtime/ur/adapters/level_zero/common.cpp"
    "../unified_runtime/ur/adapters/level_zero/context.cpp"
    "../unified_runtime/ur/adapters/level_zero/device.cpp"
    "../unified_runtime/ur/adapters/level_zero/event.cpp"
    "../unified_runtime/ur/adapters/level_zero/memory.cpp"
    "../unified_runtime/ur/adapters/level_zero/kernel.cpp"
    "../unified_runtime/ur/adapters/level_zero/platform.cpp"
    "../unified_runtime/ur/adapters/level_zero/program.cpp"
    "../unified_runtime/ur/adapters/level_zero/queue.cpp"
    "../unified_runtime/ur/adapters/level_zero/sampler.cpp"
    "../unified_runtime/ur/adapters/level_zero/usm.cpp"
    "../unified_runtime/ur/adapters/level_zero/usm_p2p.cpp"
    # Following are the PI Level-Zero Plugin only codes.
    "pi_level_zero.cpp"
    "pi_level_zero.hpp"
    "tracing.cpp"
    ${XPTI_PROXY_SRC}
  INCLUDE_DIRS
    ${CMAKE_CURRENT_BINARY_DIR} # for ze_api.def
    ${CMAKE_CURRENT_SOURCE_DIR} # for Level-Zero Plugin "ur_bindings.hpp"
    ${CMAKE_CURRENT_SOURCE_DIR}/../unified_runtime # for Unified Runtime
    ${XPTI_INCLUDE}
  LIBRARIES
    LevelZeroLoader-Headers
    UnifiedRuntime-Headers
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
