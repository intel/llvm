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

# Get the L0 adapter sources so they can be shared with the L0 PI plugin
get_target_property(UR_L0_ADAPTER_SOURCES ur_adapter_level_zero SOURCES)

add_sycl_plugin(level_zero
  SOURCES
    # These are short-term shared with Unified Runtime
    # The two plugins define a few things differrently so must
    # be built separately. This difference is spelled in
    # their "ur_bindings.hpp" files.
    #
    "ur_bindings.hpp"
    "../unified_runtime/pi2ur.hpp"
    ${UR_L0_ADAPTER_SOURCES}
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

if (WIN32)
  # 0x800: Search for the DLL only in the System32 folder
  target_link_options(pi_level_zero PUBLIC /DEPENDENTLOADFLAG:0x800)
  target_compile_definitions(pi_level_zero PRIVATE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
  )
endif()

add_dependencies(pi_level_zero ze-api)

if (SYCL_ENABLE_XPTI_TRACING)
  target_compile_definitions(pi_level_zero PRIVATE
    XPTI_ENABLE_INSTRUMENTATION
    XPTI_STATIC_LIBRARY
  )
endif()
