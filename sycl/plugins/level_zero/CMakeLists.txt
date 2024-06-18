# PI Level Zero plugin library

if (SYCL_ENABLE_XPTI_TRACING)
  set(XPTI_PROXY_SRC "${CMAKE_SOURCE_DIR}/../xpti/src/xpti_proxy.cpp")
  set(XPTI_INCLUDE "${CMAKE_SOURCE_DIR}/../xpti/include")
  set(XPTI_LIBS "${CMAKE_DL_LIBS}")
endif()

find_package(Python3 REQUIRED)

get_target_property(LEVEL_ZERO_INCLUDE_DIR LevelZeroLoader-Headers INTERFACE_INCLUDE_DIRECTORIES)

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
    UnifiedMemoryFramework
    LevelZeroLoader
    Threads::Threads
    ${XPTI_LIBS}
)

if (WIN32)
  # 0x800: Search for the DLL only in the System32 folder
  target_link_options(pi_level_zero PUBLIC /DEPENDENTLOADFLAG:0x800)
endif()

add_dependencies(pi_level_zero ze-api)

if (SYCL_ENABLE_XPTI_TRACING)
  target_compile_definitions(pi_level_zero PRIVATE
    XPTI_ENABLE_INSTRUMENTATION
    XPTI_STATIC_LIBRARY
  )
endif()
