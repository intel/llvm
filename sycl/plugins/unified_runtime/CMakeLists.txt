# PI Unified Runtime plugin library
#
if (NOT DEFINED UNIFIED_RUNTIME_LIBRARY OR NOT DEFINED UNIFIED_RUNTIME_INCLUDE_DIR)
  include(FetchContent)

  set(UNIFIED_RUNTIME_REPO "https://github.com/oneapi-src/unified-runtime.git")
  set(UNIFIED_RUNTIME_TAG 61cb864b3a73918124bff11f9ad58dafb8c7769a)

  message(STATUS "Will fetch Unified Runtime from ${UNIFIED_RUNTIME_REPO}")
  FetchContent_Declare(unified-runtime
    GIT_REPOSITORY    ${UNIFIED_RUNTIME_REPO}
    GIT_TAG           ${UNIFIED_RUNTIME_TAG}
  )

  # Disable some compilation options to avoid errors while building the UR
  if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-truncation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
  endif()

  FetchContent_GetProperties(unified-runtime)
  FetchContent_MakeAvailable(unified-runtime)

  add_library(${PROJECT_NAME}::ur_loader ALIAS loader)

  set(UNIFIED_RUNTIME_LIBRARY ${PROJECT_NAME}::ur_loader)

  set(UNIFIED_RUNTIME_SOURCE_DIR
    ${unified-runtime_SOURCE_DIR} CACHE PATH "Path to Unified Runtime Headers")
  set(UNIFIED_RUNTIME_LOADER_SOURCE_DIR
    ${unified-runtime_SOURCE_DIR}/source/loader CACHE PATH "Path to Unified Runtime Loader Headers")
  set(UNIFIED_RUNTIME_INCLUDE_DIR "${UNIFIED_RUNTIME_SOURCE_DIR}/include")
endif()

add_library (UnifiedRuntimeLoader INTERFACE)
get_filename_component(UNIFIED_RUNTIME_LIBRARY_SRC "${UNIFIED_RUNTIME_LOADER_SOURCE_DIR}" DIRECTORY)
get_filename_component(UNIFIED_RUNTIME_LIB_NAME "${UNIFIED_RUNTIME_LIBRARY}" NAME)

message(STATUS "UNIFIED_RUNTIME_LIB_NAME ${UNIFIED_RUNTIME_LIB_NAME}")

add_library (UnifiedRuntime-Headers INTERFACE)
add_library (UnifiedRuntimeLoader-Headers INTERFACE)

target_include_directories(UnifiedRuntime-Headers
  INTERFACE
    "${UNIFIED_RUNTIME_INCLUDE_DIR}"
)

target_include_directories(UnifiedRuntimeLoader-Headers
  INTERFACE "${UNIFIED_RUNTIME_INCLUDE_DIR}"
)

target_include_directories(UnifiedRuntimeLoader
  INTERFACE "${UNIFIED_RUNTIME_INCLUDE_DIR}"
)

target_link_libraries(UnifiedRuntimeLoader
  INTERFACE "${UNIFIED_RUNTIME_LIB_NAME}"
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
    "ur/adapters/level_zero/ur_level_zero.cpp"
    "ur/usm_allocator.hpp"
    "ur/usm_allocator.cpp"
    # These below belong to Unified Runtime PI Plugin only
    "pi_unified_runtime.hpp"
    "pi_unified_runtime.cpp"
  LIBRARIES
    Threads::Threads
    UnifiedRuntime-Headers
    UnifiedRuntimeLoader-Headers
    UnifiedRuntimeLoader
    LevelZeroLoader-Headers
    LevelZeroLoader
)

if (TARGET ${PROJECT_NAME}::ur_loader)
  set_target_properties(hello_world PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
  set_target_properties(ur_null PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
  install(TARGETS loader
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT unified-runtime-sycl-dev
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT unified-runtime-sycl-dev
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT unified-runtime-sycl-dev
  )
  file(GLOB UNIFIED_RUNTIME_API_HEADERS "${UNIFIED_RUNTIME_INCLUDE_DIR}/*.h")
  install(FILES ${UNIFIED_RUNTIME_API_HEADERS}
      DESTINATION ${CMAKE_INSTALL_PREFIX}/include/sycl/unified_runtime/
      COMPONENT unified-runtime-sycl-dev
  )
endif()

