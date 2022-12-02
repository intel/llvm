# PI Level Zero plugin library

if(MSVC)
  set(LEVEL_ZERO_LOADER
      "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ze_loader${CMAKE_STATIC_LIBRARY_SUFFIX}")
else()
  set(LEVEL_ZERO_LOADER
      "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}ze_loader${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

set(LEVEL_ZERO_LOADER_REPO "https://github.com/oneapi-src/level-zero.git")
set(LEVEL_ZERO_LOADER_TAG v1.8.8)

# Disable due to a bug https://github.com/oneapi-src/level-zero/issues/104
set(CMAKE_INCLUDE_CURRENT_DIR OFF)

message(STATUS "Will fetch Level Zero Loader from ${LEVEL_ZERO_LOADER_REPO}")
include(FetchContent)
FetchContent_Declare(level-zero-loader
  GIT_REPOSITORY    ${LEVEL_ZERO_LOADER_REPO}
  GIT_TAG           ${LEVEL_ZERO_LOADER_TAG}
)

FetchContent_MakeAvailable(level-zero-loader)
FetchContent_GetProperties(level-zero-loader)

set(LEVEL_ZERO_INCLUDE_DIR
  ${level-zero-loader_SOURCE_DIR}/include CACHE PATH "Path to Level Zero Headers")

add_library (LevelZeroLoader-Headers INTERFACE)
target_include_directories(LevelZeroLoader-Headers
  INTERFACE "${LEVEL_ZERO_INCLUDE_DIR}"
)

if (SYCL_ENABLE_XPTI_TRACING)
  set(XPTI_PROXY_SRC "${CMAKE_SOURCE_DIR}/../xpti/src/xpti_proxy.cpp")
  set(XPTI_INCLUDE "${CMAKE_SOURCE_DIR}/../xpti/include")
  set(XPIT_LIBS "${CMAKE_DL_LIBS}")
endif()

find_package(Python3 REQUIRED)

add_custom_target(ze-api
  COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/ze_api_generator.py
  ${LEVEL_ZERO_INCLUDE_DIR}/ze_api.h
  BYPRODUCTS
    ${CMAKE_CURRENT_BINARY_DIR}/ze_api.def
  )

find_package(Threads REQUIRED)

add_sycl_plugin(level_zero
  SOURCES
    "pi_level_zero.cpp"
    "pi_level_zero.hpp"
    "usm_allocator.cpp"
    "usm_allocator.hpp"
    "tracing.cpp"
    ${XPTI_PROXY_SRC}
  INCLUDE_DIRS
    ${CMAKE_CURRENT_BINARY_DIR} # for ze_api.def
    ${XPTI_INCLUDE}
  LIBRARIES
    LevelZeroLoader-Headers
    UnifiedRuntime-Headers
    ze_loader
    Threads::Threads
    unified_runtime_static
    ${XPTI_LIBS}
)

add_dependencies(pi_level_zero ze-api)

if (SYCL_ENABLE_XPTI_TRACING)
  target_compile_definitions(pi_level_zero PRIVATE
    XPTI_ENABLE_INSTRUMENTATION
    XPTI_STATIC_LIBRARY
  )
endif()
