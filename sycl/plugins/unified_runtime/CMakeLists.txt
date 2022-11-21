# PI Unified Runtime plugin library
#

include(FetchContent)

# TODO: Replace personal repo and tag sha with upstream as soon as
#       https://github.com/oneapi-src/unified-runtime/pull/29 is merged

#set(UNIFIED_RUNTIME_REPO "https://github.com/oneapi-src/unified-runtime.git")
set(UNIFIED_RUNTIME_REPO "https://github.com/pvchupin/unified-runtime.git")
set(UNIFIED_RUNTIME_TAG bfb0c8f01deb2517057193882d506038831b4a43)

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


#include_directories("${LEVEL_ZERO_INCLUDE_DIR}")
include_directories("${UNIFIED_RUNTIME_INCLUDE_DIR}")

add_library (UnifiedRuntime-Headers INTERFACE)
target_include_directories(UnifiedRuntime-Headers
  INTERFACE "${UNIFIED_RUNTIME_INCLUDE_DIR}"
)

list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS unified-runtime)

find_package(Threads REQUIRED)

#
# NOTE: the Unified Runtime doesn't have the loader [yet].
# So what we really build is the Unified Runtime with Level Zero Adapter
# together.
#
# TODO: begin
#       Unified Runtime today is not a plugin yet. It's experimental static
#       library that's linked into level_zero plugin. As soon as it's ready to
#       become a plugin code below should be replaced with the following:
#add_sycl_plugin(unified_runtime
#  SOURCES
#    "${sycl_inc_dir}/sycl/detail/pi.h"
#    "${CMAKE_CURRENT_SOURCE_DIR}/pi2ur.cpp"
#    "${CMAKE_CURRENT_SOURCE_DIR}/pi2ur.hpp"
#    "${CMAKE_CURRENT_SOURCE_DIR}/ur.hpp"
#    "${CMAKE_CURRENT_SOURCE_DIR}/adapters/level_zero/ur_level_zero.hpp"
#    "${CMAKE_CURRENT_SOURCE_DIR}/adapters/level_zero/ur_level_zero.cpp"
#   LIBRARIES
#    Threads::Threads
#    UnifiedRuntime-Headers
#    LevelZeroLoader-Headers
#)

add_library(unified_runtime STATIC
  "${sycl_inc_dir}/sycl/detail/pi.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/ur.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/adapters/level_zero/ur_level_zero.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/adapters/level_zero/ur_level_zero.cpp"
)

target_include_directories(unified_runtime INTERFACE
  "${UNIFIED_RUNTIME_INCLUDE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(unified_runtime PRIVATE
  UnifiedRuntime-Headers
  LevelZeroLoader-Headers
)

# TODO: end

add_dependencies(unified_runtime ze-api)

