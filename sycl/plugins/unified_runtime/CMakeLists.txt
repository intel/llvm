# PI Unified Runtime plugin library
#
if (NOT DEFINED UNIFIED_RUNTIME_INCLUDE_DIR)
	message(STATUS "Download Unified Runtime headers from github.com")
  set(UNIFIED_RUNTIME_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/unified_runtime")
  if (NOT DEFINED SYCL_EP_UNIFIED_RUNTIME_SKIP_AUTO_UPDATE)
	  set(SYCL_EP_UNIFIED_RUNTIME_SKIP_AUTO_UPDATE ${SYCL_EXTERNAL_PROJECTS_SKIP_AUTO_UPDATE})
  endif()

  set(UNIFIED_RUNTIME_SOURCE_VERSION main)

  # TODO: Explore if FetchContent or git submodules are a better option.
  include(ExternalProject)
  ExternalProject_Add(unified-runtime
    GIT_REPOSITORY    https://github.com/oneapi-src/unified-runtime.git
    GIT_TAG           ${UNIFIED_RUNTIME_SOURCE_VERSION}
    UPDATE_DISCONNECTED ${SYCL_EP_UNIFIED_RUNTIME_SKIP_AUTO_UPDATE}
    SOURCE_DIR        ${UNIFIED_RUNTIME_SOURCE_DIR}
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
               -DCMAKE_INSTALL_LIBDIR:PATH=lib${LLVM_LIBDIR_SUFFIX}
               ${AUX_CMAKE_FLAGS}
               LOG_DOWNLOAD 1
               LOG_UPDATE 1
    CONFIGURE_COMMAND :
    BUILD_COMMAND :
    INSTALL_COMMAND :
  )
  set(UNIFIED_RUNTIME_INCLUDE_DIR ${UNIFIED_RUNTIME_SOURCE_DIR}/include/
	  CACHE INTERNAL "Path containing Unified Runtime header files.")

  ExternalProject_Add_Step(unified-runtime llvminstall
	  COMMAND ${CMAKE_COMMAND} -E copy_directory ${UNIFIED_RUNTIME_INCLUDE_DIR} ${LLVM_BINARY_DIR}/include/sycl
    COMMENT "Installing unified-runtime into the LLVM include directory"
    DEPENDEES install
  )
else()
  add_custom_target(unified-runtime
      DEPENDS
	  ${LLVM_BINARY_DIR}/include/sycl/unified_runtime
  )
  add_custom_command(
      OUTPUT
	  ${LLVM_BINARY_DIR}/include/sycl/unified_runtime
      COMMENT
      "Copying Unified Runtime headers"
      COMMAND
      ${CMAKE_COMMAND} -E copy_directory ${UNIFIED_RUNTIME_INCLUDE_DIR} ${LLVM_BINARY_DIR}/include/sycl/unified_runtime
      DEPENDS
      ${UNIFIED_RUNTIME_INCLUDE_DIR}
  )
endif()

list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS unified-runtime)

#include_directories("${LEVEL_ZERO_INCLUDE_DIR}")
include_directories("${UNIFIED_RUNTIME_INCLUDE_DIR}")

add_library (UnifiedRuntime-Headers INTERFACE)
add_library (UnifiedRuntime::Headers ALIAS UnifiedRuntime-Headers)
target_include_directories(UnifiedRuntime-Headers
	INTERFACE "${UNIFIED_RUNTIME_INCLUDE_DIR}"
)

find_package(Threads REQUIRED)

#
# NOTE: the Unified Runtime doesn't have the loader [yet].
# So what we really build is the Unified Runtime with Level Zero Adapter together.
#
add_sycl_plugin(unified_runtime
  SOURCES
    "${sycl_inc_dir}/sycl/detail/pi.h"
#    "${CMAKE_CURRENT_SOURCE_DIR}/pi2ur.cpp"
#    "${CMAKE_CURRENT_SOURCE_DIR}/pi2ur.hpp"
#    "${CMAKE_CURRENT_SOURCE_DIR}/ur.hpp"
#    "${CMAKE_CURRENT_SOURCE_DIR}/adapters/level_zero/ur_level_zero.hpp"
#    "${CMAKE_CURRENT_SOURCE_DIR}/adapters/level_zero/ur_level_zero.cpp"
#  LIBRARIES
#    "${LEVEL_ZERO_LOADER}"
#    Threads::Threads
)


SET_TARGET_PROPERTIES(pi_unified_runtime PROPERTIES LINKER_LANGUAGE C)
target_include_directories(pi_unified_runtime PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_dependencies(pi_unified_runtime unified-runtime)
