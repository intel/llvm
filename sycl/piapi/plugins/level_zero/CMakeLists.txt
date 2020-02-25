# PI Level Zero plugin library

set(LEVEL_ZERO_LOADER
  "${pi_library_output_dir}/${CMAKE_STATIC_LIBRARY_PREFIX}ze_loader")
if(MSVC)
  set(LEVEL_ZERO_LOADER "${LEVEL_ZERO_LOADER}${CMAKE_STATIC_LIBRARY_SUFFIX}")
else()
  set(LEVEL_ZERO_LOADER "${LEVEL_ZERO_LOADER}${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

if (NOT DEFINED LEVEL_ZERO_LIBRARY OR NOT DEFINED LEVEL_ZERO_INCLUDE_DIR)
  message(STATUS "Download Level Zero loader and headers from github.com")
  if (CMAKE_C_COMPILER)
      list(APPEND AUX_CMAKE_FLAGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
  endif()
  if (CMAKE_CXX_COMPILER)
      list(APPEND AUX_CMAKE_FLAGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
  endif()

  set(level_zero_loader_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/level_zero_loader_build")
  file(MAKE_DIRECTORY "${level_zero_loader_binary_dir}")

  set(level_zero_loader_install_dir "${pi_library_base_dir}")

  set(LEVEL_ZERO_LOADER_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/Level0/level_zero_loader")
  if (NOT DEFINED SYCL_EP_LEVEL_ZERO_LOADER_SKIP_AUTO_UPDATE)
    set(SYCL_EP_LEVEL_ZERO_LOADER_SKIP_AUTO_UPDATE ${SYCL_EXTERNAL_PROJECTS_SKIP_AUTO_UPDATE})
  endif()

  ExternalProject_Add(level-zero-loader
    GIT_REPOSITORY    https://github.com/oneapi-src/level-zero.git
    GIT_TAG           v1.0
    UPDATE_DISCONNECTED ${SYCL_EP_LEVEL_ZERO_LOADER_SKIP_AUTO_UPDATE}
    SOURCE_DIR        ${LEVEL_ZERO_LOADER_SOURCE_DIR}
    BINARY_DIR        "${level_zero_loader_binary_dir}"
    INSTALL_DIR       "${level_zero_loader_install_dir}"
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
               -DOpenCL_INCLUDE_DIR=${OpenCL_INCLUDE_DIRS}
               -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
               -DCMAKE_INSTALL_LIBDIR:PATH=lib${PI_LIBDIR_SUFFIX}
               LOG_DOWNLOAD 1
               LOG_UPDATE 1
               LOG_CONFIGURE 1
               LOG_BUILD 1
               LOG_INSTALL 1
               ${AUX_CMAKE_FLAGS}
    STEP_TARGETS      configure,build,install
    BUILD_BYPRODUCTS ${LEVEL_ZERO_LOADER}
  )
  set(LEVEL_ZERO_INCLUDE_DIR "${level_zero_loader_install_dir}/include")
  file(MAKE_DIRECTORY "${LEVEL_ZERO_INCLUDE_DIR}")
else()
  include_directories("${LEVEL_ZERO_INCLUDE_DIR}")
  file(GLOB LEVEL_ZERO_LIBRARY_SRC "${LEVEL_ZERO_LIBRARY}*")
  file(COPY ${LEVEL_ZERO_LIBRARY_SRC} DESTINATION ${pi_library_output_dir})
  add_custom_target(level-zero-loader DEPENDS ${LEVEL_ZERO_LIBRARY} COMMENT "Copying Level Zero Loader ...")
endif()

add_library(level_zero_lib INTERFACE IMPORTED GLOBAL)
set_target_properties(level_zero_lib PROPERTIES
  INTERFACE_LINK_LIBRARIES ${LEVEL_ZERO_LOADER}
  INTERFACE_INCLUDE_DIRECTORIES "${LEVEL_ZERO_INCLUDE_DIR}"
)
add_dependencies(level_zero_lib level-zero-loader)

add_library(pi_level_zero SHARED
  "${pi_include_dir}/pi/pi.h"
  "pi_level_zero.cpp"
  "pi_level_zero.hpp"
  "usm_allocator.cpp"
  "usm_allocator.hpp"
)

set_target_properties(pi_level_zero PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${pi_library_output_dir}"
  LIBRARY_OUTPUT_DIRECTORY "${pi_library_output_dir}"
  RUNTIME_OUTPUT_DIRECTORY "${pi_binary_output_dir}"
)

target_link_libraries(pi_level_zero PUBLIC piapi)
target_link_libraries(pi_level_zero PRIVATE
  level_zero_lib
  pi_export_library
)
if (UNIX)
  target_link_libraries(pi_level_zero PRIVATE pthread)
endif()

################################################################################
## install

install(TARGETS pi_level_zero
    EXPORT piapiTargets
    LIBRARY DESTINATION "lib${PI_LIBDIR_SUFFIX}" COMPONENT pi_level_zero
    RUNTIME DESTINATION "bin" COMPONENT pi_level_zero)
export(TARGETS pi_level_zero APPEND
  FILE "${PROJECT_BINARY_DIR}/piapiTargets.cmake"
)
install(FILES "pi_level_zero.hpp" DESTINATION include/pi COMPONENT headers)
install(DIRECTORY "${LEVEL_ZERO_INCLUDE_DIR}/"
  DESTINATION "include"
  COMPONENT headers
)
install(DIRECTORY "${pi_library_output_dir}"
  DESTINATION .
  COMPONENT pi_level_zero
  FILES_MATCHING PATTERN "${CMAKE_STATIC_LIBRARY_PREFIX}ze*"
)
