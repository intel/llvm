# PI Level Zero plugin library

if(MSVC)
  set(LEVEL_ZERO_LOADER
      "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ze_loader${CMAKE_STATIC_LIBRARY_SUFFIX}")
else()
  set(LEVEL_ZERO_LOADER
      "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}ze_loader${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

set(LEVEL_ZERO_LOADER_REPO "https://github.com/oneapi-src/level-zero.git")
if (WIN32)
  set(LEVEL_ZERO_LOADER_TAG v1.2.3)
else()
  set(LEVEL_ZERO_LOADER_TAG v1.8.8)
endif()

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
set(LEVEL_ZERO_LOADER_LIBRARY ${level-zero-loader_TARGET_LOADER_NAME})

#ExternalProject_Add_Step(level-zero-loader llvminstall
#COMMAND ${CMAKE_COMMAND} -E copy_directory <INSTALL_DIR>/lib/ ${LLVM_BINARY_DIR}/lib
#COMMAND ${CMAKE_COMMAND} -E copy_directory <INSTALL_DIR>/include/ ${LLVM_BINARY_DIR}/include/sycl
#COMMENT "Installing level-zero-loader into the LLVM binary directory"
#DEPENDEES install
#)

list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS level-zero-loader level-zero-headers)

include_directories("${LEVEL_ZERO_INCLUDE_DIR}")

add_library (LevelZeroLoader INTERFACE)
target_link_libraries(LevelZeroLoader
  INTERFACE ${LEVEL_ZERO_LOADER_LIBRARY}
)

add_library (LevelZeroLoader-Headers INTERFACE)
add_library (LevelZeroLoader::Headers ALIAS LevelZeroLoader-Headers)
target_include_directories(LevelZeroLoader-Headers
  INTERFACE "${LEVEL_ZERO_INCLUDE_DIR}"
)

if (SYCL_ENABLE_XPTI_TRACING)
  set(XPTI_PROXY_SRC "${CMAKE_SOURCE_DIR}/../xpti/src/xpti_proxy.cpp")
endif()

find_package(Threads REQUIRED)

add_sycl_plugin(level_zero
  SOURCES
    "${sycl_inc_dir}/sycl/detail/pi.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/pi_level_zero.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pi_level_zero.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/usm_allocator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/usm_allocator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tracing.cpp"
    ${XPTI_PROXY_SRC}
  LIBRARIES
    ${LEVEL_ZERO_LOADER_LIBRARY}
    Threads::Threads
    pi_unified_runtime
)

find_package(Python3 REQUIRED)

add_custom_target(ze-api
  COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/ze_api_generator.py
  ${LEVEL_ZERO_INCLUDE_DIR}/ze_api.h
  BYPRODUCTS
    ${CMAKE_CURRENT_BINARY_DIR}/ze_api.def
  )
target_include_directories(pi_level_zero PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(pi_level_zero ze-api)

if (SYCL_ENABLE_XPTI_TRACING)
  target_compile_definitions(pi_level_zero PRIVATE
    XPTI_ENABLE_INSTRUMENTATION
    XPTI_STATIC_LIBRARY
  )
  target_include_directories(pi_level_zero PRIVATE "${CMAKE_SOURCE_DIR}/../xpti/include")
  target_link_libraries(pi_level_zero PRIVATE ${CMAKE_DL_LIBS})
endif()

if (TARGET level-zero-loader)
  add_dependencies(ze-api level-zero-loader)
  add_dependencies(pi_level_zero level-zero-loader)
endif()
