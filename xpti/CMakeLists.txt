cmake_minimum_required(VERSION 3.8)

set(XPTI_VERSION 0.4.1)
project (xpti VERSION "${XPTI_VERSION}" LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

set(XPTI_DIR ${CMAKE_CURRENT_LIST_DIR})
# Setting the same version as SYCL
set(CMAKE_CXX_STANDARD 17)

option(XPTI_ENABLE_WERROR OFF)

if (XPTI_ENABLE_WERROR)
  if(MSVC)
    set(CMAKE_CXX_FLAGS "/WX ${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "-Werror ${CMAKE_CXX_FLAGS}")
  endif()
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()

if (MSVC)
  # MSVC provides two incompatible build variants for its CRT: release and debug
  # To avoid potential issues in user code we also need to provide two kinds
  # of SYCL Runtime Library for release and debug configurations.
  set(XPTI_CXX_FLAGS "")
  if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(XPTI_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/MDd" "" XPTI_CXX_FLAGS "${XPTI_CXX_FLAGS}")
    string(REPLACE "/MTd" "" XPTI_CXX_FLAGS "${XPTI_CXX_FLAGS}")
  else()
    if (CMAKE_BUILD_TYPE MATCHES "Release")
      set(XPTI_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
    elseif (CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
      set(XPTI_CXX_FLAGS "${CMAKE_CXX_FLAGS_MINSIZEREL}")
    elseif (CMAKE_BUILD_TYPE MATCHES "MinSizeRel")
      set(XPTI_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    endif()
    string(REPLACE "/MD" "" XPTI_CXX_FLAGS "${XPTI_CXX_FLAGS}")
    string(REPLACE "/MT" "" XPTI_CXX_FLAGS "${XPTI_CXX_FLAGS}")
  endif()

  # target_compile_options requires list of options, not a string
  string(REPLACE " " ";" XPTI_CXX_FLAGS "${XPTI_CXX_FLAGS}")

  set(XPTI_CXX_FLAGS_RELEASE "${XPTI_CXX_FLAGS};/MD")
  set(XPTI_CXX_FLAGS_DEBUG "${XPTI_CXX_FLAGS};/MDd")

  # CMake automatically applies these flags to all targets. To override this
  # behavior, options lists are reset.
  set(CMAKE_CXX_FLAGS_RELEASE "")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "")
  set(CMAKE_CXX_FLAGS_DEBUG "")
endif()

if (SYCL_USE_LIBCXX)
  if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR
    (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    if ((NOT (DEFINED SYCL_LIBCXX_INCLUDE_PATH)) OR (NOT (DEFINED SYCL_LIBCXX_LIBRARY_PATH)))
      message(FATAL_ERROR "When building with libc++ SYCL_LIBCXX_INCLUDE_PATHS and"
                          "SYCL_LIBCXX_LIBRARY_PATH should be set")
    endif()
    set(CMAKE_CXX_FLAGS "-nostdinc++ -I ${SYCL_LIBCXX_INCLUDE_PATH} ${CMAKE_CXX_FLAGS}")
  else()
    message(FATAL_ERROR "Build with libc++ is not yet supported for this compiler")
  endif()
endif()
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_subdirectory(src)

if (LLVM_BINARY_DIR)
  file(GLOB_RECURSE XPTI_HEADERS_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/xpti/*")
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "${LLVM_BINARY_DIR}"
    XPTI_HEADERS_OUT_LIST "${XPTI_HEADERS_LIST}")
  add_custom_target(xpti-headers
    DEPENDS ${XPTI_HEADERS_OUT_LIST})

  add_custom_command(
    OUTPUT  ${XPTI_HEADERS_OUT_LIST}
    DEPENDS ${XPTI_HEADERS_LIST}
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_CURRENT_SOURCE_DIR}/include/xpti
      ${LLVM_BINARY_DIR}/include/xpti
    COMMENT "Copying XPTI headers ..."
  )
  add_dependencies(xpti xpti-headers)
endif()

include(GNUInstallDirs)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/xpti
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT xpti
)
