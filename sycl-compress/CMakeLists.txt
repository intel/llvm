cmake_minimum_required(VERSION 3.20.0)

set(SYCL_COMPRESS_VERSION 0.0.7)
project (sycl-compress VERSION "${SYCL_COMPRESS_VERSION}" LANGUAGES CXX)

# Setting the same version as SYCL
set(CMAKE_CXX_STANDARD 17)

set(SYCL_COMPRESS_DIR ${CMAKE_CURRENT_LIST_DIR})

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# Download and build zstd
include(ExternalProject)
ExternalProject_Add(zstd
  URL https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-1.5.6.tar.gz
  URL_HASH SHA256=8c29e06cf42aacc1eafc4077ae2ec6c6fcb96a626157e0593d5e82a34fd403c1
  SOURCE_SUBDIR build/cmake
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/zstd/install -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DZSTD_BUILD_STATIC=ON -DZSTD_BUILD_SHARED=OFF 
)

set(ZSTD_LIBRARY ${CMAKE_CURRENT_BINARY_DIR}/zstd/install/lib/libzstd.a)
set(ZSTD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/zstd/install/include)

include_directories(${ZSTD_INCLUDE_DIR})

add_subdirectory(src)
add_dependencies(sycl-compress zstd)
target_link_libraries(sycl-compress ${ZSTD_LIBRARY})

if (LLVM_BINARY_DIR)
  file(GLOB_RECURSE SYCL_COMPRESS_HEADERS_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/sycl-compress/*")
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "${LLVM_BINARY_DIR}"
    SYCL_COMPRESS_HEADERS_OUT_LIST "${SYCL_COMPRESS_HEADERS_LIST}")
  add_custom_target(sycl-compress-headers
    DEPENDS ${SYCL_COMPRESS_HEADERS_OUT_LIST})

  add_custom_command(
    OUTPUT  ${SYCL_COMPRESS_HEADERS_OUT_LIST}
    DEPENDS ${SYCL_COMPRESS_HEADERS_LIST}
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_CURRENT_SOURCE_DIR}/include/sycl-compress
      ${LLVM_BINARY_DIR}/include/sycl-compress
    COMMENT "Copying sycl-compress headers ..."
  )
  add_dependencies(sycl-compress sycl-compress-headers zstd)
endif()

include(GNUInstallDirs)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/sycl-compress
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT sycl-compress
)