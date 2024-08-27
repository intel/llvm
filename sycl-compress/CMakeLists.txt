cmake_minimum_required(VERSION 3.20.0)

set(SYCL_COMPRESS_VERSION 0.0.7)
project (sycl-compress VERSION "${SYCL_COMPRESS_VERSION}" LANGUAGES CXX)

# Setting the same version as SYCL
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)

set(SYCL_COMPRESS_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# Download and build zstd statically.
set(ZSTD_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/zstd/install)

if (WIN32)
  set(ZSTD_LIBRARY_NAME zstd_static.lib)
else()
  set(ZSTD_LIBRARY_NAME libzstd.a)
endif()

include(ExternalProject)
ExternalProject_Add(zstd
  DEPENDS
  URL https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-1.5.6.tar.gz
  URL_HASH SHA256=8c29e06cf42aacc1eafc4077ae2ec6c6fcb96a626157e0593d5e82a34fd403c1
  SOURCE_SUBDIR build/cmake
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${ZSTD_INSTALL_DIR} -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DZSTD_BUILD_PROGRAMS=OFF -DZSTD_BUILD_STATIC=ON -DZSTD_BUILD_SHARED=OFF -DZSTD_MULTITHREAD_SUPPORT=OFF
  BUILD_BYPRODUCTS "${ZSTD_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/${ZSTD_LIBRARY_NAME}"
  DOWNLOAD_EXTRACT_TIMESTAMP ON
  )

set(ZSTD_LIBRARY ${ZSTD_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/${ZSTD_LIBRARY_NAME})
set(ZSTD_INCLUDE_DIR ${ZSTD_INSTALL_DIR}/include)

# Setup sycl-compress library. Add dependency on zstd.
set(SOURCES
  src/sycl-compress.cpp
  include/sycl-compress/sycl-compress.h
)

add_library(sycl-compress SHARED ${SOURCES})

add_dependencies(sycl-compress zstd)
target_compile_definitions(sycl-compress PRIVATE SYCL_COMPRESS_BUILD)
target_include_directories(sycl-compress PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include ${ZSTD_INCLUDE_DIR})
target_include_directories(sycl-compress PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(sycl-compress PRIVATE ${ZSTD_LIBRARY})

# Set the location of the library installation
include(GNUInstallDirs)
install(TARGETS sycl-compress
  LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT sycl-compress
  ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT sycl-compress
)

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
