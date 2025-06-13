# Finds or fetches emhash.
if(DEFINED SYCL_EMHASH_DIR OR DEFINED EMHASH_SYS_LOC)
  return()
endif()
find_file(EMHASH_SYS_LOC "hash_table8.hpp" PATH_SUFFIXES "emhash")
if(NOT EMHASH_SYS_LOC)
  set(EMHASH_REPO https://github.com/ktprime/emhash)
  message(STATUS "Will fetch emhash from ${EMHASH_REPO}")
  FetchContent_Declare(emhash
    GIT_REPOSITORY    ${EMHASH_REPO}
    GIT_TAG           3ba9abdfdc2e0430fcc2fd8993cad31945b6a02b
    SOURCE_SUBDIR  emhash
  )
  FetchContent_MakeAvailable(emhash)

  # FetchContent downloads the files into a directory with
  # '-src' as the suffix and emhash has the headers in the
  # top level directory in the repo, so copy the headers to a directory
  # named `emhash` so source files can include with <emhash/header.hpp>
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/emhash)
  file(GLOB HEADERS "${emhash_SOURCE_DIR}/*.h*")
  file(COPY ${HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/include/emhash)
  set(SYCL_EMHASH_DIR ${CMAKE_BINARY_DIR}/include/ CACHE INTERNAL "")
endif()
