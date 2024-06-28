# boost/mp11 headers import and preprocessing
# See more comments in cmake/modules/PreprocessBoostMp11Headers.cmake

include(FetchContent)

set(BOOST_MP11_GIT_REPO https://github.com/boostorg/mp11.git)
# Author: pdimov
# Date: Dec 31, 2023
# Release: boost-1.85.0
set(BOOST_MP11_GIT_TAG 863d8b8d2b20f2acd0b5870f23e553df9ce90e6c)

# Either download from github or use existing if BOOST_MP11_SOURCE_DIR is set
if (NOT DEFINED BOOST_MP11_SOURCE_DIR)
  message(STATUS "BOOST_MP11_SOURCE_DIR not set, downloading boost/mp11 headers from ${BOOST_MP11_GIT_REPO}")

  FetchContent_Declare(boost_mp11
    GIT_REPOSITORY ${BOOST_MP11_GIT_REPO}
    GIT_TAG ${BOOST_MP11_GIT_TAG}
  )
  FetchContent_GetProperties(boost_mp11)
  FetchContent_MakeAvailable(boost_mp11)

  set(BOOST_MP11_SOURCE_DIR ${boost_mp11_SOURCE_DIR})
  set(BOOST_MP11_SRC_PATH ${BOOST_MP11_GIT_REPO})
  set(BOOST_MP11_SRC_ID "git commit hash: ${BOOST_MP11_GIT_TAG}")
else (NOT DEFINED BOOST_MP11_SOURCE_DIR)
  message(STATUS "Using boost/mp11 headers from ${BOOST_MP11_SOURCE_DIR}")
  set(BOOST_MP11_SRC_PATH ${BOOST_MP11_SOURCE_DIR})
  set(BOOST_MP11_SRC_ID "ID not set")
endif(NOT DEFINED BOOST_MP11_SOURCE_DIR)

# Read all header file names into HEADERS_BOOST_MP11
file(GLOB_RECURSE HEADERS_BOOST_MP11 CONFIGURE_DEPENDS "${BOOST_MP11_SOURCE_DIR}/include/boost/*")

set(BOOST_MP11_DESTINATION_DIR ${SYCL_INCLUDE_BUILD_DIR}/sycl/detail/boost)
string(REPLACE "${BOOST_MP11_SOURCE_DIR}/include/boost" "${BOOST_MP11_DESTINATION_DIR}"
  OUT_HEADERS_BOOST_MP11 "${HEADERS_BOOST_MP11}")

# The target which produces preprocessed boost/mp11 headers
add_custom_target(boost_mp11-headers
  DEPENDS ${OUT_HEADERS_BOOST_MP11})

# Run preprocessing on each header, output result into
# ${BOOST_MP11_DESTINATION_DIR}
add_custom_command(
  OUTPUT ${OUT_HEADERS_BOOST_MP11}
  DEPENDS ${HEADERS_BOOST_MP11} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/PreprocessBoostMp11Headers.cmake
  COMMAND ${CMAKE_COMMAND}
    -DIN=${BOOST_MP11_SOURCE_DIR}/include/boost
    -DOUT=${BOOST_MP11_DESTINATION_DIR}
    -DSRC_PATH="${BOOST_MP11_SRC_PATH}"
    -DSRC_ID="${BOOST_MP11_SRC_ID}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/PreprocessBoostMp11Headers.cmake
  COMMENT "Preprocessing boost/mp11 headers ${BOOST_MP11_SOURCE_DIR}/include/boost -> ${BOOST_MP11_DESTINATION_DIR}...")
