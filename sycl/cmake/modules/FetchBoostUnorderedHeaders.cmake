# Fetches the unordered boost module and its dependencies
function(add_boost_module_headers)
  cmake_parse_arguments(
    BOOST_MODULE               # prefix
    ""                         # options
    "NAME;SRC_DIR;GIT_TAG;"    # one value keywords
    ""                         # multi-value keywords
    ${ARGN})                   # arguments

  if (NOT DEFINED BOOST_MODULE_SRC_DIR)
    set(BOOST_MODULE_GIT_REPO "https://github.com/boostorg/${BOOST_MODULE_NAME}.git")
    message(STATUS "Source dir not set for boost module ${BOOST_MODULE_NAME}, downloading headers from ${BOOST_MODULE_GIT_REPO}")

    set(BOOST_MODULE_FULL_NAME "boost_${BOOST_MODULE_NAME}")
    FetchContent_Declare(${BOOST_MODULE_FULL_NAME}
      GIT_REPOSITORY ${BOOST_MODULE_GIT_REPO}
      GIT_TAG ${BOOST_MODULE_GIT_TAG}
    )
    FetchContent_GetProperties(${BOOST_MODULE_FULL_NAME})
    FetchContent_MakeAvailable(${BOOST_MODULE_FULL_NAME})

    set(BOOST_MODULE_SRC_DIR ${${BOOST_MODULE_FULL_NAME}_SOURCE_DIR})
  else (NOT DEFINED BOOST_MODULE_SRC_DIR)
    message(STATUS "Using boost/${BOOST_MODULE_NAME} headers from ${BOOST_MODULE_SRC_DIR}")
  endif(NOT DEFINED BOOST_MODULE_SRC_DIR)

  set(BOOST_UNORDERED_INCLUDE_DIRS ${BOOST_UNORDERED_INCLUDE_DIRS} "${BOOST_MODULE_SRC_DIR}/include" PARENT_SCOPE)
endfunction(add_boost_module_headers)

set(BOOST_UNORDERED_GIT_TAG bd24dfd284dbc70e7521915af0d8d049f74a1e85)
# Author: joaquintides <joaquin@tid.es>
# Date:   Tue Jul 18 18:19:13 2023 +0200
#
#     updated concurrent map benchmark plots
add_boost_module_headers(NAME "unordered" SRC_DIR ${BOOST_UNORDERED_SOURCE_DIR} GIT_TAG ${BOOST_UNORDERED_GIT_TAG})

set(BOOST_ASSERT_GIT_TAG 02256c84fd0cd58a139d9dc1b25b5019ca976ada)
# Author: Peter Dimov <pdimov@gmail.com>
# Date:   Thu Jun 22 18:11:58 2023 +0300
#
#     Do not use std::source_location::current under nvcc. Fixes #32.
add_boost_module_headers(NAME "assert" SRC_DIR ${BOOST_ASSERT_SOURCE_DIR} GIT_TAG ${BOOST_ASSERT_GIT_TAG})

set(BOOST_CONFIG_GIT_TAG a1cf5d531405e62927b0257b5cbecc66a545b508)
# Merge: f5726a26 a1edcd56
# Author: jzmaddock <john@johnmaddock.co.uk>
# Date:   Sat Apr 15 13:20:12 2023 +0100
#
#     Merge pull request #475 from boostorg/ci_2023_04
add_boost_module_headers(NAME "config" SRC_DIR ${BOOST_CONFIG_SOURCE_DIR} GIT_TAG ${BOOST_CONFIG_GIT_TAG})

set(BOOST_CONTAINER_HASH_GIT_TAG 226eb066e949adbf37b220e993d64ecefeeaae99)
# Author: Peter Dimov <pdimov@gmail.com>
# Date:   Thu Jun 29 14:38:53 2023 +0300
#
#     Update .drone.jsonnet
add_boost_module_headers(NAME "container_hash" SRC_DIR ${BOOST_CONTAINER_HASH_SOURCE_DIR} GIT_TAG ${BOOST_CONTAINER_HASH_GIT_TAG})

set(BOOST_CORE_GIT_TAG 216999e552e7f73e63c7bcc88b8ce9c179bbdbe2)
# Author: Peter Dimov <pdimov@gmail.com>
# Date:   Sun Jun 25 13:46:53 2023 +0300
#
#     Avoid -Wsign-conversion warning in checked_delete.hpp
add_boost_module_headers(NAME "core" SRC_DIR ${BOOST_CORE_SOURCE_DIR} GIT_TAG ${BOOST_CORE_GIT_TAG})

# Describe is a dependency of container_hash
set(BOOST_DESCRIBE_GIT_TAG a0eafb08100eb15a57b6dae6d270c0012a56aa21)
# Merge: 1692c3e b54fda5
# Author: Peter Dimov <pdimov@gmail.com>
# Date:   Sun May 21 04:51:35 2023 +0300
#
#     Merge branch 'fix-deprecated-inline-static-variables' of https://github.com/Romain-Geissler-1A/describe into feature/pr-40
add_boost_module_headers(NAME "describe" SRC_DIR ${BOOST_DESCRIBE_SOURCE_DIR} GIT_TAG ${BOOST_DESCRIBE_GIT_TAG})

set(BOOST_MOVE_GIT_TAG f1fbb45134065deebe95249c616a967d4b66c809)
# Author: Ion Gaztañaga <igaztanaga@gmail.com>
# Date:   Mon Mar 13 13:32:29 2023 +0100
#
#     Use [[msvc::intrinsic] attribute if available in move/forward in order to improve debug experience
add_boost_module_headers(NAME "move" SRC_DIR ${BOOST_MOVE_SOURCE_DIR} GIT_TAG ${BOOST_MOVE_GIT_TAG})

# Reuse mp11 fetched earlier for DPC++ headers
set(BOOST_UNORDERED_INCLUDE_DIRS ${BOOST_UNORDERED_INCLUDE_DIRS} "${BOOST_MP11_SOURCE_DIR}/include/")

set(BOOST_PREDEF_GIT_TAG 392e4e767469e3469c9390f0d9cca16724dc3fc8)
# Merge: a12c7fd 499d28e
# Author: Rene Rivera <grafikrobot@gmail.com>
# Date:   Sun Feb 27 14:44:35 2022 -0600
#
#     Release 1.14.
add_boost_module_headers(NAME "predef" SRC_DIR ${BOOST_PREDEF_SOURCE_DIR} GIT_TAG ${BOOST_PREDEF_GIT_TAG})

set(BOOST_PREPROCESSOR_GIT_TAG 667e87b3392db338a919cbe0213979713aca52e3)
# Author: Peter Dimov <pdimov@gmail.com>
# Date:   Tue Aug 16 20:59:52 2022 +0300
#
#     Change C test names to not conflict with the C++ ones
add_boost_module_headers(NAME "preprocessor" SRC_DIR ${BOOST_PREPROCESSOR_SOURCE_DIR} GIT_TAG ${BOOST_PREPROCESSOR_GIT_TAG})

set(BOOST_STATIC_ASSERT_GIT_TAG 45eec41c293bc5cd36ec3ed83671f70bc1aadc9f)
# Merge: ba72d33 a1abfec
# Author: jzmaddock <john@johnmaddock.co.uk>
# Date:   Tue Mar 8 09:35:50 2022 +0000
#
#     Merge pull request #15 from sdarwin/githubactions
add_boost_module_headers(NAME "static_assert" SRC_DIR ${BOOST_STATIC_ASSERT_SOURCE_DIR} GIT_TAG ${BOOST_STATIC_ASSERT_GIT_TAG})

set(BOOST_THROW_EXCEPTION_GIT_TAG 23dd41e920ecd91237500ac6428f7d392a7a875c)
# Author: Peter Dimov <pdimov@gmail.com>
# Date:   Sun Jun 25 16:12:57 2023 +0300
#
#     Update ci.yml
add_boost_module_headers(NAME "throw_exception" SRC_DIR ${BOOST_THROW_EXCEPTION_SOURCE_DIR} GIT_TAG ${BOOST_THROW_EXCEPTION_GIT_TAG})

set(BOOST_TUPLE_GIT_TAG 500e4fa0a2845b96c0dd919e7485e0f216438a01)
# Merge: aa16ae3 ded3c1d
# Author: Joel de Guzman <djowel@gmail.com>
# Date:   Thu Dec 30 23:20:18 2021 +0800
#
#     Merge pull request #21 from igaztanaga/patch-1
add_boost_module_headers(NAME "tuple" SRC_DIR ${BOOST_TUPLE_SOURCE_DIR} GIT_TAG ${BOOST_TUPLE_GIT_TAG})

set(BOOST_TYPE_TRAITS_GIT_TAG 89f5011b4a79d91e42735670e39f72cb25c86c72)
# Merge: 55feb75 1ebd31e
# Author: John Maddock <john@johnmaddock.co.uk>
# Date:   Fri Feb 24 18:02:30 2023 +0000
#
#     Merge branch 'develop'
add_boost_module_headers(NAME "type_traits" SRC_DIR ${BOOST_TYPE_TRAITS_SOURCE_DIR} GIT_TAG ${BOOST_TYPE_TRAITS_GIT_TAG})
