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

set(BOOST_UNORDERED_GIT_TAG 5e6b9291deb55567d41416af1e77c2516dc1250f)
# Merge: 15cfef69 ccf9a76e
# Author: joaquintides <joaquin.lopezmunoz@gmail.com>
# Date:   Sat Mar 16 09:18:41 2024 +0100
#
#     Merge pull request #238 from boostorg/fix/gh-237
add_boost_module_headers(NAME "unordered" SRC_DIR ${BOOST_UNORDERED_SOURCE_DIR} GIT_TAG ${BOOST_UNORDERED_GIT_TAG})

set(BOOST_ASSERT_GIT_TAG 447e0b3a331930f8708ade0e42683d12de9dfbc3)
# Author: Peter Dimov <pdimov@gmail.com>
# Date:   Sat Feb 3 20:43:55 2024 +0200
#
#     Use __builtin_FUNCSIG() under MSVC 19.35+. Fixes #35.
add_boost_module_headers(NAME "assert" SRC_DIR ${BOOST_ASSERT_SOURCE_DIR} GIT_TAG ${BOOST_ASSERT_GIT_TAG})

set(BOOST_CONFIG_GIT_TAG 11385ec21012926e15a612e3bf9f9a71403c1e5b)
# Merge: eef05e98 601598f8
# Author: jzmaddock <john@johnmaddock.co.uk>
# Date:   Sun Feb 4 09:46:22 2024 +0000
#
#     Merge branch 'develop'
add_boost_module_headers(NAME "config" SRC_DIR ${BOOST_CONFIG_SOURCE_DIR} GIT_TAG ${BOOST_CONFIG_GIT_TAG})

set(BOOST_CONTAINER_HASH_GIT_TAG 6d214eb776456bf17fbee20780a034a23438084f)
# Author: Peter Dimov <pdimov@gmail.com>
# Date:   Wed Mar 6 05:13:53 2024 +0200
#
#     Update .appveyor.yml
add_boost_module_headers(NAME "container_hash" SRC_DIR ${BOOST_CONTAINER_HASH_SOURCE_DIR} GIT_TAG ${BOOST_CONTAINER_HASH_GIT_TAG})

set(BOOST_CORE_GIT_TAG 083b41c17e34f1fc9b43ab796b40d0d8bece685c)
# Merge: 8cc2fda a973490
# Author: Andrey Semashev <Lastique@users.noreply.github.com>
# Date:   Tue Mar 19 18:10:04 2024 +0300
#
#     Merge pull request #169 from k3DW/feature/168
add_boost_module_headers(NAME "core" SRC_DIR ${BOOST_CORE_SOURCE_DIR} GIT_TAG ${BOOST_CORE_GIT_TAG})

# Describe is a dependency of container_hash
set(BOOST_DESCRIBE_GIT_TAG 50719b212349f3d1268285c586331584d3dbfeb5)
# Author: Peter Dimov <pdimov@gmail.com>
# Date:   Sat Mar 23 20:27:08 2024 +0200
#
#     Update .drone.jsonnet
add_boost_module_headers(NAME "describe" SRC_DIR ${BOOST_DESCRIBE_SOURCE_DIR} GIT_TAG ${BOOST_DESCRIBE_GIT_TAG})

# Reuse mp11 fetched earlier for DPC++ headers
set(BOOST_UNORDERED_INCLUDE_DIRS ${BOOST_UNORDERED_INCLUDE_DIRS} "${BOOST_MP11_SOURCE_DIR}/include/")

set(BOOST_PREDEF_GIT_TAG 0fdfb49c3a6789e50169a44e88a07cc889001106)
# Merge: 392e4e7 614546d
# Author: Rene Rivera <grafikrobot@gmail.com>
# Date:   Tue Oct 31 20:24:41 2023 -0500
#
#     Merge branch 'develop'
add_boost_module_headers(NAME "predef" SRC_DIR ${BOOST_PREDEF_SOURCE_DIR} GIT_TAG ${BOOST_PREDEF_GIT_TAG})

# Static assert is a dependency of core
set(BOOST_STATIC_ASSERT_GIT_TAG ba72d3340f3dc6e773868107f35902292f84b07e)
# Merge: 392e4e7 614546d
# Author: Rene Rivera <grafikrobot@gmail.com>
# Date:   Tue Oct 31 20:24:41 2023 -0500
#
#     Merge branch 'develop'
add_boost_module_headers(NAME "static_assert" SRC_DIR ${BOOST_STATIC_ASSERT_SOURCE_DIR} GIT_TAG ${BOOST_STATIC_ASSERT_GIT_TAG})

set(BOOST_THROW_EXCEPTION_GIT_TAG 7c8ec2114bc1f9ab2a8afbd629b96fbdd5901294)
# Author: Peter Dimov <pdimov@gmail.com>
# Date:   Sat Jan 6 19:41:56 2024 +0200
#
#     Add -Wundef to test/Jamfile
add_boost_module_headers(NAME "throw_exception" SRC_DIR ${BOOST_THROW_EXCEPTION_SOURCE_DIR} GIT_TAG ${BOOST_THROW_EXCEPTION_GIT_TAG})
