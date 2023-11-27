# Preprocess boost/mp11 headers to allow using specific version of mp11 (as
# defined in these project's cmake files) within SYCL library w/o risk of
# conflict with user program using another version of boost/mp11. Basically,
# this transformation moves all APIs from boost namespace to sycl::boost and
# adds SYCL_ prefix to boost macros names. See more specific comments in the
# code below.
# Variables which must be set by the caller to control behavior of this module:
# - IN
#   The source directory with mp11 headers, must contain mp11.h.
# - OUT
#   The destination directory where preprocessed source headers are put.
# - SRC_PATH
#   An URL or directory name the source directory originates from. Used only in
#   generated README text.
# - SRC_ID
#   Git hash/tag or other ID identifying the original source. Used only in
#   generated README text.
#
# Assumed to be invoked as a script:
# ${CMAKE_COMMAND} -DIN=... -DOUT=... -DSRC_PATH=... -DSRC_ID=... -P <this file>

function(preprocess_mp11_header)
  cmake_parse_arguments(
    MP11_HDR                   # prefix
    ""                         # options
    "SRC_NAME;DST_NAME;IN_DIR" # one value keywords
    ""                         # multi-value keywords
    ${ARGN})                   # arguments
  file(READ ${MP11_HDR_SRC_NAME} FILE_CONTENTS)

  # 1) replace `BOOST_*` macros with `SYCL_BOOST_*`.
  string(REGEX REPLACE
    "([ \t\n\r!(])BOOST_"
    "\\1SYCL_DETAIL_BOOST_"
    FILE_CONTENTS "${FILE_CONTENTS}")
  # 2) replace `namespace boost { ... }` with
  # `namespace sycl { namespace detail { namespace boost { ... } } }`
  string(REGEX REPLACE
    "(\n[ \t]*namespace[ \t\n\r]+boost)"
    "namespace sycl\n{\ninline namespace _V1\n{\nnamespace detail\n{\\1"
    FILE_CONTENTS "${FILE_CONTENTS}")
  # ... use '} // namespace boost' as a marker for end-of-scope '}' replacement
  string(REGEX REPLACE
    "(\n[ \t]*}[ \t]*//[ \t]*namespace[ \t]+boost[ \t]*\n)"
    "\\1} // namespace detail\n} // namespace _V1\n} // namespace sycl\n"
    FILE_CONTENTS "${FILE_CONTENTS}")
  # 3) replace `boost` in `#include <boost/...>` or `#include "boost/..."` with
  # `sycl/detail/boost`
  string(REGEX REPLACE
    "(\n#include[ \t]*[<\"])boost"
    "\\1sycl/detail/boost"
    FILE_CONTENTS "${FILE_CONTENTS}")

  set(SYCL_DERIVED_COPYRIGHT_NOTICE "\
  // -*- C++ -*-\n\
  //===----------------------------------------------------------------------===//\n\
  // Modifications Copyright Intel Corporation 2022\n\
  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n\
  //===----------------------------------------------------------------------===//\n\
  // Auto-generated from boost/mp11 sources https://github.com/boostorg/mp11\n\n")

  # 4) add proper copyright notice atop
  string(PREPEND FILE_CONTENTS ${SYCL_DERIVED_COPYRIGHT_NOTICE})
  file(WRITE ${MP11_HDR_DST_NAME} "${FILE_CONTENTS}")
endfunction(preprocess_mp11_header)

function(preprocess_mp11_headers)
  cmake_parse_arguments(
    MP11_HDRS                # prefix
    ""                       # options
    "IN;OUT;SRC_PATH;SRC_ID" # one value keywords
    ""                       # multi-value keywords
    ${ARGN})                 # arguments

  # 1) Perform necessary preprocessing of headers.
  file(GLOB_RECURSE BOOST_MP11_SOURCES "${MP11_HDRS_IN}/*")
  
  foreach(SRC ${BOOST_MP11_SOURCES})
    string(REPLACE "${MP11_HDRS_IN}" "${MP11_HDRS_OUT}" DST "${SRC}")
    preprocess_mp11_header(
      SRC_NAME ${SRC}
      DST_NAME ${DST}
      IN_DIR ${MP11_HDRS_IN}
    )
  endforeach(SRC ${BOOST_MP11_SOURCES})
  
  # 2) Add SYCL_README.txt to the output directory root
  set(SYCL_README_TEXT "\
  This directory contains boost/mp11 headers adapted for use in SYCL headers in\n\
  a way that does not conflict with potential use of boost in user code.\n\
  Particularly, `BOOST_*` macros are replaced with `SYCL_DETAIL_BOOST_*`, APIs\n\
  are moved into the top-level `sycl::detail` namespace. For example,\n\
  `sycl::detail::boost::mp11::mp_list`.\n")
  
  set(SYCL_README_FILE_NAME "${MP11_HDRS_OUT}/README.txt")
  
  file(WRITE ${SYCL_README_FILE_NAME} "${SYCL_README_TEXT}")
endfunction(preprocess_mp11_headers)

preprocess_mp11_headers(
    IN ${IN}
    OUT ${OUT}
    SRC_PATH ${SRC_PATH}
    SRC_ID ${SRC_ID}
)
