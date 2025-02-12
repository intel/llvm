// sycl/sycl.hpp is a very heavy header, because it includes all SYCL feature.
//
// In order to be able to split it, we need to make sure that all
// inter-dependencies between headers are resolved. We already have a set of
// tests which make sure that every header can be compiled if included alone,
// but to achieve that, every header can simply include every other header.
//
// Considering that the main reason for sycl.hpp splitting is compilation time
// improvement, it is important that SYCL headers do not include each other
// unnecessary to make sure that when you include a single specific header, you
// only get specific functionality and nothing extra.
//
// Note that this test ignores "ext" subfolder for now and concentrates on
// core SYCL functionality for now.
//
// REQUIRES: linux
//
// RUN: grep -rl "#include <sycl/sub_group.hpp>" %sycl_include/sycl \
// RUN:     --exclude-dir=*ext* | FileCheck %s --check-prefix=SUB-GROUP
// SUB-GROUP: sycl/sycl.hpp
// SUB-GROUP-EMPTY:
//
// RUN: grep -rl "#include <sycl/stream.hpp>" %sycl_include/sycl \
// RUN:     --exclude-dir=*ext* | FileCheck %s --check-prefix=STREAM
// STREAM: sycl/sycl.hpp
// STREAM-EMPTY:

