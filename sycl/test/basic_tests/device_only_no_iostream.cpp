// CUSTOMER REQUIREMENT: the SYCL KHR header surface must NOT pull in
// <iostream>, <istream>, <ostream> (or the runtime iostream_proxy shim) on
// any compilation pass, including device. Customers compile their device
// code with -fsycl-device-only and expect a clean dependency graph; pulling
// the standard iostream machinery breaks that contract by introducing
// globals (std::cin/cout/cerr) and ios_base::Init static
// initializers that fail in device JIT/AOT pipelines.
//
// This test enforces that contract for every KHR header by compiling a
// test-only umbrella (Inputs/khr_all.hpp) that pulls in the entire
// <sycl/khr/...> tree, then checking the produced -MD dependency list.
//
// If you change a KHR header (or anything it transitively includes from
// sycl/...) and break this test, do not relax the check -- move the
// offending stream code into the SYCL runtime (sycl/source/...) and keep
// the header device-clean.
//
// We enforce the contract with TWO complementary guards, because the -MD
// dependency list is flat (it records that a header was reached, not who
// pulled it in):
//
//   1. A FileCheck pass over the -MD list that rejects the iostream
//      machinery -- <iostream>, the iostream_proxy shim, and the C-style
//      *.h stream headers. These are never dragged in by <iterator>; their
//      presence means real std::cin/cout/cerr globals and an ios_base::Init
//      static initializer entered the graph.
//
//   2. A source-grep guard that rejects any *raw* `#include <ostream>`,
//      <istream>, <iostream> or <sstream> written in a SYCL header that the
//      KHR umbrella actually reaches. This is what catches an author adding a
//      stream include to a KHR-reachable header.
//
// Why not simply CHECK-NOT the bare <ostream>/<istream> paths in the -MD list?
// Because on old libstdc++ (e.g. gcc 8) <iterator> unconditionally pulls in
// <ostream>/<istream> to define std::ostream_iterator/istream_iterator, and
// the KHR surface legitimately uses <iterator> (via sycl/multi_ptr.hpp).
// Newer libstdc++ and libc++ do not. So a bare-path check both false-fails on
// old toolchains and, being a stdlib artifact, does not actually indicate a
// SYCL header regression. Guard #2 catches the real regression (a raw include
// in our own header) regardless of standard-library version.
//
// <sycl/sycl.hpp> and headers under <sycl/ext/...> are intentionally out of
// scope -- only the KHR set is required to be device-safe today. The OV team
// will interface mainly through the kernel compiler and the KHR headers. As
// such this test covers their use case.
//
// NOTE: sycl.hpp pulls in iostream indirectly through
// the filesystem inclusion in:
// sycl/ext/oneapi/experimental/syclbin_kernel_bundle.hpp:88:get_kernel_bundle.
// Filesystem transitively pulls in iostream, so we intentionally exclude
// sycl.hpp from this test.
//
// RUN: %clangxx -fsycl -fsycl-device-only -include %S/Inputs/khr_all.hpp \
// RUN:   -c %s -o %t.o -MD -MF %t.d
//
// Guard 1: no iostream machinery in the dependency graph.
// RUN: FileCheck %s < %t.d
//
// Guard 2: no raw stream #include in any KHR-reachable SYCL header. A small
// Python helper extracts the SYCL header paths from the -MD list and greps
// their sources; it exits nonzero if it finds a match. Done in Python rather
// than a grep/xargs pipeline so the guard runs on the Windows lit shell too,
// and so it handles both '/' and '\' dependency-path separators.
// RUN: %python %sycl_tools_src_dir/check_no_raw_stream_include.py %t.d
//
// CHECK-NOT: iostream_proxy.hpp
// CHECK-NOT: {{/iostream[ \\]}}
// CHECK-NOT: {{/iostream\.h}}
// CHECK-NOT: {{/ostream\.h}}
// CHECK-NOT: {{/istream\.h}}
