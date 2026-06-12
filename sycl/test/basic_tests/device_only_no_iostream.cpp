// CUSTOMER REQUIREMENT: the SYCL KHR header surface must NOT pull in
// <iostream>, <istream>, <ostream> (or the runtime iostream_proxy shim) on
// any compilation pass, including device. Customers compile their device
// code with -fsycl-device-only and expect a clean dependency graph; pulling
// the standard iostream machinery breaks that contract by introducing
// host-only globals (std::cin/cout/cerr) and ios_base::Init static
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
// <sycl/sycl.hpp> and headers under <sycl/ext/...> are intentionally out of
// scope -- only the KHR set is required to be device-safe today.
//
// RUN: %clangxx -fsycl -fsycl-device-only -include %S/Inputs/khr_all.hpp \
// RUN:   -c %s -o %t.o -MD -MF - | FileCheck %s
//
// CHECK-NOT: iostream_proxy.hpp
// CHECK-NOT: {{/iostream[ \\]}}
// CHECK-NOT: {{/ostream[ \\]}}
// CHECK-NOT: {{/istream[ \\]}}
// CHECK-NOT: {{/iostream\.h}}
// CHECK-NOT: {{/ostream\.h}}
// CHECK-NOT: {{/istream\.h}}
