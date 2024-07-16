// RUN: %clangxx -std=c++17 -I %sycl_include -I %sycl_include/sycl -fsycl-device-only -c -fno-color-diagnostics -Xclang -fdump-record-layouts-complete %s -o %t.out | grep -Pzo "0 \| class sycl::.*\n([^\n].*\n)*" | sort -z | FileCheck --implicit-check-not "{{std::basic_string|std::list}}" %s
// RUN: %clangxx -std=c++17 -I %sycl_include -I %sycl_include/sycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts-complete %s -o %t.out | grep -Pzo "0 \| class sycl::.*\n([^\n].*\n)*" | sort -z | FileCheck --implicit-check-not "{{std::basic_string|std::list}}" %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// The purpose of this test is to check that classes in sycl namespace that are
// defined in SYCL headers don't have std::string and std::list data members to
// avoid having the dual ABI issue (see
// https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html). I.e. if
// application is built with the old ABI and such data member is crossing ABI
// boundary then it will result in issues as SYCL RT is using new ABI by
// default. All such data members can potentially cross ABI boundaries and
// that's why we need to be sure that we use only ABI-neutral data members.

// Exclusions are NOT ALLOWED to this file unless it is guaranteed that data
// member is not crossing ABI boundary. If there is a std::string/std::list
// data member which is guaranteed to not cross ABI boundary then it must be
// matched in this test explicitly.

#include <sycl/sycl.hpp>
