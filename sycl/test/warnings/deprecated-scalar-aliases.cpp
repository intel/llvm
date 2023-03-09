// RUN: %clangxx %fsycl-host-only -fsyntax-only -sycl-std=2020 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
//
// Our implementation has a number of aliases defined for scalar types, which
// were either changed or completely removed from SYCL 2020. To ensure a smooth
// transition for users, we deprecated them first and then remove.
//
// This test is intended to check that we emit proper deprecation message for
// those aliases.

#include <sycl/sycl.hpp>

int main() {
  sycl::schar sc;      // expected-warning {{'schar' is deprecated}}
  sycl::uchar uc;      // expected-warning {{'uchar' is deprecated}}
  sycl::ushort us;     // expected-warning {{'ushort' is deprecated}}
  sycl::uint ui;       // expected-warning {{'uint' is deprecated}}
  sycl::ulong ul;      // expected-warning {{'ulong' is deprecated}}
  sycl::longlong ll;   // expected-warning {{'longlong' is deprecated}}
  sycl::ulonglong ull; // expected-warning {{'ulonglong' is deprecated}}

  // expected-warning@+1 {{'cl_bool' is deprecated: use sycl::opencl::cl_bool instead}}
  sycl::cl_bool clb;
  // expected-warning@+1 {{'cl_char' is deprecated: use sycl::opencl::cl_char instead}}
  sycl::cl_char clc;
  // expected-warning@+1 {{'cl_uchar' is deprecated: use sycl::opencl::cl_uchar instead}}
  sycl::cl_uchar cluc;
  // expected-warning@+1 {{'cl_short' is deprecated: use sycl::opencl::cl_short instead}}
  sycl::cl_short cls;
  // expected-warning@+1 {{'cl_ushort' is deprecated: use sycl::opencl::cl_ushort instead}}
  sycl::cl_ushort clus;
  // expected-warning@+1 {{'cl_int' is deprecated: use sycl::opencl::cl_int instead}}
  sycl::cl_int cli;
  // expected-warning@+1 {{'cl_uint' is deprecated: use sycl::opencl::cl_uint instead}}
  sycl::cl_uint clui;
  // expected-warning@+1 {{'cl_long' is deprecated: use sycl::opencl::cl_long instead}}
  sycl::cl_long cll;
  // expected-warning@+1 {{'cl_ulong' is deprecated: use sycl::opencl::cl_ulong instead}}
  sycl::cl_ulong clul;
  // expected-warning@+1 {{'cl_half' is deprecated: use sycl::opencl::cl_half instead}}
  sycl::cl_half clh;
  // expected-warning@+1 {{'cl_float' is deprecated: use sycl::opencl::cl_float instead}}
  sycl::cl_float clf;
  // expected-warning@+1 {{'cl_double' is deprecated: use sycl::opencl::cl_double instead}}
  sycl::cl_double cld;

  return 0;
}
