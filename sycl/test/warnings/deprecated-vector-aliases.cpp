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

  // expected-warning@+1 {{'schar2' is deprecated}}
  sycl::schar2 sc2;
  // expected-warning@+1 {{'schar3' is deprecated}}
  sycl::schar3 sc3;
  // expected-warning@+1 {{'schar4' is deprecated}}
  sycl::schar4 sc4;
  // expected-warning@+1 {{'schar8' is deprecated}}
  sycl::schar8 sc8;
  // expected-warning@+1 {{'schar16' is deprecated}}
  sycl::schar16 sc16;

  // expected-warning@+1 {{'longlong2' is deprecated}}
  sycl::longlong2 ll2;
  // expected-warning@+1 {{'longlong3' is deprecated}}
  sycl::longlong3 ll3;
  // expected-warning@+1 {{'longlong4' is deprecated}}
  sycl::longlong4 ll4;
  // expected-warning@+1 {{'longlong8' is deprecated}}
  sycl::longlong8 ll8;
  // expected-warning@+1 {{'longlong16' is deprecated}}
  sycl::longlong16 ll16;

  // expected-warning@+1 {{'ulonglong2' is deprecated}}
  sycl::ulonglong2 ull2;
  // expected-warning@+1 {{'ulonglong3' is deprecated}}
  sycl::ulonglong3 ull3;
  // expected-warning@+1 {{'ulonglong4' is deprecated}}
  sycl::ulonglong4 ull4;
  // expected-warning@+1 {{'ulonglong8' is deprecated}}
  sycl::ulonglong8 ull8;
  // expected-warning@+1 {{'ulonglong16' is deprecated}}
  sycl::ulonglong16 ull16;

  // expected-warning@+1 {{'cl_char2' is deprecated}}
  sycl::cl_char2 cl_c2;
  // expected-warning@+1 {{'cl_char3' is deprecated}}
  sycl::cl_char3 cl_c3;
  // expected-warning@+1 {{'cl_char4' is deprecated}}
  sycl::cl_char4 cl_c4;
  // expected-warning@+1 {{'cl_char8' is deprecated}}
  sycl::cl_char8 cl_c8;
  // expected-warning@+1 {{'cl_char16' is deprecated}}
  sycl::cl_char16 cl_c16;

  // expected-warning@+1 {{'cl_uchar2' is deprecated}}
  sycl::cl_uchar2 cl_uc2;
  // expected-warning@+1 {{'cl_uchar3' is deprecated}}
  sycl::cl_uchar3 cl_uc3;
  // expected-warning@+1 {{'cl_uchar4' is deprecated}}
  sycl::cl_uchar4 cl_uc4;
  // expected-warning@+1 {{'cl_uchar8' is deprecated}}
  sycl::cl_uchar8 cl_uc8;
  // expected-warning@+1 {{'cl_uchar16' is deprecated}}
  sycl::cl_uchar16 cl_uc16;

  // expected-warning@+1 {{'cl_short2' is deprecated}}
  sycl::cl_short2 cl_s2;
  // expected-warning@+1 {{'cl_short3' is deprecated}}
  sycl::cl_short3 cl_s3;
  // expected-warning@+1 {{'cl_short4' is deprecated}}
  sycl::cl_short4 cl_s4;
  // expected-warning@+1 {{'cl_short8' is deprecated}}
  sycl::cl_short8 cl_s8;
  // expected-warning@+1 {{'cl_short16' is deprecated}}
  sycl::cl_short16 cl_s16;

  // expected-warning@+1 {{'cl_ushort2' is deprecated}}
  sycl::cl_ushort2 cl_us2;
  // expected-warning@+1 {{'cl_ushort3' is deprecated}}
  sycl::cl_ushort3 cl_us3;
  // expected-warning@+1 {{'cl_ushort4' is deprecated}}
  sycl::cl_ushort4 cl_us4;
  // expected-warning@+1 {{'cl_ushort8' is deprecated}}
  sycl::cl_ushort8 cl_us8;
  // expected-warning@+1 {{'cl_ushort16' is deprecated}}
  sycl::cl_ushort16 cl_us16;

  // expected-warning@+1 {{'cl_int2' is deprecated}}
  sycl::cl_int2 cl_i2;
  // expected-warning@+1 {{'cl_int3' is deprecated}}
  sycl::cl_int3 cl_i3;
  // expected-warning@+1 {{'cl_int4' is deprecated}}
  sycl::cl_int4 cl_i4;
  // expected-warning@+1 {{'cl_int8' is deprecated}}
  sycl::cl_int8 cl_i8;
  // expected-warning@+1 {{'cl_int16' is deprecated}}
  sycl::cl_int16 cl_i16;

  // expected-warning@+1 {{'cl_uint2' is deprecated}}
  sycl::cl_uint2 cl_ui2;
  // expected-warning@+1 {{'cl_uint3' is deprecated}}
  sycl::cl_uint3 cl_ui3;
  // expected-warning@+1 {{'cl_uint4' is deprecated}}
  sycl::cl_uint4 cl_ui4;
  // expected-warning@+1 {{'cl_uint8' is deprecated}}
  sycl::cl_uint8 cl_ui8;
  // expected-warning@+1 {{'cl_uint16' is deprecated}}
  sycl::cl_uint16 cl_ui16;

  // expected-warning@+1 {{'cl_long2' is deprecated}}
  sycl::cl_long2 cl_l2;
  // expected-warning@+1 {{'cl_long3' is deprecated}}
  sycl::cl_long3 cl_l3;
  // expected-warning@+1 {{'cl_long4' is deprecated}}
  sycl::cl_long4 cl_l4;
  // expected-warning@+1 {{'cl_long8' is deprecated}}
  sycl::cl_long8 cl_l8;
  // expected-warning@+1 {{'cl_long16' is deprecated}}
  sycl::cl_long16 cl_l16;

  // expected-warning@+1 {{'cl_ulong2' is deprecated}}
  sycl::cl_ulong2 cl_ul2;
  // expected-warning@+1 {{'cl_ulong3' is deprecated}}
  sycl::cl_ulong3 cl_ul3;
  // expected-warning@+1 {{'cl_ulong4' is deprecated}}
  sycl::cl_ulong4 cl_ul4;
  // expected-warning@+1 {{'cl_ulong8' is deprecated}}
  sycl::cl_ulong8 cl_ul8;
  // expected-warning@+1 {{'cl_ulong16' is deprecated}}
  sycl::cl_ulong16 cl_ul16;

  // expected-warning@+1 {{'cl_float2' is deprecated}}
  sycl::cl_float2 cl_f2;
  // expected-warning@+1 {{'cl_float3' is deprecated}}
  sycl::cl_float3 cl_f3;
  // expected-warning@+1 {{'cl_float4' is deprecated}}
  sycl::cl_float4 cl_f4;
  // expected-warning@+1 {{'cl_float8' is deprecated}}
  sycl::cl_float8 cl_f8;
  // expected-warning@+1 {{'cl_float16' is deprecated}}
  sycl::cl_float16 cl_f16;

  // expected-warning@+1 {{'cl_double2' is deprecated}}
  sycl::cl_double2 cl_d2;
  // expected-warning@+1 {{'cl_double3' is deprecated}}
  sycl::cl_double3 cl_d3;
  // expected-warning@+1 {{'cl_double4' is deprecated}}
  sycl::cl_double4 cl_d4;
  // expected-warning@+1 {{'cl_double8' is deprecated}}
  sycl::cl_double8 cl_d8;
  // expected-warning@+1 {{'cl_double16' is deprecated}}
  sycl::cl_double16 cl_d16;

  // expected-warning@+1 {{'cl_half2' is deprecated}}
  sycl::cl_half2 cl_h2;
  // expected-warning@+1 {{'cl_half3' is deprecated}}
  sycl::cl_half3 cl_h3;
  // expected-warning@+1 {{'cl_half4' is deprecated}}
  sycl::cl_half4 cl_h4;
  // expected-warning@+1 {{'cl_half8' is deprecated}}
  sycl::cl_half8 cl_h8;
  // expected-warning@+1 {{'cl_half16' is deprecated}}
  sycl::cl_half16 cl_h16;
};
