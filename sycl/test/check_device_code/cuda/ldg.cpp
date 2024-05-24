// REQUIRES: cuda

// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xclang -fnative-half-type -S -Xclang -emit-llvm %s -o -| FileCheck %s --check-prefixes=CHECK-OPAQUE

#include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::cuda;
using namespace sycl::ext::oneapi::experimental;

char *in_c;
signed char *in_sc;
short *in_s;
int *in_i;
long *in_l;
long long *in_ll;

unsigned char *in_uc;
unsigned short *in_us;
unsigned int *in_ui;
unsigned long *in_ul;
unsigned long long *in_ull;

sycl::vec<char, 2> *in_c2;
sycl::vec<char, 3> *in_c3;
sycl::vec<signed char, 2> *in_sc2;
sycl::vec<signed char, 3> *in_sc3;
sycl::vec<short, 2> *in_s2;
sycl::vec<short, 3> *in_s3;
sycl::vec<int, 2> *in_i2;
sycl::vec<int, 3> *in_i3;
sycl::vec<long, 2> *in_l2;
sycl::vec<long, 3> *in_l3;
sycl::vec<long long, 2> *in_ll2;
sycl::vec<long long, 3> *in_ll3;
sycl::vec<long, 4> *in_l4;
sycl::vec<long long, 4> *in_ll4;

sycl::vec<char, 4> *in_c4;
sycl::vec<signed char, 4> *in_sc4;
sycl::vec<short, 4> *in_s4;
sycl::vec<int, 4> *in_i4;

sycl::vec<unsigned char, 2> *in_uc2;
sycl::vec<unsigned char, 3> *in_uc3;
sycl::vec<unsigned short, 2> *in_us2;
sycl::vec<unsigned short, 3> *in_us3;
sycl::vec<unsigned int, 2> *in_ui2;

sycl::vec<unsigned int, 3> *in_ui3;
sycl::vec<unsigned long, 2> *in_ul2;
sycl::vec<unsigned long, 3> *in_ul3;
sycl::vec<unsigned long long, 2> *in_ull2;
sycl::vec<unsigned long long, 3> *in_ull3;
sycl::vec<unsigned long, 4> *in_ul4;
sycl::vec<unsigned long long, 4> *in_ull4;

sycl::vec<unsigned char, 4> *in_uc4;
sycl::vec<unsigned short, 4> *in_us4;
sycl::vec<unsigned int, 4> *in_ui4;

half *in_h;
float *in_f;
double *in_d;

sycl::vec<half, 2> *in_h2;
sycl::vec<half, 3> *in_h3;
sycl::vec<half, 4> *in_h4;
sycl::vec<float, 2> *in_f2;
sycl::vec<float, 3> *in_f3;
sycl::vec<float, 4> *in_f4;
sycl::vec<double, 2> *in_d2;
sycl::vec<double, 3> *in_d3;
sycl::vec<double, 4> *in_d4;

SYCL_EXTERNAL void check_ldg() {
  //CHECK-OPAQUE: tail call half @llvm.nvvm.ldg.global.f.f16.p0(ptr %{{.*}}, i32 2)
  auto cached_h = ldg(&in_h[0]);
  //CHECK-OPAQUE: tail call noundef float @llvm.nvvm.ldg.global.f.f32.p0(ptr %{{.*}}, i32 4)
  auto cached_f = ldg(&in_f[0]);
  //CHECK-OPAQUE: tail call noundef double @llvm.nvvm.ldg.global.f.f64.p0(ptr %{{.*}}, i32 8)
  auto cached_d = ldg(&in_d[0]);

  //CHECK-OPAQUE: tail call <2 x half> @llvm.nvvm.ldg.global.f.v2f16.p0(ptr %{{.*}}, i32 4)
  auto cached_h2 = ldg(&in_h2[0]);
  //CHECK-OPAQUE: tail call <2 x half> @llvm.nvvm.ldg.global.f.v2f16.p0(ptr %{{.*}}, i32 4)
  //CHECK-OPAQUE: tail call half @llvm.nvvm.ldg.global.f.f16.p0(ptr nonnull %{{.*}}, i32 2)
  auto cached_h3 = ldg(&in_h3[0]);
  //CHECK-OPAQUE: tail call <2 x half> @llvm.nvvm.ldg.global.f.v2f16.p0(ptr %{{.*}}, i32 4)
  //CHECK-OPAQUE: tail call <2 x half> @llvm.nvvm.ldg.global.f.v2f16.p0(ptr nonnull %{{.*}}, i32 4)
  auto cached_h4 = ldg(&in_h4[0]);
  //CHECK-OPAQUE: tail call <2 x float> @llvm.nvvm.ldg.global.f.v2f32.p0(ptr %{{.*}}, i32 8)
  auto cached_f2 = ldg(&in_f2[0]);
  //CHECK-OPAQUE: tail call <2 x float> @llvm.nvvm.ldg.global.f.v2f32.p0(ptr %{{.*}}, i32 8)
  //CHECK-OPAQUE: tail call float @llvm.nvvm.ldg.global.f.f32.p0(ptr nonnull %{{.*}}, i32 4)
  auto cached_f3 = ldg(&in_f3[0]);
  //CHECK-OPAQUE: tail call <2 x double> @llvm.nvvm.ldg.global.f.v2f64.p0(ptr %{{.*}}, i32 16)
  auto cached_d2 = ldg(&in_d2[0]);
  //CHECK-OPAQUE: tail call <2 x double> @llvm.nvvm.ldg.global.f.v2f64.p0(ptr %{{.*}}, i32 16)
  //CHECK-OPAQUE: tail call double @llvm.nvvm.ldg.global.f.f64.p0(ptr nonnull %{{.*}}, i32 8)
  auto cached_d3 = ldg(&in_d3[0]);
  //CHECK-OPAQUE: tail call <2 x double> @llvm.nvvm.ldg.global.f.v2f64.p0(ptr %{{.*}}, i32 16)
  //CHECK-OPAQUE: tail call <2 x double> @llvm.nvvm.ldg.global.f.v2f64.p0(ptr nonnull %{{.*}}, i32 16)
  auto cached_d4 = ldg(&in_d4[0]);
  //CHECK-OPAQUE: tail call <4 x float> @llvm.nvvm.ldg.global.f.v4f32.p0(ptr %{{.*}}, i32 16)
  auto cached_f4 = ldg(&in_f4[0]);

  // Unsigned variants are identical to signed variants, but this leads to
  // correct behavior.

  //CHECK-OPAQUE: tail call noundef i8 @llvm.nvvm.ldg.global.i.i8.p0(ptr %{{.*}}, i32 1)
  auto cached_c = ldg(&in_c[0]);
  //CHECK-OPAQUE: tail call noundef i8 @llvm.nvvm.ldg.global.i.i8.p0(ptr %{{.*}}, i32 1)
  auto cached_sc = ldg(&in_sc[0]);
  //CHECK-OPAQUE: tail call noundef i16 @llvm.nvvm.ldg.global.i.i16.p0(ptr %{{.*}}, i32 2)
  auto cached_s = ldg(&in_s[0]);
  //CHECK-OPAQUE: tail call noundef i32 @llvm.nvvm.ldg.global.i.i32.p0(ptr %{{.*}}, i32 4)
  auto cached_i = ldg(&in_i[0]);
  //CHECK-OPAQUE: tail call noundef i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr %{{.*}}, i32 8)
  auto cached_l = ldg(&in_l[0]);
  //CHECK-OPAQUE: tail call noundef i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr %{{.*}}, i32 8)
  auto cached_ll = ldg(&in_ll[0]);
  //CHECK-OPAQUE: tail call noundef i8 @llvm.nvvm.ldg.global.i.i8.p0(ptr %{{.*}}, i32 1)
  auto cached_uc = ldg(&in_uc[0]);
  //CHECK-OPAQUE: tail call noundef i16 @llvm.nvvm.ldg.global.i.i16.p0(ptr %{{.*}}, i32 2)
  auto cached_us = ldg(&in_us[0]);
  //CHECK-OPAQUE: tail call noundef i32 @llvm.nvvm.ldg.global.i.i32.p0(ptr %{{.*}}, i32 4)
  auto cached_ui = ldg(&in_ui[0]);
  //CHECK-OPAQUE: tail call noundef i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr %{{.*}}, i32 8)
  auto cached_ul = ldg(&in_ul[0]);
  //CHECK-OPAQUE: tail call noundef i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr %{{.*}}, i32 8)
  auto cached_ull = ldg(&in_ull[0]);

  //CHECK-OPAQUE: tail call <2 x i8> @llvm.nvvm.ldg.global.i.v2i8.p0(ptr %{{.*}}, i32 2)
  auto cached_c2 = ldg(&in_c2[0]);
  //CHECK-OPAQUE: tail call <2 x i8> @llvm.nvvm.ldg.global.i.v2i8.p0(ptr %{{.*}}, i32 2)
  //CHECK-OPAQUE: tail call i8 @llvm.nvvm.ldg.global.i.i8.p0(ptr nonnull %{{.*}}, i32 1)
  auto cached_c3 = ldg(&in_c3[0]);
  //CHECK-OPAQUE: tail call <2 x i8> @llvm.nvvm.ldg.global.i.v2i8.p0(ptr %{{.*}}, i32 2)
  auto cached_sc2 = ldg(&in_sc2[0]);
  //CHECK-OPAQUE: tail call <2 x i8> @llvm.nvvm.ldg.global.i.v2i8.p0(ptr %{{.*}}, i32 2)
  //CHECK-OPAQUE: tail call i8 @llvm.nvvm.ldg.global.i.i8.p0(ptr nonnull %{{.*}}, i32 1)
  auto cached_sc3 = ldg(&in_sc3[0]);
  //CHECK-OPAQUE: tail call <2 x i16> @llvm.nvvm.ldg.global.i.v2i16.p0(ptr %{{.*}}, i32 4)
  auto cached_s2 = ldg(&in_s2[0]);
  //CHECK-OPAQUE: tail call <2 x i16> @llvm.nvvm.ldg.global.i.v2i16.p0(ptr %{{.*}}, i32 4)
  //CHECK-OPAQUE: tail call i16 @llvm.nvvm.ldg.global.i.i16.p0(ptr nonnull %{{.*}}, i32 2)
  auto cached_s3 = ldg(&in_s3[0]);
  //CHECK-OPAQUE: tail call <2 x i32> @llvm.nvvm.ldg.global.i.v2i32.p0(ptr %{{.*}}, i32 8)
  auto cached_i2 = ldg(&in_i2[0]);
  //CHECK-OPAQUE: tail call <2 x i32> @llvm.nvvm.ldg.global.i.v2i32.p0(ptr %{{.*}}, i32 8)
  //CHECK-OPAQUE: tail call i32 @llvm.nvvm.ldg.global.i.i32.p0(ptr nonnull %{{.*}}, i32 4)
  auto cached_i3 = ldg(&in_i3[0]);
  //CHECK-OPAQUE: tail call <2 x i{{32|64}}> @llvm.nvvm.ldg.global.i.v2i{{32|64}}.p0(ptr %{{.*}}, i32 {{8|16}})
  auto cached_l2 = ldg(&in_l2[0]);
  //CHECK-OPAQUE: tail call <2 x i{{32|64}}> @llvm.nvvm.ldg.global.i.v2i{{32|64}}.p0(ptr %{{.*}}, i32 {{8|16}})
  //CHECK-OPAQUE: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr nonnull %{{.*}}, i32 8)
  auto cached_l3 = ldg(&in_l3[0]);
  //CHECK-OPAQUE: tail call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0(ptr %{{.*}}, i32 16)
  auto cached_ll2 = ldg(&in_ll2[0]);
  //CHECK-OPAQUE: tail call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0(ptr %{{.*}}, i32 16)
  //CHECK-OPAQUE: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr nonnull %{{.*}}, i32 8)
  auto cached_ll3 = ldg(&in_ll3[0]);
  //CHECK-OPAQUE: tail call <2 x i{{32|64}}> @llvm.nvvm.ldg.global.i.v2i{{32|64}}.p0(ptr %{{.*}}, i32 {{8|16}})
  //CHECK-OPAQUE: tail call <2 x i{{32|64}}> @llvm.nvvm.ldg.global.i.v2i{{32|64}}.p0(ptr nonnull %{{.*}}, i32 {{8|16}})
  auto cached_l4 = ldg(&in_l4[0]);
  //CHECK-OPAQUE: tail call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0(ptr %{{.*}}, i32 16)
  //CHECK-OPAQUE: tail call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0(ptr nonnull %{{.*}}, i32 16)
  auto cached_ll4 = ldg(&in_ll4[0]);
  //CHECK-OPAQUE: tail call <2 x i8> @llvm.nvvm.ldg.global.i.v2i8.p0(ptr %{{.*}}, i32 2)
  auto cached_uc2 = ldg(&in_uc2[0]);
  //CHECK-OPAQUE: tail call <2 x i8> @llvm.nvvm.ldg.global.i.v2i8.p0(ptr %{{.*}}, i32 2)
  //CHECK-OPAQUE: tail call i8 @llvm.nvvm.ldg.global.i.i8.p0(ptr nonnull %{{.*}}, i32 1)
  auto cached_uc3 = ldg(&in_uc3[0]);
  //CHECK-OPAQUE: tail call <2 x i16> @llvm.nvvm.ldg.global.i.v2i16.p0(ptr %{{.*}}, i32 4)
  auto cached_us2 = ldg(&in_us2[0]);
  //CHECK-OPAQUE: tail call <2 x i16> @llvm.nvvm.ldg.global.i.v2i16.p0(ptr %{{.*}}, i32 4)
  //CHECK-OPAQUE: tail call i16 @llvm.nvvm.ldg.global.i.i16.p0(ptr nonnull %{{.*}}, i32 2)
  auto cached_us3 = ldg(&in_us3[0]);
  //CHECK-OPAQUE: tail call <2 x i32> @llvm.nvvm.ldg.global.i.v2i32.p0(ptr %{{.*}}, i32 8)
  auto cached_ui2 = ldg(&in_ui2[0]);
  //CHECK-OPAQUE: tail call <2 x i32> @llvm.nvvm.ldg.global.i.v2i32.p0(ptr %{{.*}}, i32 8)
  //CHECK-OPAQUE: tail call i32 @llvm.nvvm.ldg.global.i.i32.p0(ptr nonnull %{{.*}}, i32 4)
  auto cached_ui3 = ldg(&in_ui3[0]);
  //CHECK-OPAQUE: tail call <2 x i{{64|32}}> @llvm.nvvm.ldg.global.i.v2i{{64|32}}.p0(ptr %{{.*}}, i32 {{8|16}})
  auto cached_ul2 = ldg(&in_ul2[0]);
  //CHECK-OPAQUE: tail call <2 x i{{64|32}}> @llvm.nvvm.ldg.global.i.v2i{{64|32}}.p0(ptr %{{.*}}, i32 {{8|16}})
  //CHECK-OPAQUE: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr nonnull %{{.*}}, i32 8)
  auto cached_ul3 = ldg(&in_ul3[0]);
  //CHECK-OPAQUE: tail call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0(ptr %{{.*}}, i32 16)
  auto cached_ull2 = ldg(&in_ull2[0]);
  //CHECK-OPAQUE: tail call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0(ptr %{{.*}}, i32 16)
  //CHECK-OPAQUE: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr nonnull %{{.*}}, i32 8)
  auto cached_ull3 = ldg(&in_ull3[0]);
  //CHECK-OPAQUE: tail call <2 x i{{64|32}}> @llvm.nvvm.ldg.global.i.v2i{{64|32}}.p0(ptr %{{.*}}, i32 {{8|16}})
  //CHECK-OPAQUE: tail call <2 x i{{64|32}}> @llvm.nvvm.ldg.global.i.v2i{{64|32}}.p0(ptr nonnull %{{.*}}, i32 {{8|16}})
  auto cached_ul4 = ldg(&in_ul4[0]);
  //CHECK-OPAQUE: tail call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0(ptr %{{.*}}, i32 16)
  //CHECK-OPAQUE: tail call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0(ptr nonnull %{{.*}}, i32 16)
  auto cached_ull4 = ldg(&in_ull4[0]);

  //CHECK-OPAQUE: tail call <4 x i8> @llvm.nvvm.ldg.global.i.v4i8.p0(ptr %{{.*}}, i32 4)
  auto cached_c4 = ldg(&in_c4[0]);
  //CHECK-OPAQUE: tail call <4 x i8> @llvm.nvvm.ldg.global.i.v4i8.p0(ptr %{{.*}}, i32 4)
  auto cached_sc4 = ldg(&in_sc4[0]);
  //CHECK-OPAQUE: tail call <4 x i16> @llvm.nvvm.ldg.global.i.v4i16.p0(ptr %{{.*}}, i32 8)
  auto cached_s4 = ldg(&in_s4[0]);
  //CHECK-OPAQUE: tail call <4 x i32> @llvm.nvvm.ldg.global.i.v4i32.p0(ptr %{{.*}}, i32 16)
  auto cached_i4 = ldg(&in_i4[0]);

  //CHECK-OPAQUE: tail call <4 x i8> @llvm.nvvm.ldg.global.i.v4i8.p0(ptr %{{.*}}, i32 4)
  auto cached_uc4 = ldg(&in_uc4[0]);
  //CHECK-OPAQUE: tail call <4 x i16> @llvm.nvvm.ldg.global.i.v4i16.p0(ptr %{{.*}}, i32 8)
  auto cached_us4 = ldg(&in_us4[0]);
  //CHECK-OPAQUE: tail call <4 x i32> @llvm.nvvm.ldg.global.i.v4i32.p0(ptr %{{.*}}, i32 16)
  auto cached_ui4 = ldg(&in_ui4[0]);
}