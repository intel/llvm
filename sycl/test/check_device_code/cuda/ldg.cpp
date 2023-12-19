// REQUIRES: cuda

// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xclang -fnative-half-type -S -Xclang -emit-llvm %s -o -| FileCheck %s --check-prefixes=CHECK-OPAQUE

#include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::cuda;
using namespace sycl::ext::oneapi::experimental;

int main() {

  sycl::queue q;

  auto *in_c = sycl::malloc_device<char>(1, q);
  auto *in_sc = sycl::malloc_device<signed char>(1, q);
  auto *in_s = sycl::malloc_device<short>(1, q);
  auto *in_i = sycl::malloc_device<int>(1, q);
  auto *in_l = sycl::malloc_device<long>(1, q);
  auto *in_ll = sycl::malloc_device<long long>(1, q);

  auto *in_uc = sycl::malloc_device<unsigned char>(1, q);
  auto *in_us = sycl::malloc_device<unsigned short>(1, q);
  auto *in_ui = sycl::malloc_device<unsigned int>(1, q);
  auto *in_ul = sycl::malloc_device<unsigned long>(1, q);
  auto *in_ull = sycl::malloc_device<unsigned long long>(1, q);

  auto *in_c2 = sycl::malloc_device<sycl::vec<char, 2>>(1, q);
  auto *in_c3 = sycl::malloc_device<sycl::vec<char, 3>>(1, q);
  auto *in_sc2 = sycl::malloc_device<sycl::vec<signed char, 2>>(1, q);
  auto *in_sc3 = sycl::malloc_device<sycl::vec<signed char, 3>>(1, q);
  auto *in_s2 = sycl::malloc_device<sycl::vec<short, 2>>(1, q);
  auto *in_s3 = sycl::malloc_device<sycl::vec<short, 3>>(1, q);
  auto *in_i2 = sycl::malloc_device<sycl::vec<int, 2>>(1, q);
  auto *in_i3 = sycl::malloc_device<sycl::vec<int, 3>>(1, q);
  auto *in_l2 = sycl::malloc_device<sycl::vec<long, 2>>(1, q);
  auto *in_l3 = sycl::malloc_device<sycl::vec<long, 3>>(1, q);
  auto *in_ll2 = sycl::malloc_device<sycl::vec<long long, 2>>(1, q);
  auto *in_ll3 = sycl::malloc_device<sycl::vec<long long, 3>>(1, q);
  auto *in_l4 = sycl::malloc_device<sycl::vec<long, 4>>(1, q);
  auto *in_ll4 = sycl::malloc_device<sycl::vec<long long, 4>>(1, q);

  auto *in_c4 = sycl::malloc_device<sycl::vec<char, 4>>(1, q);
  auto *in_sc4 = sycl::malloc_device<sycl::vec<signed char, 4>>(1, q);
  auto *in_s4 = sycl::malloc_device<sycl::vec<short, 4>>(1, q);
  auto *in_i4 = sycl::malloc_device<sycl::vec<int, 4>>(1, q);

  auto *in_uc2 = sycl::malloc_device<sycl::vec<unsigned char, 2>>(1, q);
  auto *in_uc3 = sycl::malloc_device<sycl::vec<unsigned char, 3>>(1, q);
  auto *in_us2 = sycl::malloc_device<sycl::vec<unsigned short, 2>>(1, q);
  auto *in_us3 = sycl::malloc_device<sycl::vec<unsigned short, 3>>(1, q);
  auto *in_ui2 = sycl::malloc_device<sycl::vec<unsigned int, 2>>(1, q);
  auto *in_ui3 = sycl::malloc_device<sycl::vec<unsigned int, 3>>(1, q);
  auto *in_ul2 = sycl::malloc_device<sycl::vec<unsigned long, 2>>(1, q);
  auto *in_ul3 = sycl::malloc_device<sycl::vec<unsigned long, 3>>(1, q);
  auto *in_ull2 = sycl::malloc_device<sycl::vec<unsigned long long, 2>>(1, q);
  auto *in_ull3 = sycl::malloc_device<sycl::vec<unsigned long long, 3>>(1, q);
  auto *in_ul4 = sycl::malloc_device<sycl::vec<unsigned long, 4>>(1, q);
  auto *in_ull4 = sycl::malloc_device<sycl::vec<unsigned long long, 4>>(1, q);

  auto *in_uc4 = sycl::malloc_device<sycl::vec<unsigned char, 4>>(1, q);
  auto *in_us4 = sycl::malloc_device<sycl::vec<unsigned short, 4>>(1, q);
  auto *in_ui4 = sycl::malloc_device<sycl::vec<unsigned int, 4>>(1, q);

  auto *in_h = sycl::malloc_device<half>(1, q);
  auto *in_f = sycl::malloc_device<float>(1, q);
  auto *in_d = sycl::malloc_device<double>(1, q);

  auto *in_h2 = sycl::malloc_device<sycl::vec<half, 2>>(1, q);
  auto *in_h3 = sycl::malloc_device<sycl::vec<half, 3>>(1, q);
  auto *in_h4 = sycl::malloc_device<sycl::vec<half, 4>>(1, q);
  auto *in_f2 = sycl::malloc_device<sycl::vec<float, 2>>(1, q);
  auto *in_f3 = sycl::malloc_device<sycl::vec<float, 3>>(1, q);
  auto *in_f4 = sycl::malloc_device<sycl::vec<float, 4>>(1, q);
  auto *in_d2 = sycl::malloc_device<sycl::vec<double, 2>>(1, q);
  auto *in_d3 = sycl::malloc_device<sycl::vec<double, 3>>(1, q);
  auto *in_d4 = sycl::malloc_device<sycl::vec<double, 4>>(1, q);

  q.wait();

  q.submit([=](sycl::handler &h) {
    h.single_task<class check>([=] {
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
    });
  });

  q.wait();

  free(in_h, q);
  free(in_f, q);
  free(in_d, q);
  free(in_h2, q);
  free(in_h3, q);
  free(in_h4, q);
  free(in_f2, q);
  free(in_f3, q);
  free(in_f4, q);
  free(in_d2, q);
  free(in_d3, q);
  free(in_d4, q);
  free(in_c, q);
  free(in_sc, q);
  free(in_s, q);
  free(in_i, q);
  free(in_l, q);
  free(in_ll, q);
  free(in_uc, q);
  free(in_us, q);
  free(in_ui, q);
  free(in_ul, q);
  free(in_ull, q);
  free(in_c2, q);
  free(in_c3, q);
  free(in_sc2, q);
  free(in_sc3, q);
  free(in_s2, q);
  free(in_s3, q);
  free(in_i2, q);
  free(in_i3, q);
  free(in_l2, q);
  free(in_l3, q);
  free(in_ll2, q);
  free(in_ll3, q);
  free(in_l4, q);
  free(in_ll4, q);
  free(in_uc2, q);
  free(in_uc3, q);
  free(in_us2, q);
  free(in_us3, q);
  free(in_ui2, q);
  free(in_ui3, q);
  free(in_ul2, q);
  free(in_ul3, q);
  free(in_ull2, q);
  free(in_ull3, q);
  free(in_ul4, q);
  free(in_ull4, q);
  free(in_c4, q);
  free(in_sc4, q);
  free(in_s4, q);
  free(in_i4, q);
  free(in_uc4, q);
  free(in_us4, q);
  free(in_ui4, q);

  return 0;
};
