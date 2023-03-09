// REQUIRES: cuda

// RUN: %clangxx -Xclang -no-opaque-pointers -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -S -Xclang -emit-llvm %s -o -| FileCheck %s
// RUN: %clangxx -Xclang -opaque-pointers -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -S -Xclang -emit-llvm %s -o -| FileCheck %s --check-prefixes=CHECK-OPAQUE

#include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::cuda;
using namespace sycl::ext::oneapi::experimental;

int main() {

  sycl::queue q;

  auto *in_c = sycl::malloc_device<char>(1, q);
  auto *in_s = sycl::malloc_device<short>(1, q);
  auto *in_i = sycl::malloc_device<int>(1, q);
  auto *in_l = sycl::malloc_device<long>(1, q);
  auto *in_ll = sycl::malloc_device<long long>(1, q);

  auto *in_uc = sycl::malloc_device<unsigned char>(1, q);
  auto *in_us = sycl::malloc_device<unsigned short>(1, q);
  auto *in_ui = sycl::malloc_device<unsigned int>(1, q);
  auto *in_ul = sycl::malloc_device<unsigned long>(1, q);
  auto *in_ull = sycl::malloc_device<unsigned long long>(1, q);

  auto *in_c2 = sycl::malloc_device<char2>(1, q);
  auto *in_s2 = sycl::malloc_device<short2>(1, q);
  auto *in_i2 = sycl::malloc_device<int2>(1, q);
  auto *in_ll2 = sycl::malloc_device<longlong2>(1, q);

  auto *in_c4 = sycl::malloc_device<char4>(1, q);
  auto *in_s4 = sycl::malloc_device<short4>(1, q);
  auto *in_i4 = sycl::malloc_device<int4>(1, q);

  auto *in_uc2 = sycl::malloc_device<uchar2>(1, q);
  auto *in_us2 = sycl::malloc_device<ushort2>(1, q);
  auto *in_ui2 = sycl::malloc_device<uint2>(1, q);
  auto *in_ull2 = sycl::malloc_device<ulonglong2>(1, q);

  auto *in_uc4 = sycl::malloc_device<uchar4>(1, q);
  auto *in_us4 = sycl::malloc_device<ushort4>(1, q);
  auto *in_ui4 = sycl::malloc_device<uint4>(1, q);

  auto *in_f = sycl::malloc_device<float>(1, q);
  auto *in_d = sycl::malloc_device<double>(1, q);

  auto *in_f2 = sycl::malloc_device<float2>(1, q);
  auto *in_d2 = sycl::malloc_device<double2>(1, q);

  auto *in_f4 = sycl::malloc_device<float4>(1, q);

  auto *out_d = sycl::malloc_device<double>(1, q);

  q.wait();

  q.submit([=](sycl::handler &h) {
    h.single_task<class check>([=] {
      //CHECK: tail call float @llvm.nvvm.ldg.global.f.f32.p0f32(float* %{{.*}}, i32 4)
      //CHECK-OPAQUE: tail call float @llvm.nvvm.ldg.global.f.f32.p0(ptr %{{.*}}, i32 4)
      auto cached_f = ldg(&in_f[0]);
      //CHECK: tail call double @llvm.nvvm.ldg.global.f.f64.p0f64(double* %{{.*}}, i32 8)
      //CHECK-OPAQUE: tail call double @llvm.nvvm.ldg.global.f.f64.p0(ptr %{{.*}}, i32 8)
      auto cached_d = ldg(&in_d[0]);

      //CHECK: tail call <2 x float> @llvm.nvvm.ldg.global.f.v2f32.p0v2f32(<2 x float>* %{{.*}}, i32 8)
      //CHECK-OPAQUE: tail call <2 x float> @llvm.nvvm.ldg.global.f.v2f32.p0(ptr %{{.*}}, i32 8)
      auto cached_f2 = ldg(&in_f2[0]);
      //CHECK: tail call <2 x double> @llvm.nvvm.ldg.global.f.v2f64.p0v2f64(<2 x double>* %{{.*}}, i32 16)
      //CHECK-OPAQUE: tail call <2 x double> @llvm.nvvm.ldg.global.f.v2f64.p0(ptr %{{.*}}, i32 16)
      auto cached_d2 = ldg(&in_d2[0]);
      //CHECK: tail call <4 x float> @llvm.nvvm.ldg.global.f.v4f32.p0v4f32(<4 x float>* %{{.*}}, i32 16)
      //CHECK-OPAQUE: tail call <4 x float> @llvm.nvvm.ldg.global.f.v4f32.p0(ptr %{{.*}}, i32 16)
      auto cached_f4 = ldg(&in_f4[0]);

      // Unsigned variants are identical to signed variants, but this leads to
      // correct behavior.

      //CHECK: tail call i8 @llvm.nvvm.ldg.global.i.i8.p0i8(i8* %{{.*}}, i32 1)
      //CHECK-OPAQUE: tail call i8 @llvm.nvvm.ldg.global.i.i8.p0(ptr %{{.*}}, i32 1)
      auto cached_c = ldg(&in_c[0]);
      //CHECK: tail call i16 @llvm.nvvm.ldg.global.i.i16.p0i16(i16* %{{.*}}, i32 2)
      //CHECK-OPAQUE: tail call i16 @llvm.nvvm.ldg.global.i.i16.p0(ptr %{{.*}}, i32 2)
      auto cached_s = ldg(&in_s[0]);
      //CHECK: tail call i32 @llvm.nvvm.ldg.global.i.i32.p0i32(i32* %{{.*}}, i32 4)
      //CHECK-OPAQUE: tail call i32 @llvm.nvvm.ldg.global.i.i32.p0(ptr %{{.*}}, i32 4)
      auto cached_i = ldg(&in_i[0]);
      //CHECK: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0i64(i64* %{{.*}}, i32 8)
      //CHECK-OPAQUE: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr %{{.*}}, i32 8)
      auto cached_l = ldg(&in_l[0]);
      //CHECK: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0i64(i64* %{{.*}}, i32 8)
      //CHECK-OPAQUE: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr %{{.*}}, i32 8)
      auto cached_ll = ldg(&in_ll[0]);
      //CHECK: tail call i8 @llvm.nvvm.ldg.global.i.i8.p0i8(i8* %{{.*}}, i32 1)
      //CHECK-OPAQUE: tail call i8 @llvm.nvvm.ldg.global.i.i8.p0(ptr %{{.*}}, i32 1)
      auto cached_uc = ldg(&in_uc[0]);
      //CHECK: tail call i16 @llvm.nvvm.ldg.global.i.i16.p0i16(i16* %{{.*}}, i32 2)
      //CHECK-OPAQUE: tail call i16 @llvm.nvvm.ldg.global.i.i16.p0(ptr %{{.*}}, i32 2)
      auto cached_us = ldg(&in_us[0]);
      //CHECK: tail call i32 @llvm.nvvm.ldg.global.i.i32.p0i32(i32* %{{.*}}, i32 4)
      //CHECK-OPAQUE: tail call i32 @llvm.nvvm.ldg.global.i.i32.p0(ptr %{{.*}}, i32 4)
      auto cached_ui = ldg(&in_ui[0]);
      //CHECK: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0i64(i64* %{{.*}}, i32 8)
      //CHECK-OPAQUE: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr %{{.*}}, i32 8)
      auto cached_ul = ldg(&in_ul[0]);
      //CHECK: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0i64(i64* %{{.*}}, i32 8)
      //CHECK-OPAQUE: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr %{{.*}}, i32 8)
      auto cached_ull = ldg(&in_ull[0]);

      //CHECK: tail call <2 x i8> @llvm.nvvm.ldg.global.i.v2i8.p0v2i8(<2 x i8>* %{{.*}}, i32 2)
      //CHECK-OPAQUE: tail call <2 x i8> @llvm.nvvm.ldg.global.i.v2i8.p0(ptr %{{.*}}, i32 2)
      auto cached_c2 = ldg(&in_c2[0]);
      //CHECK: tail call <2 x i16> @llvm.nvvm.ldg.global.i.v2i16.p0v2i16(<2 x i16>* %{{.*}}, i32 4)
      //CHECK-OPAQUE: tail call <2 x i16> @llvm.nvvm.ldg.global.i.v2i16.p0(ptr %{{.*}}, i32 4)
      auto cached_s2 = ldg(&in_s2[0]);
      //CHECK: tail call <2 x i32> @llvm.nvvm.ldg.global.i.v2i32.p0v2i32(<2 x i32>* %{{.*}}, i32 8)
      //CHECK-OPAQUE: tail call <2 x i32> @llvm.nvvm.ldg.global.i.v2i32.p0(ptr %{{.*}}, i32 8)
      auto cached_i2 = ldg(&in_i2[0]);
      //CHECK: tail call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0v2i64(<2 x i64>* %{{.*}}, i32 16)
      //CHECK-OPAQUE: tail call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0(ptr %{{.*}}, i32 16)
      auto cached_ll2 = ldg(&in_ll2[0]);
      //CHECK: tail call <2 x i8> @llvm.nvvm.ldg.global.i.v2i8.p0v2i8(<2 x i8>* %{{.*}}, i32 2)
      //CHECK-OPAQUE: tail call <2 x i8> @llvm.nvvm.ldg.global.i.v2i8.p0(ptr %{{.*}}, i32 2)
      auto cached_uc2 = ldg(&in_uc2[0]);
      //CHECK: tail call <2 x i16> @llvm.nvvm.ldg.global.i.v2i16.p0v2i16(<2 x i16>* %{{.*}}, i32 4)
      //CHECK-OPAQUE: tail call <2 x i16> @llvm.nvvm.ldg.global.i.v2i16.p0(ptr %{{.*}}, i32 4)
      auto cached_us2 = ldg(&in_us2[0]);
      //CHECK: tail call <2 x i32> @llvm.nvvm.ldg.global.i.v2i32.p0v2i32(<2 x i32>* %{{.*}}, i32 8)
      //CHECK-OPAQUE: tail call <2 x i32> @llvm.nvvm.ldg.global.i.v2i32.p0(ptr %{{.*}}, i32 8)
      auto cached_ui2 = ldg(&in_ui2[0]);
      //CHECK: tail call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0v2i64(<2 x i64>* %{{.*}}, i32 16)
      //CHECK-OPAQUE: tail call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0(ptr %{{.*}}, i32 16)
      auto cached_ull2 = ldg(&in_ull2[0]);

      //CHECK: tail call <4 x i8> @llvm.nvvm.ldg.global.i.v4i8.p0v4i8(<4 x i8>* %{{.*}}, i32 4)
      //CHECK-OPAQUE: tail call <4 x i8> @llvm.nvvm.ldg.global.i.v4i8.p0(ptr %{{.*}}, i32 4)
      auto cached_c4 = ldg(&in_c4[0]);
      //CHECK: tail call <4 x i16> @llvm.nvvm.ldg.global.i.v4i16.p0v4i16(<4 x i16>* %{{.*}}, i32 8)
      //CHECK-OPAQUE: tail call <4 x i16> @llvm.nvvm.ldg.global.i.v4i16.p0(ptr %{{.*}}, i32 8)
      auto cached_s4 = ldg(&in_s4[0]);
      //CHECK: tail call <4 x i32> @llvm.nvvm.ldg.global.i.v4i32.p0v4i32(<4 x i32>* %{{.*}}, i32 16)
      //CHECK-OPAQUE: tail call <4 x i32> @llvm.nvvm.ldg.global.i.v4i32.p0(ptr %{{.*}}, i32 16)
      auto cached_i4 = ldg(&in_i4[0]);

      //CHECK: tail call <4 x i8> @llvm.nvvm.ldg.global.i.v4i8.p0v4i8(<4 x i8>* %{{.*}}, i32 4)
      //CHECK-OPAQUE: tail call <4 x i8> @llvm.nvvm.ldg.global.i.v4i8.p0(ptr %{{.*}}, i32 4)
      auto cached_uc4 = ldg(&in_uc4[0]);
      //CHECK: tail call <4 x i16> @llvm.nvvm.ldg.global.i.v4i16.p0v4i16(<4 x i16>* %{{.*}}, i32 8)
      //CHECK-OPAQUE: tail call <4 x i16> @llvm.nvvm.ldg.global.i.v4i16.p0(ptr %{{.*}}, i32 8)
      auto cached_us4 = ldg(&in_us4[0]);
      //CHECK: tail call <4 x i32> @llvm.nvvm.ldg.global.i.v4i32.p0v4i32(<4 x i32>* %{{.*}}, i32 16)
      //CHECK-OPAQUE: tail call <4 x i32> @llvm.nvvm.ldg.global.i.v4i32.p0(ptr %{{.*}}, i32 16)
      auto cached_ui4 = ldg(&in_ui4[0]);
    });
  });

  q.wait();

  free(in_f, q);
  free(in_d, q);
  free(in_f2, q);
  free(in_f4, q);
  free(in_d2, q);
  free(in_c, q);
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
  free(in_s2, q);
  free(in_i2, q);
  free(in_ll2, q);
  free(in_uc2, q);
  free(in_us2, q);
  free(in_ui2, q);
  free(in_ull2, q);
  free(in_c4, q);
  free(in_s4, q);
  free(in_i4, q);
  free(in_uc4, q);
  free(in_us4, q);
  free(in_ui4, q);

  return 0;
};
