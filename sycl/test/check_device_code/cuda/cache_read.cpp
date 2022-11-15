// REQUIRES: cuda

// RUN: %clangxx -Xclang -no-opaque-pointers -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -S -Xclang -emit-llvm %s -o -| FileCheck %s
// RUN: %clangxx -Xclang -opaque-pointers -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -S -Xclang -emit-llvm %s -o -| FileCheck %s --check-prefixes=CHECK-OPAQUE

#include <sycl/ext/oneapi/experimental/bfloat16.hpp>
#include <sycl/ext/oneapi/experimental/cuda/cache_read.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::cuda;
using namespace sycl::ext::oneapi::experimental;

int main() {

  sycl::queue q;

  auto *in_bfloat16 = sycl::malloc_device<bfloat16>(1, q);

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

  auto *in_f = sycl::malloc_device<float>(1, q);
  auto *in_d = sycl::malloc_device<double>(1, q);

  auto *out_d = sycl::malloc_device<double>(1, q);

  q.wait();

  q.submit([=](sycl::handler &h) {
    h.single_task<class ldg>([=] {

      //CHECK: tail call i16 @llvm.nvvm.ldg.global.i.i16.p0i16(i16* %14, i32 2)
      //CHECK-OPAQUE: tail call i16 @llvm.nvvm.ldg.global.i.i16.p0(ptr %0, i32 2)
      auto cached_bfloat16 = cache_read(&in_bfloat16[0]);
      //CHECK: tail call i8 @llvm.nvvm.ldg.global.i.i8.p0i8(i8* %1, i32 1)
      //CHECK-OPAQUE: tail call i8 @llvm.nvvm.ldg.global.i.i8.p0(ptr %1, i32 1)
      auto cached_c = cache_read(&in_c[0]);
      //CHECK: tail call i16 @llvm.nvvm.ldg.global.i.i16.p0i16(i16* %2, i32 2)
      //CHECK-OPAQUE: tail call i16 @llvm.nvvm.ldg.global.i.i16.p0(ptr %2, i32 2)
      auto cached_s = cache_read(&in_s[0]);
      //CHECK: tail call i32 @llvm.nvvm.ldg.global.i.i32.p0i32(i32* %3, i32 4)
      //CHECK-OPAQUE: tail call i32 @llvm.nvvm.ldg.global.i.i32.p0(ptr %3, i32 4)
      auto cached_i = cache_read(&in_i[0]);
      //CHECK: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0i64(i64* %4, i32 8)
      //CHECK-OPAQUE: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr %4, i32 8)
      auto cached_l = cache_read(&in_l[0]);
      //CHECK: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0i64(i64* %5, i32 8)
      //CHECK-OPAQUE: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr %5, i32 8)
      auto cached_ll = cache_read(&in_ll[0]);
      //CHECK: tail call i8 @llvm.nvvm.ldg.global.i.i8.p0i8(i8* %6, i32 1)
      //CHECK-OPAQUE: tail call i8 @llvm.nvvm.ldg.global.i.i8.p0(ptr %6, i32 1)
      auto cached_uc = cache_read(&in_uc[0]);
      //CHECK: tail call i16 @llvm.nvvm.ldg.global.i.i16.p0i16(i16* %7, i32 2)
      //CHECK-OPAQUE: tail call i16 @llvm.nvvm.ldg.global.i.i16.p0(ptr %7, i32 2)
      auto cached_us = cache_read(&in_us[0]);
      //CHECK: tail call i32 @llvm.nvvm.ldg.global.i.i32.p0i32(i32* %8, i32 4)
      //CHECK-OPAQUE: tail call i32 @llvm.nvvm.ldg.global.i.i32.p0(ptr %8, i32 4)
      auto cached_ui = cache_read(&in_ui[0]);
      //CHECK: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0i64(i64* %9, i32 8)
      //CHECK-OPAQUE: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr %9, i32 8)
      auto cached_ul = cache_read(&in_ul[0]);
      //CHECK: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0i64(i64* %10, i32 8)
      //CHECK-OPAQUE: tail call i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr %10, i32 8)
      auto cached_ull = cache_read(&in_ull[0]);
      //CHECK: tail call float @llvm.nvvm.ldg.global.f.f32.p0f32(float* %11, i32 4)
      //CHECK-OPAQUE: tail call float @llvm.nvvm.ldg.global.f.f32.p0(ptr %11, i32 4)
      auto cached_f = cache_read(&in_f[0]);
      //CHECK: tail call double @llvm.nvvm.ldg.global.f.f64.p0f64(double* %12, i32 8)
      //CHECK-OPAQUE: tail call double @llvm.nvvm.ldg.global.f.f64.p0(ptr %12, i32 8)
      auto cached_d = cache_read(&in_d[0]);

      out_d[0] = cached_bfloat16;
      out_d[0] += cached_d + cached_f + cached_c + cached_s + cached_i +
                  cached_l + cached_ll + cached_uc + cached_us + cached_ui +
                  cached_ul + cached_ull;
    });
  });

  q.wait();

  free(in_bfloat16, q);

  free(in_c);
  free(in_s);
  free(in_i);
  free(in_l);
  free(in_ll);

  free(in_uc);
  free(in_us);
  free(in_ui);
  free(in_ul);
  free(in_ull);

  free(in_d);
  free(in_f);

  free(out_d, q);

  return 0;
};
