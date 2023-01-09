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

  auto *in_f = sycl::malloc_device<float>(1, q);
  auto *in_d = sycl::malloc_device<double>(1, q);

  auto *in_f2 = sycl::malloc_device<float2>(1, q);
  auto *in_d2 = sycl::malloc_device<double2>(1, q);

  auto *in_f4 = sycl::malloc_device<float4>(1, q);

  auto *out_d = sycl::malloc_device<double>(1, q);

  q.wait();

  q.submit([=](sycl::handler &h) {
    h.single_task<class check>([=] {
      //CHECK: tail call float @llvm.nvvm.ldg.global.f.f32.p0f32(float* %0, i32 4)
      //CHECK-OPAQUE: tail call float @llvm.nvvm.ldg.global.f.f32.p0(ptr %0, i32 4)
      auto cached_f = ldg(&in_f[0]);
      //CHECK: tail call double @llvm.nvvm.ldg.global.f.f64.p0f64(double* %1, i32 8)
      //CHECK-OPAQUE: tail call double @llvm.nvvm.ldg.global.f.f64.p0(ptr %1, i32 8)
      auto cached_d = ldg(&in_d[0]);

      //CHECK: tail call <2 x float> @llvm.nvvm.ldg.global.f.v2f32.p0v2f32(<2 x float>* %8, i32 8)
      //CHECK-OPAQUE: tail call <2 x float> @llvm.nvvm.ldg.global.f.v2f32.p0(ptr %2, i32 8)
      auto cached_f2 = ldg(&in_f2[0]);
      //CHECK: tail call <2 x double> @llvm.nvvm.ldg.global.f.v2f64.p0v2f64(<2 x double>* %10, i32 16)
      //CHECK-OPAQUE: tail call <2 x double> @llvm.nvvm.ldg.global.f.v2f64.p0(ptr %3, i32 16)
      auto cached_d2 = ldg(&in_d2[0]);
      //CHECK: tail call <4 x float> @llvm.nvvm.ldg.global.f.v4f32.p0v4f32(<4 x float>* %12, i32 16)
      //CHECK-OPAQUE: tail call <4 x float> @llvm.nvvm.ldg.global.f.v4f32.p0(ptr %4, i32 16)
      auto cached_f4 = ldg(&in_f4[0]);

      out_d[0] =
          cached_d + cached_d2.x() + cached_f + cached_f2.x() + cached_f4.x();
    });
  });

  q.wait();

  free(in_f, q);
  free(in_d, q);

  free(in_f2, q);
  free(in_f4, q);
  free(in_d2, q);

  free(out_d, q);

  return 0;
};
