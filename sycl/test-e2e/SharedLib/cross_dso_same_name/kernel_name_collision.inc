// Shared test body for kernel_name_collision tests.
// Included by variant-specific .cpp files that define:
//   DLOPEN_FLAGS - dlopen flags (e.g. RTLD_NOW | RTLD_LOCAL) [Linux only]
//   KERNEL_CLASS_DECL - how the kernel class is declared (may include
//   namespace)
//   LIB_SMALL, LIB_LARGE - library file names (e.g. "small.so" or "small.dll")
//
// This test verifies that two shared libraries defining SYCL kernels with the
// same class name but different argument layouts work correctly when loaded
// into the same process.

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <cstdio>

#ifdef _WIN32
#define EXPORT_FN extern "C" __declspec(dllexport)
#else
#define EXPORT_FN extern "C" __attribute__((visibility("default")))
#endif

KERNEL_CLASS_DECL

#ifdef BUILD_SMALL_LIB
EXPORT_FN int run_small(sycl::queue *q) {
  constexpr int N = 64;
  float *out = sycl::malloc_device<float>(N, *q);
  float *in = sycl::malloc_device<float>(N, *q);
  q->memset(out, 0, N * sizeof(float)).wait();
  q->fill(in, 1.0f, N).wait();
  float alpha = 2.0f;

  q->submit([&](sycl::handler &cgh) {
     cgh.parallel_for<KernelFunctor>(
         sycl::range<1>(N), [=](sycl::id<1> i) { out[i] = in[i] * alpha; });
   }).wait();

  std::vector<float> result(N);
  q->memcpy(result.data(), out, N * sizeof(float)).wait();
  sycl::free(out, *q);
  sycl::free(in, *q);

  for (int i = 0; i < N; ++i) {
    if (result[i] != 2.0f)
      return -1;
  }
  return 0;
}
#endif

#ifdef BUILD_LARGE_LIB
EXPORT_FN int run_large(sycl::queue *q) {
  constexpr int N = 64;
  float *p0 = sycl::malloc_device<float>(N, *q);
  float *p1 = sycl::malloc_device<float>(N, *q);
  float *p2 = sycl::malloc_device<float>(N, *q);
  float *p3 = sycl::malloc_device<float>(N, *q);
  q->fill(p0, 0.0f, N).wait();
  q->fill(p1, 1.0f, N).wait();
  q->fill(p2, 2.0f, N).wait();
  q->fill(p3, 3.0f, N).wait();
  float s0 = 1.0f, s1 = 1.0f;

  q->submit([&](sycl::handler &cgh) {
     cgh.parallel_for<KernelFunctor>(sycl::range<1>(N), [=](sycl::id<1> i) {
       p0[i] = p1[i] * s0 + p2[i] * s1 + p3[i];
     });
   }).wait();

  std::vector<float> result(N);
  q->memcpy(result.data(), p0, N * sizeof(float)).wait();
  sycl::free(p0, *q);
  sycl::free(p1, *q);
  sycl::free(p2, *q);
  sycl::free(p3, *q);

  // Expected: 1*1 + 2*1 + 3 = 6
  for (int i = 0; i < N; ++i) {
    if (result[i] != 6.0f)
      return -1;
  }
  return 0;
}
#endif

#ifdef BUILD_MAIN
#include "dynlib.inc"
#include <string>

using fn_t = int (*)(sycl::queue *);

static std::string makePath(const char *Lib) {
  return std::string(LIB_DIR) + "/" + Lib;
}

int main() {
  sycl::queue q;

  auto hs = loadLib(makePath(LIB_SMALL).c_str());
  if (!hs)
    return 1;
  auto hl = loadLib(makePath(LIB_LARGE).c_str());
  if (!hl)
    return 1;

  auto *fn_small = (fn_t)getSymbol(hs, "run_small");
  auto *fn_large = (fn_t)getSymbol(hl, "run_large");

  if (!fn_small || !fn_large) {
    std::fprintf(stderr, "Failed to find symbols\n");
    return 1;
  }

  int rc_small = fn_small(&q);
  int rc_large = fn_large(&q);

  std::printf("small kernel: %s\n", rc_small == 0 ? "PASS" : "FAIL");
  std::printf("large kernel: %s\n", rc_large == 0 ? "PASS" : "FAIL");

  assert(rc_small == 0 && "Small kernel failed when both libs are loaded");
  assert(rc_large == 0 && "Large kernel failed when both libs are loaded");

  unloadLib(hs);
  unloadLib(hl);

  return 0;
}
#endif
