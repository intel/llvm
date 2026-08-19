// Shared test body for kernel_name_collision_kernel_bundle tests.
// Included by variant-specific .cpp files that define:
//   DLOPEN_FLAGS - dlopen flags (e.g. RTLD_NOW | RTLD_LOCAL) [Linux only]
//   KERNEL_CLASS_DECL - how the kernel class is declared (may include
//   namespace)
//   LIB_A, LIB_B - library file names (e.g. "lib_a.so" or "lib_a.dll")
//
// This test verifies that invoking a kernel via kernel_bundle from within each
// DSO (and main) runs that DSO's own version of the kernel. Three compilation
// units define the same KernelFunctor name with identical layout but different
// bodies (+10, +20, +30).

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <cstdio>
#include <vector>

#ifdef _WIN32
#define EXPORT_FN extern "C" __declspec(dllexport)
#else
#define EXPORT_FN extern "C" __attribute__((visibility("default")))
#endif

KERNEL_CLASS_DECL

#ifdef BUILD_LIB_A
// KernelFunctor in lib_a: out[i] = in[i] + 10
EXPORT_FN int run_via_bundle_a(sycl::queue *q) {
  constexpr int N = 64;
  float *out = sycl::malloc_device<float>(N, *q);
  float *in = sycl::malloc_device<float>(N, *q);
  q->fill(in, 1.0f, N).wait();
  q->memset(out, 0, N * sizeof(float)).wait();

  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      q->get_context(), {q->get_device()},
      {sycl::get_kernel_id<KernelFunctor>()});

  q->submit([&](sycl::handler &cgh) {
     cgh.use_kernel_bundle(bundle);
     cgh.parallel_for<KernelFunctor>(
         sycl::range<1>(N), [=](sycl::id<1> i) { out[i] = in[i] + 10.0f; });
   }).wait();

  std::vector<float> result(N);
  q->memcpy(result.data(), out, N * sizeof(float)).wait();
  sycl::free(out, *q);
  sycl::free(in, *q);

  for (int i = 0; i < N; ++i) {
    if (result[i] != 11.0f) {
      std::fprintf(stderr, "run_via_bundle_a: result[%d] = %f, expected 11.0\n",
                   i, result[i]);
      return -1;
    }
  }
  return 0;
}
#endif

#ifdef BUILD_LIB_B
// KernelFunctor in lib_b: out[i] = in[i] + 20
EXPORT_FN int run_via_bundle_b(sycl::queue *q) {
  constexpr int N = 64;
  float *out = sycl::malloc_device<float>(N, *q);
  float *in = sycl::malloc_device<float>(N, *q);
  q->fill(in, 1.0f, N).wait();
  q->memset(out, 0, N * sizeof(float)).wait();

  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      q->get_context(), {q->get_device()},
      {sycl::get_kernel_id<KernelFunctor>()});

  q->submit([&](sycl::handler &cgh) {
     cgh.use_kernel_bundle(bundle);
     cgh.parallel_for<KernelFunctor>(
         sycl::range<1>(N), [=](sycl::id<1> i) { out[i] = in[i] + 20.0f; });
   }).wait();

  std::vector<float> result(N);
  q->memcpy(result.data(), out, N * sizeof(float)).wait();
  sycl::free(out, *q);
  sycl::free(in, *q);

  for (int i = 0; i < N; ++i) {
    if (result[i] != 21.0f) {
      std::fprintf(stderr, "run_via_bundle_b: result[%d] = %f, expected 21.0\n",
                   i, result[i]);
      return -1;
    }
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

// KernelFunctor in main: out[i] = in[i] + 30
int run_via_bundle_main(sycl::queue *q) {
  constexpr int N = 64;
  float *out = sycl::malloc_device<float>(N, *q);
  float *in = sycl::malloc_device<float>(N, *q);
  q->fill(in, 1.0f, N).wait();
  q->memset(out, 0, N * sizeof(float)).wait();

  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      q->get_context(), {q->get_device()},
      {sycl::get_kernel_id<KernelFunctor>()});

  q->submit([&](sycl::handler &cgh) {
     cgh.use_kernel_bundle(bundle);
     cgh.parallel_for<KernelFunctor>(
         sycl::range<1>(N), [=](sycl::id<1> i) { out[i] = in[i] + 30.0f; });
   }).wait();

  std::vector<float> result(N);
  q->memcpy(result.data(), out, N * sizeof(float)).wait();
  sycl::free(out, *q);
  sycl::free(in, *q);

  for (int i = 0; i < N; ++i) {
    if (result[i] != 31.0f) {
      std::fprintf(stderr,
                   "run_via_bundle_main: result[%d] = %f, expected 31.0\n", i,
                   result[i]);
      return -1;
    }
  }
  return 0;
}

int main() {
  sycl::queue q;

  auto h_a = loadLib(makePath(LIB_A).c_str());
  if (!h_a)
    return 1;
  auto h_b = loadLib(makePath(LIB_B).c_str());
  if (!h_b)
    return 1;

  auto *fn_a = (fn_t)getSymbol(h_a, "run_via_bundle_a");
  auto *fn_b = (fn_t)getSymbol(h_b, "run_via_bundle_b");
  if (!fn_a || !fn_b) {
    std::fprintf(stderr, "Failed to find symbols\n");
    return 1;
  }

  int rc_a = fn_a(&q);
  int rc_b = fn_b(&q);
  int rc_main = run_via_bundle_main(&q);

  std::printf("lib_a via bundle (expect 11.0): %s\n",
              rc_a == 0 ? "PASS" : "FAIL");
  std::printf("lib_b via bundle (expect 21.0): %s\n",
              rc_b == 0 ? "PASS" : "FAIL");
  std::printf("main via bundle (expect 31.0): %s\n",
              rc_main == 0 ? "PASS" : "FAIL");

  assert(rc_a == 0 && "lib_a kernel_bundle invocation ran wrong kernel");
  assert(rc_b == 0 && "lib_b kernel_bundle invocation ran wrong kernel");
  assert(rc_main == 0 && "main kernel_bundle invocation ran wrong kernel");

  unloadLib(h_a);
  unloadLib(h_b);

  return 0;
}
#endif
