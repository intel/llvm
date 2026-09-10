// Shared test body for free function kernel name collision with different
// signatures. Included by variant-specific .cpp files that define:
//   DLOPEN_FLAGS - dlopen flags (e.g. RTLD_NOW | RTLD_GLOBAL) [Linux only]
//   LIB_A, LIB_B - library file names (e.g. "lib_a.so" or "lib_b.so")
//
// This test verifies that two shared libraries defining free function SYCL
// kernels with the same name but DIFFERENT SIGNATURES work correctly even with
// RTLD_GLOBAL (no symbol interposition because they are different function
// types).

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <cstdio>

#ifdef _WIN32
#define EXPORT_FN extern "C" __declspec(dllexport)
#else
#define EXPORT_FN extern "C" __attribute__((visibility("default")))
#endif

namespace syclexp = sycl::ext::oneapi::experimental;

#ifdef BUILD_LIB_A
// my_kernel in lib_a: takes 2 parameters (out, in)
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void my_kernel(float *out, float *in) {
  size_t i = sycl::ext::oneapi::this_work_item::get_nd_item<1>()
                 .get_global_linear_id();
  out[i] = in[i] + 10.0f;
}

EXPORT_FN int run_kernel_a(sycl::queue *q) {
  constexpr int N = 64;
  float *out = sycl::malloc_device<float>(N, *q);
  float *in = sycl::malloc_device<float>(N, *q);
  q->fill(in, 1.0f, N).wait();
  q->memset(out, 0, N * sizeof(float)).wait();

  q->submit([&](sycl::handler &cgh) {
     syclexp::nd_launch(cgh,
                        sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(N)),
                        syclexp::kernel_function<my_kernel>, out, in);
   }).wait();

  std::vector<float> result(N);
  q->memcpy(result.data(), out, N * sizeof(float)).wait();
  sycl::free(out, *q);
  sycl::free(in, *q);

  // Expected: 1 + 10 = 11
  for (int i = 0; i < N; ++i) {
    if (result[i] != 11.0f)
      return -1;
  }
  return 0;
}
#endif

#ifdef BUILD_LIB_B
// my_kernel in lib_b: takes 3 parameters (out, in1, in2) - DIFFERENT SIGNATURE
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void my_kernel(float *out, float *in1, float *in2) {
  size_t i = sycl::ext::oneapi::this_work_item::get_nd_item<1>()
                 .get_global_linear_id();
  out[i] = in1[i] + in2[i];
}

EXPORT_FN int run_kernel_b(sycl::queue *q) {
  constexpr int N = 64;
  float *out = sycl::malloc_device<float>(N, *q);
  float *in1 = sycl::malloc_device<float>(N, *q);
  float *in2 = sycl::malloc_device<float>(N, *q);
  q->fill(in1, 2.0f, N).wait();
  q->fill(in2, 3.0f, N).wait();
  q->memset(out, 0, N * sizeof(float)).wait();

  q->submit([&](sycl::handler &cgh) {
     syclexp::nd_launch(cgh,
                        sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(N)),
                        syclexp::kernel_function<my_kernel>, out, in1, in2);
   }).wait();

  std::vector<float> result(N);
  q->memcpy(result.data(), out, N * sizeof(float)).wait();
  sycl::free(out, *q);
  sycl::free(in1, *q);
  sycl::free(in2, *q);

  // Expected: 2 + 3 = 5
  for (int i = 0; i < N; ++i) {
    if (result[i] != 5.0f)
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

  auto h_a = loadLib(makePath(LIB_A).c_str());
  if (!h_a)
    return 1;
  auto h_b = loadLib(makePath(LIB_B).c_str());
  if (!h_b)
    return 1;

  auto *fn_a = (fn_t)getSymbol(h_a, "run_kernel_a");
  auto *fn_b = (fn_t)getSymbol(h_b, "run_kernel_b");
  if (!fn_a || !fn_b) {
    std::fprintf(stderr, "Failed to find symbols\n");
    return 1;
  }

  int rc_a = fn_a(&q);
  int rc_b = fn_b(&q);

  std::printf("lib_a kernel (expect 11.0): %s\n", rc_a == 0 ? "PASS" : "FAIL");
  std::printf("lib_b kernel (expect 5.0): %s\n", rc_b == 0 ? "PASS" : "FAIL");

  assert(rc_a == 0 && "lib_a kernel produced wrong results");
  assert(rc_b == 0 && "lib_b kernel produced wrong results");

  unloadLib(h_a);
  unloadLib(h_b);

  return 0;
}
#endif
