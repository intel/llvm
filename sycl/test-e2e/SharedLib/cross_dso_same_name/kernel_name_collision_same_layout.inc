// Shared test body for kernel_name_collision_same_layout tests.
// Included by variant-specific .cpp files that define:
//   DLOPEN_FLAGS - dlopen flags (e.g. RTLD_NOW | RTLD_LOCAL) [Linux only]
//   KERNEL_CLASS_DECL - how the kernel class is declared (may include
//   namespace)
//   LIB_ADD, LIB_MUL - library file names (e.g. "lib_add.so" or "lib_add.dll")
//
// This test verifies that two shared libraries defining SYCL kernels with the
// same class name AND identical argument layouts but different bodies work
// correctly when loaded into the same process.

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

#ifdef BUILD_LIB_ADD
// Same name (KernelFunctor), same captures (two float* pointers), body adds 1.
EXPORT_FN int run_add(sycl::queue *q) {
  constexpr int N = 64;
  float *out = sycl::malloc_device<float>(N, *q);
  float *in = sycl::malloc_device<float>(N, *q);
  q->fill(in, 1.0f, N).wait();
  q->memset(out, 0, N * sizeof(float)).wait();

  q->submit([&](sycl::handler &cgh) {
     cgh.parallel_for<KernelFunctor>(
         sycl::range<1>(N), [=](sycl::id<1> i) { out[i] = in[i] + 1.0f; });
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

#ifdef BUILD_LIB_MUL
// Same name, same captures, body multiplies by 3.
EXPORT_FN int run_mul(sycl::queue *q) {
  constexpr int N = 64;
  float *out = sycl::malloc_device<float>(N, *q);
  float *in = sycl::malloc_device<float>(N, *q);
  q->fill(in, 1.0f, N).wait();
  q->memset(out, 0, N * sizeof(float)).wait();

  q->submit([&](sycl::handler &cgh) {
     cgh.parallel_for<KernelFunctor>(
         sycl::range<1>(N), [=](sycl::id<1> i) { out[i] = in[i] * 3.0f; });
   }).wait();

  std::vector<float> result(N);
  q->memcpy(result.data(), out, N * sizeof(float)).wait();
  sycl::free(out, *q);
  sycl::free(in, *q);

  for (int i = 0; i < N; ++i) {
    if (result[i] != 3.0f)
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

  auto h_add = loadLib(makePath(LIB_ADD).c_str());
  if (!h_add)
    return 1;
  auto h_mul = loadLib(makePath(LIB_MUL).c_str());
  if (!h_mul)
    return 1;

  auto *fn_add = (fn_t)getSymbol(h_add, "run_add");
  auto *fn_mul = (fn_t)getSymbol(h_mul, "run_mul");
  if (!fn_add || !fn_mul) {
    std::fprintf(stderr, "Failed to find symbols\n");
    return 1;
  }

  int rc_add = fn_add(&q);
  int rc_mul = fn_mul(&q);

  std::printf("add kernel (expect 2.0): %s\n", rc_add == 0 ? "PASS" : "FAIL");
  std::printf("mul kernel (expect 3.0): %s\n", rc_mul == 0 ? "PASS" : "FAIL");

  assert(rc_add == 0 && "add kernel produced wrong results");
  assert(rc_mul == 0 && "mul kernel produced wrong results");

  unloadLib(h_add);
  unloadLib(h_mul);

  return 0;
}
#endif
