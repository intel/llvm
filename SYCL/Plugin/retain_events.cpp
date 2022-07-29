// REQUIRES: gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <sycl/sycl.hpp>

namespace {

void print_device_properties(sycl::device const &dev) {
  auto plat_name =
      dev.get_platform().template get_info<sycl::info::platform::name>();
  auto dev_name = dev.template get_info<sycl::info::device::name>();
  auto driver_version =
      dev.template get_info<sycl::info::device::driver_version>();

  fprintf(stdout, " platform name: %s\n", plat_name.c_str());
  fprintf(stdout, "   device name: %s\n", dev_name.c_str());
  fprintf(stdout, "driver version: %s\n", driver_version.c_str());
}

void async_sycl_error(sycl::exception_list el) {
  fprintf(stderr, "async exceptions caught:\n");
  for (auto l = el.begin(); l != el.end(); ++l) {
    try {
      std::rethrow_exception(*l);
    } catch (const sycl::exception &e) {
      fprintf(stderr, "what: %s code: %d\n", e.what(), e.get_cl_code());
      std::exit(-1);
    }
  }
}

} // namespace

int test(sycl::device &D) {
  sycl::context C(D);
  constexpr size_t N = 1024 * 1000;

  int *src = sycl::malloc_shared<int>(N, D, C);
  int *dst = sycl::malloc_shared<int>(N, D, C);

  std::iota(src, src + N, 0);
  std::fill(dst, dst + N, 0);

  for (int i = 0; i < 256; ++i) {
    sycl::queue Q{
        C, D,
        async_sycl_error}; //, sycl::property::queue::enable_profiling() };

    Q.submit([&](sycl::handler &SH) {
      sycl::range<1> r(N);
      SH.parallel_for<class X>(r, [=](sycl::id<1> id) {
        size_t i = id.get(0);
        dst[i] = src[i] + 1;
      });
    });
  }

  sycl::free(dst, C);
  sycl::free(src, C);

  return 0;
}

int main() {
  try {
    sycl::device D1{sycl::gpu_selector{}};
    print_device_properties(D1);
    for (int i = 0; i < 10; ++i) {
      test(D1);
    }
  } catch (sycl::exception &e) {
    fprintf(stderr, "...sycl failed to entertain with \"%s\" (%d) \n", e.what(),
            e.get_cl_code());
  }
  return 0;
}
