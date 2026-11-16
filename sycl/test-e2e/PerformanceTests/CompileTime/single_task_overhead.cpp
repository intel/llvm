// RUN: python3 %S/../perf_compile.py --name "single_task 400 kernels, device syntax-only" -- %{build} -fsycl-device-only -fsyntax-only
// RUN: python3 %S/../perf_compile.py --name "single_task 400 kernels, device syntax-only, submit" -- %{build} -fsycl-device-only -fsyntax-only -DSUBMIT

// PERF-DESCRIPTION: Device compile time of 400 template-heavy single_tasks.

#include <sycl/detail/core.hpp>

template <int N, int M> struct compile_time_heavy_krn {
  static void foo(int x = N) { compile_time_heavy_krn<N - 1, M>::foo(x); }
};

template <int M> struct compile_time_heavy_krn<0, M> {
  static void foo(int x = 0) { std::ignore = x; }
};

int main() {
  sycl::queue q;
  q.single_task([]() { std::ignore = 42; });
  sycl::detail::loop<2>([&](auto outer_idx) {
    sycl::detail::loop<200>([&](auto idx) {
      auto krn = [=]() {
        compile_time_heavy_krn<idx * 5, outer_idx * 1000 + idx>::foo();
      };
      auto s = [&](sycl::handler &cgh) {
#if SUBMIT
        cgh.single_task(krn);
#endif
      };
#if SUBMIT
      q.submit(s);
#endif
    });
  });
  return 0;
}
