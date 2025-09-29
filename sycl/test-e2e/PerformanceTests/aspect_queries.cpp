// RUN: %{build} -O3 -DSYCL_DISABLE_IMAGE_ASPECT_WARNING=1 -o %t.out
// RUN: %{run} %t.out

#include <chrono>
#include <sycl/detail/core.hpp>

struct SimpleTimer {
  using clock = std::chrono::high_resolution_clock;
  SimpleTimer() : start(clock::now()) {}
  ~SimpleTimer() {
    auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() -
                                                                  start)
                 .count();
    std::cout << d << "ns" << " ";
  }

  std::chrono::time_point<clock> start;
};

int main() {
  sycl::device d;

#define ASPECT(A)                                                              \
  {                                                                            \
    std::cout << "aspect::" << #A << ": ";                                     \
    for (int i = 0; i < 10; ++i) {                                             \
      SimpleTimer t;                                                           \
      (void)d.has(sycl::aspect::A);                                            \
    }                                                                          \
    std::cout << std::endl;                                                    \
  }

#define __SYCL_ASPECT(A, ID) ASPECT(A)
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MSG) __SYCL_ASPECT(ASPECT, ID)
#include <sycl/info/aspects.def>
#include <sycl/info/aspects_deprecated.def>
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT
}
