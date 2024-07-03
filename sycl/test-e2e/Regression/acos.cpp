// REQUIRES: aspect-fp64
// UNSUPPORTED: cuda || hip

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
#include <complex>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

using namespace sycl;
using T = std::complex<double>;

int main() {
  queue q;
  const double pi = std::atan2(+0., -0.);
  T data[][2] = {{{-2, -INFINITY}, {pi / 2, INFINITY}},
                 {{-0., INFINITY}, {pi / 2, -INFINITY}}};
  int N = std::size(data);
  auto *p = malloc_shared<T>(N, q);

  q.single_task([=] {
     for (int i = 0; i < N; ++i) {
       p[i] = std::acos(data[i][0]);
     }
   }).wait();

  int fails = 0;
  for (int i = 0; i < N; ++i) {
    auto actual = p[i];
    auto expected = data[i][1];
    if (expected != actual) {
      std::cout << i << " fail:"
                << "expected = " << expected << ", actual = " << actual << "\n";
      ++fails;
    }
  }

  return fails;
}
