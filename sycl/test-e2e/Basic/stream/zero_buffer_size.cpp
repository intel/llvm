// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/stream.hpp>

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
     sycl::stream s{0, 0, cgh};
     cgh.single_task([=]() { s << 42 << sycl::flush; });
   }).wait_and_throw();
  return 0;
}
