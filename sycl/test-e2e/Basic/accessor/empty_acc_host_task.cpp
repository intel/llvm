// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q;
  sycl::buffer<int, 1> b(1);
  q.submit([&](sycl::handler &cgh) {
     sycl::accessor acc{b, cgh, sycl::range<1>{0}, sycl::id<1>{0}};
     assert(acc.empty());
     cgh.host_task([=]() {
       if (!acc.empty())
         acc[0] = 1;
     });
   }).wait_and_throw();
  return 0;
}
