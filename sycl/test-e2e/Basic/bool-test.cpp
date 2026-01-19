// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Check booleans are promoted correctly

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

void run_test(sycl::queue q, bool test, int *res) {
  q.submit([&](sycl::handler &cgh) {
     cgh.single_task([=]() {
       if (test)
         *res = 42;
       else
         *res = -42;
     });
   }).wait();
}

int main() {
  sycl::queue q;
  int *p = sycl::malloc_shared<int>(1, q);
  *p = 0;
  run_test(q, true, p);
  assert(*p == 42);
  *p = 0;
  run_test(q, false, p);
  assert(*p == -42);
  sycl::free(p, q);
}
