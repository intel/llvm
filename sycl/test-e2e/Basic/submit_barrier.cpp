// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <stdlib.h>
#include <sycl/detail/core.hpp>

int main() {

  sycl::device dev{sycl::default_selector_v};
  sycl::queue q{dev};

  q.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel1>([]() {}); });

  sycl::event e = q.ext_oneapi_submit_barrier();
  e.wait_and_throw();

  return 0;
}
