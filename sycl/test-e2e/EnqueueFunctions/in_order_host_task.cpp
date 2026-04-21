// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue q{{sycl::property::queue::in_order{}}};

  syclex::parallel_for(q, sycl::range{1}, [=](size_t i) {});

  syclex::host_task(q, [=] {});

  q.wait();
}
