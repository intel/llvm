// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
#include <CL/sycl.hpp>
#include <iostream>

#include <sycl/ext/oneapi/experimental/graph.hpp>

int main() {

  sycl::property_list properties{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::lazy_execution{}};

  sycl::queue q{sycl::gpu_selector_v, properties};

  sycl::ext::oneapi::experimental::command_graph g;

  const size_t n = 10;
  float *arr = sycl::malloc_shared<float>(n, q);

  g.add([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
      size_t i = idx;
      arr[i] = 1;
    });
  });

  auto result_before_exec1 = arr[0];

  auto executable_graph = g.finalize(q.get_context());

  auto result_before_exec2 = arr[0];

  q.submit([&](sycl::handler &h) { h.ext_oneapi_graph(executable_graph); });

  auto result = arr[0];

  sycl::free(arr, q);

  std::cout << "done.\n";

  return 0;
}
