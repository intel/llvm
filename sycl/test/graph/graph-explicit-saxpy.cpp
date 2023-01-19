// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
#include <sycl/sycl.hpp>
#include <iostream>

#include <sycl/ext/oneapi/experimental/graph.hpp>

int main() {

  sycl::property_list properties{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::lazy_execution{}};

  sycl::queue q{sycl::gpu_selector_v, properties};

  sycl::ext::oneapi::experimental::command_graph g;

  const size_t n = 1000;
  const float a = 3.0f;
  float *x = sycl::malloc_device<float>(n, q);
  float *y = sycl::malloc_shared<float>(n, q);

  auto init = g.add([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
      size_t i = idx;
      x[i] = 1.0f;
      y[i] = 2.0f;
    });
  });

  auto compute = g.add([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
      size_t i = idx;
      y[i] = a * x[i] + y[i];
    });
  });

  g.make_edge(init, compute);

  auto executable_graph = g.finalize(q.get_context());

  q.submit([&](sycl::handler &h) { h.ext_oneapi_graph(executable_graph); }).wait();

  sycl::free(x, q);
  sycl::free(y, q);

  std::cout << "done.\n";

  return 0;
}
