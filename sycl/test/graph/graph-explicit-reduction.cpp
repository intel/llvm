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
  float *input = sycl::malloc_shared<float>(n, q);
  float *output = sycl::malloc_shared<float>(1, q);
  for (size_t i = 0; i < n; i++) {
    input[i] = i;
  }

  auto e = q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n},
                   sycl::reduction(output, 0.0f, std::plus()),
                   [=](sycl::id<1> idx, auto &sum) { sum += input[idx]; });
  });

  e.wait();

  sycl::free(input, q);
  sycl::free(output, q);

  std::cout << "done\n";

  return 0;
}
