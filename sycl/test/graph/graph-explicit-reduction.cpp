// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

int main() {
  sycl::queue q{sycl::gpu_selector_v};

  sycl::ext::oneapi::experimental::command_graph g{q.get_context(),
                                                   q.get_device()};

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

  auto executable_graph = g.finalize();
  q.ext_oneapi_graph(executable_graph).wait();

  assert(*output == 45);

  sycl::free(input, q);
  sycl::free(output, q);

  return 0;
}
