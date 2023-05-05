// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

int main() {

  sycl::queue q{sycl::gpu_selector_v};

  sycl::ext::oneapi::experimental::command_graph g{q.get_context(),
                                                   q.get_device()};

  const size_t n = 10;
  float *arr = sycl::malloc_shared<float>(n, q);

  float pattern = 3.14f;
  auto nodeA = g.add([&](sycl::handler &h) { h.fill(arr, pattern, n); });

  auto executable_graph = g.finalize();

  q.submit([&](sycl::handler &h) {
     h.ext_oneapi_graph(executable_graph);
   }).wait();

  for (int i = 0; i < n; i++)
    assert(arr[i] == pattern);

  sycl::free(arr, q);

  return 0;
}
