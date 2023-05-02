// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
#include <iostream>
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

int main() {

  sycl::queue q{sycl::gpu_selector_v};

  sycl::ext::oneapi::experimental::command_graph g{q.get_context(),
                                                   q.get_device()};

  const size_t n = 10;
  float *arr = sycl::malloc_shared<float>(n, q);
  for (int i = 0; i < n; i++) {
    arr[i] = 0;
  }

  g.add([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
      size_t i = idx;
      arr[i] = 3.14f;
    });
  });

  for (int i = 0; i < n; i++) {
    assert(arr[i] == 0);
  }

  auto executable_graph = g.finalize();

  for (int i = 0; i < n; i++) {
    assert(arr[i] == 0);
  }

  q.submit([&](sycl::handler &h) {
     h.ext_oneapi_graph(executable_graph);
   }).wait();

  for (int i = 0; i < n; i++)
    assert(arr[i] == 3.14f);

  sycl::free(arr, q);

  return 0;
}
