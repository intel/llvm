// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

int main() {

  sycl::queue q{sycl::gpu_selector_v};

  // Test passing empty property list, which is the default
  sycl::property_list empty_properties;
  sycl::ext::oneapi::experimental::command_graph g(
      q.get_context(), q.get_device(), empty_properties);

  const size_t n = 10;
  float *arr = sycl::malloc_shared<float>(n, q);

  g.add([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
      size_t i = idx;
      arr[i] = 1;
    });
  });

  auto executable_graph = g.finalize(empty_properties);

  auto e1 = q.ext_oneapi_graph(executable_graph);
  auto e2 = q.ext_oneapi_graph(executable_graph, e1);
  auto e3 = q.ext_oneapi_graph(executable_graph, e1);
  q.ext_oneapi_graph(executable_graph, {e2, e3}).wait();

  for (int i = 0; i < n; i++)
    assert(arr[i] == 1);

  sycl::free(arr, q);

  return 0;
}
