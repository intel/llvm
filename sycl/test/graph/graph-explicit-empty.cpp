// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

int main() {
  namespace sycl_ext = sycl::ext::oneapi::experimental;

  sycl::queue q{sycl::gpu_selector_v};

  auto my_properties =
      sycl::property_list{sycl_ext::property::graph::no_cycle_check()};
  sycl_ext::command_graph g{q.get_context(), q.get_device(), my_properties};

  const size_t n = 10;
  float *arr = sycl::malloc_device<float>(n, q);

  auto start = g.add();

  auto init = g.add(
      [&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
          size_t i = idx;
          arr[i] = 0;
        });
      },
      {sycl_ext::property::node::depends_on(start)});

  auto empty = g.add({sycl_ext::property::node::depends_on(init)});
  auto empty2 = g.add({sycl_ext::property::node::depends_on(empty)});

  g.add(
      [&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
          size_t i = idx;
          arr[i] = 1;
        });
      },
      {sycl_ext::property::node::depends_on(empty2)});

  auto executable_graph = g.finalize();

  q.submit([&](sycl::handler &h) {
     h.ext_oneapi_graph(executable_graph);
   }).wait();

  std::vector<float> HostData(n);
  q.memcpy(HostData.data(), arr, n * sizeof(float)).wait();

  for (int i = 0; i < n; i++)
    assert(HostData[i] == 1.f);

  sycl::free(arr, q);

  return 0;
}
