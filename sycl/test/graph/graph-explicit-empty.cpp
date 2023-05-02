// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

int main() {

  sycl::queue q{sycl::gpu_selector_v};

  sycl::ext::oneapi::experimental::command_graph g{q.get_context(),
                                                   q.get_device()};

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
      {start});

  auto empty = g.add({init});
  auto empty2 = g.add({empty});

  g.add(
      [&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
          size_t i = idx;
          arr[i] = 1;
        });
      },
      {empty2});

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
