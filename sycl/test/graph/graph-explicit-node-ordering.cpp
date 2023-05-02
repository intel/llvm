// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

int main() {

  sycl::queue q{sycl::gpu_selector_v};

  sycl::ext::oneapi::experimental::command_graph g{q.get_context(),
                                                   q.get_device()};

  const size_t n = 10;
  float *x = sycl::malloc_shared<float>(n, q);

  auto init = g.add([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
      size_t i = idx;
      x[i] = 2.0f;
    });
  });

  auto add = g.add([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
      size_t i = idx;
      x[i] += 2.0f;
    });
  });

  auto mult = g.add([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
      size_t i = idx;
      x[i] *= 3.0f;
    });
  });

  g.make_edge(init, mult);
  g.make_edge(mult, add);

  auto executable_graph = g.finalize();

  q.submit([&](sycl::handler &h) {
     h.ext_oneapi_graph(executable_graph);
   }).wait();

  for (int i = 0; i < n; i++)
    assert(x[i] == 8.0f);

  sycl::free(x, q);

  return 0;
}
