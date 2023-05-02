// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

int main() {
  const size_t n = 10;
  const float expectedValue = 7.f;

  sycl::queue q{sycl::default_selector_v};
  sycl::queue q2;

  sycl::ext::oneapi::experimental::command_graph g{q.get_context(),
                                                   q.get_device()};

  float *arr = sycl::malloc_shared<float>(n, q);

  g.begin_recording(q);

  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
      size_t i = idx;
      arr[i] = expectedValue;
    });
  });

  g.end_recording(q);
  g.end_recording(q2);

  auto exec_graph = g.finalize();

  q.submit([&](sycl::handler &h) { h.ext_oneapi_graph(exec_graph); }).wait();

  // Verify results
  for (size_t i = 0; i < n; i++) {
    assert(arr[i] == expectedValue);
  }

  sycl::free(arr, q.get_context());

  return 0;
}
