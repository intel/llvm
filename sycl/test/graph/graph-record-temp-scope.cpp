// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

const size_t n = 10;
const float expectedValue = 42.0f;

void run_some_kernel(sycl::queue q, float *data) {
  // data is captured by ref here but will have gone out of scope when the
  // CGF is later run when the graph is executed.
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
      size_t i = idx;
      data[i] = expectedValue;
    });
  });
}

int main() {

  sycl::queue q{sycl::default_selector_v};

  sycl::ext::oneapi::experimental::command_graph g{q.get_context(),
                                                   q.get_device()};

  float *arr = sycl::malloc_shared<float>(n, q);

  g.begin_recording(q);
  run_some_kernel(q, arr);
  g.end_recording(q);

  auto exec_graph = g.finalize();

  q.submit([&](sycl::handler &h) { h.ext_oneapi_graph(exec_graph); }).wait();

  // Verify results
  for (size_t i = 0; i < n; i++) {
    assert(arr[i] == expectedValue);
  }

  sycl::free(arr, q.get_context());

  return 0;
}
