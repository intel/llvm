// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

int main() {

  sycl::queue q{sycl::gpu_selector_v};

  syclexp::command_graph g{q.get_context(), q.get_device()};

  const size_t n = 10;
  float *arr = sycl::malloc_shared<float>(n, q);

  g.begin_recording(q);
  float patternA = 1.0f;
  auto eventA = q.fill(arr, patternA, n);
  g.end_recording(q);
  auto execGraphA = g.finalize();

  g.begin_recording(q);
  float patternB = 2.0f;
  auto eventB = q.fill(arr, patternB, n, eventA);
  g.end_recording(q);
  auto execGraphB = g.finalize();

  g.begin_recording(q);
  float patternC = 3.0f;
  auto eventC = q.fill(arr, patternC, n, {eventA, eventB});
  g.end_recording(q);
  auto execGraphC = g.finalize();

  g.begin_recording(q);
  float patternD = 3.14f;
  q.submit([&](sycl::handler &h) {
    h.depends_on(eventC);
    h.fill(arr, patternD, n);
  });
  g.end_recording(q);
  auto execGraphD = g.finalize();

  auto verifyLambda =
      [&](syclexp::command_graph<syclexp::graph_state::executable> execGraph,
          float pattern) {
        q.submit([&](sycl::handler &h) {
           h.ext_oneapi_graph(execGraph);
         }).wait();

        for (int i = 0; i < n; i++)
          assert(arr[i] == pattern);
      };

  verifyLambda(execGraphA, patternA);
  verifyLambda(execGraphB, patternB);
  verifyLambda(execGraphC, patternC);
  verifyLambda(execGraphD, patternD);

  sycl::free(arr, q);

  return 0;
}
