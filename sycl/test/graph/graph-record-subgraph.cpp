// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

// Modified version of the dotp example which records part of the graph as a
// sub-graph
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

const size_t n = 10;

float host_gold_result() {
  float alpha = 1.0f;
  float beta = 2.0f;
  float gamma = 3.0f;

  float sum = 0.0f;

  for (size_t i = 0; i < n; ++i) {
    sum += (alpha * 1.0f + beta * 2.0f) * (gamma * 3.0f + beta * 2.0f);
  }

  return sum;
}

int main() {
  float alpha = 1.0f;
  float beta = 2.0f;
  float gamma = 3.0f;

  sycl::queue q{sycl::gpu_selector_v};

  sycl::ext::oneapi::experimental::command_graph g{q.get_context(),
                                                   q.get_device()};
  sycl::ext::oneapi::experimental::command_graph subGraph{q.get_context(),
                                                          q.get_device()};

  float *dotp = sycl::malloc_shared<float>(1, q);

  float *x = sycl::malloc_shared<float>(n, q);
  float *y = sycl::malloc_shared<float>(n, q);
  float *z = sycl::malloc_shared<float>(n, q);

  subGraph.begin_recording(q);
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> it) {
      const size_t i = it[0];
      x[i] = alpha * x[i] + beta * y[i];
    });
  });

  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> it) {
      const size_t i = it[0];
      z[i] = gamma * z[i] + beta * y[i];
    });
  });

  subGraph.end_recording();
  auto subGraphExec = subGraph.finalize();

  g.begin_recording(q);
  /* init data on the device */
  auto init = q.submit([&](sycl::handler &h) {
    h.parallel_for(n, [=](sycl::id<1> it) {
      const size_t i = it[0];
      x[i] = 1.0f;
      y[i] = 2.0f;
      z[i] = 3.0f;
    });
  });

  auto sub = q.submit([&](sycl::handler &h) {
    h.depends_on(init);
    h.ext_oneapi_graph(subGraphExec);
  });

  q.submit([&](sycl::handler &h) {
    h.depends_on(sub);
#ifdef TEST_GRAPH_REDUCTIONS
    h.parallel_for(sycl::range<1>{n}, sycl::reduction(dotp, 0.0f, std::plus()),
                   [=](sycl::id<1> it, auto &sum) {
                     const size_t i = it[0];
                     sum += x[i] * z[i];
                   });
#else
    h.single_task([=]() {
      // Doing a manual reduction here because reduction objects cause
      // issues with graphs.
      for (size_t j = 0; j < n; j++) {
        dotp[0] += x[j] * z[j];
      }
    });
#endif
  });

  g.end_recording();

  auto executable_graph = g.finalize();

  // Using shortcut for executing a graph of commands
  q.ext_oneapi_graph(executable_graph).wait();

  assert(*dotp == host_gold_result());

  sycl::free(dotp, q);
  sycl::free(x, q);
  sycl::free(y, q);
  sycl::free(z, q);

  return 0;
}
