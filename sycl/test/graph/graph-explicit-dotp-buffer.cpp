// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
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

  float dotpData = 0.f;
  std::vector<float> xData(n);
  std::vector<float> yData(n);
  std::vector<float> zData(n);

  {
    sycl::buffer dotpBuf(&dotpData, sycl::range<1>(1));

    sycl::buffer xBuf(xData);
    sycl::buffer yBuf(yData);
    sycl::buffer zBuf(zData);

    /* init data on the device */
    auto n_i = g.add([&](sycl::handler &h) {
      auto x = xBuf.get_access(h);
      auto y = yBuf.get_access(h);
      auto z = zBuf.get_access(h);
      h.parallel_for(n, [=](sycl::id<1> it) {
        const size_t i = it[0];
        x[i] = 1.0f;
        y[i] = 2.0f;
        z[i] = 3.0f;
      });
    });

    auto node_a = g.add([&](sycl::handler &h) {
      auto x = xBuf.get_access(h);
      auto y = yBuf.get_access(h);
      h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> it) {
        const size_t i = it[0];
        x[i] = alpha * x[i] + beta * y[i];
      });
    });

    auto node_b = g.add([&](sycl::handler &h) {
      auto y = yBuf.get_access(h);
      auto z = zBuf.get_access(h);
      h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> it) {
        const size_t i = it[0];
        z[i] = gamma * z[i] + beta * y[i];
      });
    });

    auto node_c = g.add([&](sycl::handler &h) {
      auto dotp = dotpBuf.get_access(h);
      auto x = xBuf.get_access(h);
      auto z = zBuf.get_access(h);
#ifdef TEST_GRAPH_REDUCTIONS
      h.parallel_for(sycl::range<1>{n},
                     sycl::reduction(dotpBuf, h, 0.0f, std::plus()),
                     [=](sycl::id<1> it, auto &sum) {
                       const size_t i = it[0];
                       sum += x[i] * z[i];
                     });
#else
      h.single_task([=]() {
        // Doing a manual reduction here because reduction objects cause issues
        // with graphs.
        for (size_t j = 0; j < n; j++) {
          dotp[0] += x[j] * z[j];
        }
      });
#endif
    });

    auto executable_graph = g.finalize();

    // Using shortcut for executing a graph of commands
    q.ext_oneapi_graph(executable_graph).wait();
  }

  assert(dotpData == host_gold_result());
  return 0;
}
