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
  namespace sycl_ext = sycl::ext::oneapi::experimental;

  float alpha = 1.0f;
  float beta = 2.0f;
  float gamma = 3.0f;

  sycl::queue q{sycl::gpu_selector_v};

  sycl_ext::command_graph g{q.get_context(), q.get_device()};

  float *dotp = sycl::malloc_device<float>(1, q);

  float *x = sycl::malloc_device<float>(n, q);
  float *y = sycl::malloc_device<float>(n, q);
  float *z = sycl::malloc_device<float>(n, q);

  /* init data on the device */
  auto n_i = g.add([&](sycl::handler &h) {
    h.parallel_for(n, [=](sycl::id<1> it) {
      const size_t i = it[0];
      x[i] = 1.0f;
      y[i] = 2.0f;
      z[i] = 3.0f;
    });
  });

  auto node_a = g.add(
      [&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> it) {
          const size_t i = it[0];
          x[i] = alpha * x[i] + beta * y[i];
        });
      },
      {sycl_ext::property::node::depends_on(n_i)});

  auto node_b = g.add(
      [&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> it) {
          const size_t i = it[0];
          z[i] = gamma * z[i] + beta * y[i];
        });
      },
      {sycl_ext::property::node::depends_on(n_i)});

  auto node_c = g.add(
      [&](sycl::handler &h) {
#ifdef TEST_GRAPH_REDUCTIONS
        h.parallel_for(sycl::range<1>{n},
                       sycl::reduction(dotp, 0.0f, std::plus()),
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
      },
      {sycl_ext::property::node::depends_on(node_a, node_b)});

  auto executable_graph = g.finalize();

  // Using shortcut for executing a graph of commands
  q.ext_oneapi_graph(executable_graph).wait();

  std::vector<float> dotp_host(1);
  q.copy(dotp, dotp_host.data(), 1).wait();

  assert(dotp_host[0] == host_gold_result());

  sycl::free(dotp, q);
  sycl::free(x, q);
  sycl::free(y, q);
  sycl::free(z, q);

  return 0;
}
