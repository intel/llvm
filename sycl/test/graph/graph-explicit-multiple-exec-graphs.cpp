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

  float *dotp = sycl::malloc_shared<float>(1, q);

  float *x = sycl::malloc_shared<float>(n, q);
  float *y = sycl::malloc_shared<float>(n, q);
  float *z = sycl::malloc_shared<float>(n, q);

  /* init data on the device */
  auto n_i = g.add([&](sycl::handler &h) {
    h.parallel_for(n, [=](sycl::id<1> it) {
      const size_t i = it[0];
      x[i] = 1.0f;
      y[i] = 2.0f;
      z[i] = 3.0f;
      dotp[0] = 0.0f;
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
        h.single_task([=]() {
          for (size_t j = 0; j < n; j++) {
            dotp[0] += x[j] * z[j];
          }
        });
      },
      {sycl_ext::property::node::depends_on(node_a, node_b)});

  auto executable_graph = g.finalize();

  // Add an extra node for the second executable graph which modifies the output
  auto node_d =
      g.add([&](sycl::handler &h) { h.single_task([=]() { dotp[0] += 1; }); },
            {sycl_ext::property::node::depends_on(node_c)});

  auto executable_graph_2 = g.finalize();

  // Using shortcut for executing a graph of commands
  q.ext_oneapi_graph(executable_graph).wait();

  assert(*dotp == host_gold_result());

  q.ext_oneapi_graph(executable_graph_2).wait();

  assert(*dotp == host_gold_result() + 1);

  sycl::free(dotp, q);
  sycl::free(x, q);
  sycl::free(y, q);
  sycl::free(z, q);

  return 0;
}
