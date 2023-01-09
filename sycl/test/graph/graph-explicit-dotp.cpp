// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
#include <CL/sycl.hpp>
#include <iostream>

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

  sycl::property_list properties{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::lazy_execution{}};

  sycl::queue q{sycl::gpu_selector_v, properties};

  sycl::ext::oneapi::experimental::command_graph g;

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
    });
  });

  auto node_a = g.add(
      [&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> it) {
          const size_t i = it[0];
          x[i] = alpha * x[i] + beta * y[i];
        });
      },
      {n_i});

  auto node_b = g.add(
      [&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> it) {
          const size_t i = it[0];
          z[i] = gamma * z[i] + beta * y[i];
        });
      },
      {n_i});

  auto node_c = g.add(
      [&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>{n},
                       sycl::reduction(dotp, 0.0f, std::plus()),
                       [=](sycl::id<1> it, auto &sum) {
                         const size_t i = it[0];
                         sum += x[i] * z[i];
                       });
      },
      {node_a, node_b});

  auto executable_graph = g.finalize(q.get_context());

  // Using shortcut for executing a graph of commands
  q.ext_oneapi_graph(executable_graph).wait();

  if (*dotp != host_gold_result()) {
    std::cout << "Error unexpected result!\n";
  }

  sycl::free(dotp, q);
  sycl::free(x, q);
  sycl::free(y, q);
  sycl::free(z, q);

  std::cout << "done.\n";

  return 0;
}
