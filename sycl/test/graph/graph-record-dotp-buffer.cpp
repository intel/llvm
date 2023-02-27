// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
#include <CL/sycl.hpp>
#include <iostream>
#include <thread>

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
      sycl::property::queue::in_order(),
      sycl::ext::oneapi::property::queue::lazy_execution{}};

  sycl::queue q{sycl::gpu_selector_v, properties};

  sycl::ext::oneapi::experimental::command_graph g;

  float dotpData = 0.f;
  std::vector<float> xData(n);
  std::vector<float> yData(n);
  std::vector<float> zData(n);

  {
    sycl::buffer dotpBuf(&dotpData, sycl::range<1>(1));

    sycl::buffer xBuf(xData);
    sycl::buffer yBuf(yData);
    sycl::buffer zBuf(zData);

    g.begin_recording(q);

    /* init data on the device */
    q.submit([&](sycl::handler &h) {
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

    q.submit([&](sycl::handler &h) {
      auto x = xBuf.get_access(h);
      auto y = yBuf.get_access(h);
      h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> it) {
        const size_t i = it[0];
        x[i] = alpha * x[i] + beta * y[i];
      });
    });

    q.submit([&](sycl::handler &h) {
      auto y = yBuf.get_access(h);
      auto z = zBuf.get_access(h);
      h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> it) {
        const size_t i = it[0];
        z[i] = gamma * z[i] + beta * y[i];
      });
    });

    q.submit([&](sycl::handler &h) {
      auto dotp = dotpBuf.get_access(h);
      auto x = xBuf.get_access(h);
      auto z = zBuf.get_access(h);
      h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> it) {
        const size_t i = it[0];
        // Doing a manual reduction here because reduction objects cause issues
        // with graphs.
        if (i == 0) {
          for (size_t j = 0; j < n; j++) {
            dotp[0] += x[j] * z[j];
          }
        }
      });
    });

    g.end_recording();

    auto exec_graph = g.finalize(q.get_context());

    q.submit([&](sycl::handler &h) { h.ext_oneapi_graph(exec_graph); });
  }

  if (dotpData != host_gold_result()) {
    std::cout << "Test failed: Error unexpected result!\n";
  } else {
    std::cout << "Test passed successfuly.";
  }

  return 0;
}