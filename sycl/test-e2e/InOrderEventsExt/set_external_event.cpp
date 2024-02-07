// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the ext_oneapi_set_external_event extension member on in-order queues.

#include <iostream>
#include <sycl.hpp>

constexpr size_t N = 1024;

int main() {
  sycl::context Ctx;
  sycl::device Dev = Ctx.get_devices()[0];

  sycl::queue Q1{Ctx, Dev, {sycl::property::queue::in_order{}}};
  sycl::queue Q2{Ctx, Dev, {sycl::property::queue::in_order{}}};

  sycl::buffer<int> DevDatabuf{sycl::range{N}};
  sycl::accessor DevData{DevDatabuf};
  int *HostData = (int *)malloc(N * sizeof(int) * 10);

  for (size_t I = 0; I < 10; ++I) {
    Q1.fill(DevData, 0);
    sycl::event E1 = Q1.submit([&](sycl::handler &h) {
      h.require(DevData);
      h.parallel_for(
          N, [=](sycl::item<1> Idx) { DevData[Idx] = 42 + Idx[0] + I; });
    });

    Q2.ext_oneapi_set_external_event(E1);
    sycl::event E2 = Q2.submit([&](sycl::handler &h) {
      h.require(DevData);
      h.parallel_for(N, [=](sycl::item<1> Idx) { ++DevData[Idx]; });
    });

    Q1.ext_oneapi_set_external_event(E2);
    Q1.copy(DevData, HostData + N * I);
  }

  Q1.wait_and_throw();

  int Failures = 0;
  for (size_t I = 0; I < 10; ++I) {
    for (size_t J = 0; J < N; ++J) {
      int Expected = 43 + J + I;
      int Actual = HostData[N * I + J];
      if (Expected != Actual) {
        std::cout << "Result not matching the expected value at index {" << I
                  << ", " << J << "}: " << Expected << " != " << Actual
                  << std::endl;
        ++Failures;
      }
    }
  }

  return Failures;
}
