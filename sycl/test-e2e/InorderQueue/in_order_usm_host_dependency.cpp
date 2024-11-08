// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test checks that dependency chain between commands is preserved for in-order
// queue in the case when usm commands and host tasks are interleaved.

#include <sycl/detail/core.hpp>

#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

int main() {
  static constexpr size_t Size = 100;
  queue Q{property::queue::in_order()};

  int *X = malloc_host<int>(Size, Q);
  int *Y = malloc_host<int>(Size, Q);
  int *Z = malloc_host<int>(Size, Q);

  Q.memset(X, 0, sizeof(int) * Size);
  Q.submit([&](handler &CGH) {
    auto HostTask = [=] {
      for (int i = 0; i < Size; i++)
        X[i] += 99;
    };
    CGH.host_task(HostTask);
  });
  Q.memcpy(Y, X, sizeof(int) * Size);
  Q.submit([&](handler &CGH) {
    auto HostTask = [=] {
      for (int i = 0; i < Size; i++)
        Y[i] *= 2;
    };
    CGH.host_task(HostTask);
  });
  Q.fill(Z, 2, Size);

  Q.submit([&](handler &CGH) {
    auto HostTask = [=] { Z[99] += Y[99]; };
    CGH.host_task(HostTask);
  });
  Q.wait();

  int Error = (Z[99] != 200) ? 1 : 0;
  std::cout << (Error ? "failed\n" : "passed\n");

  free(X, Q);
  free(Y, Q);
  free(Z, Q);

  return Error;
}
