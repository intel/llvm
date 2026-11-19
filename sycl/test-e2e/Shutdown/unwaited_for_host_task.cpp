// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

using namespace sycl;

// This test checks that host tasks that haven't been waited for do not cause
// issues during shutdown.
int main(int argc, char *argv[]) {
  queue q;

  q.submit([&](handler &h) { h.host_task([=]() {}); });

  return 0;
}
