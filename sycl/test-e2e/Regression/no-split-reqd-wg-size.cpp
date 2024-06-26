// This test checks that with -fsycl-device-code-split=off, kernels
// with different reqd_work_group_size dimensions can be launched.

// RUN: %{build} -fsycl -fsycl-device-code-split=off -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: hip

#include <sycl/detail/core.hpp>

using namespace sycl;

#define TEST(...)                                                              \
  {                                                                            \
    range globalRange(__VA_ARGS__);                                            \
    range localRange(__VA_ARGS__);                                             \
    nd_range NDRange(globalRange, localRange);                                 \
    q.parallel_for(NDRange,                                                    \
                   [=](auto) [[sycl::reqd_work_group_size(__VA_ARGS__)]] {});  \
  }

int main(int argc, char **argv) {
  queue q;
  TEST(4);
  TEST(4, 5);
  TEST(4, 5, 6);
  return 0;
}
