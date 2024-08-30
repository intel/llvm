// RUN: %{build} %{embed-ir} -o %t.out
// RUN: %{run} %t.out

// COM: When ran on HIP and CUDA, this algorithm launches 'memcpy' commands
// leading to an infinite loop due to a bug in kernel fusion.

#include "./reduction.hpp"

int main() {
  test<detail::reduction::strategy::group_reduce_and_last_wg_detection>();
}
