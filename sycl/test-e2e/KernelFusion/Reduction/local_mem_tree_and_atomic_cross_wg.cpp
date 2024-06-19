// RUN: %{build} %if any-device-is-hip || any-device-is-cuda %{ -fsycl-embed-ir %} -o %t.out
// RUN: %{run} %t.out

#include "./reduction.hpp"

int main() {
  test<detail::reduction::strategy::local_mem_tree_and_atomic_cross_wg>();
}
