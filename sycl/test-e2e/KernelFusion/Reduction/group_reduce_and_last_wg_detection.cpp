// RUN: %{build} -fsycl-embed-ir -o %t.out
// RUN: %{run} %t.out
// UNSUPPORTED: hip || cuda

#include "./reduction.hpp"

int main() {
  test<detail::reduction::strategy::group_reduce_and_last_wg_detection>();
}
