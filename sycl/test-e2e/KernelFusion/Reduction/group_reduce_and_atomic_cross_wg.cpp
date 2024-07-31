// RUN: %{build} %{embed-ir} -o %t.out
// RUN: %{run} %t.out

#include "./reduction.hpp"

int main() {
  test<detail::reduction::strategy::group_reduce_and_atomic_cross_wg>();
}
