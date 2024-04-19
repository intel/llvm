// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}

#include "vec_common.hpp"

// Alias is needed to const-qualify vec without template args.
template <typename T, int NumElems>
using ConstVec = const sycl::vec<T, NumElems>;

int main() {
  run_test<ConstVec>();
  return 0;
}
