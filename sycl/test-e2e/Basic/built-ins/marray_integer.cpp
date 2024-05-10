// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t_preview.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t_preview.out%}

#include "helpers.hpp"

int main() {
  using namespace sycl;

  // upsample
  test(F(upsample), marray<int16_t, 2>{0x203, 0x302}, marray<int8_t, 2>{2, 3},
       marray<uint8_t, 2>{3, 2});
  test(F(upsample), marray<uint16_t, 2>{0x203, 0x302}, marray<uint8_t, 2>{2, 3},
       marray<uint8_t, 2>{3, 2});

  // abs_diff
  test(F(abs_diff), marray<unsigned, 2>{1, 1}, marray<unsigned, 2>{0, 1},
       marray<unsigned, 2>{1, 0});

  return 0;
}
