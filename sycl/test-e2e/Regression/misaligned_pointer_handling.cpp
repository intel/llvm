// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//===----------------- misaligned_pointer_handling.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstring>
#include <stdint.h>
#include <sycl/detail/core.hpp>

using data_type_t = uint32_t;

void overflow(data_type_t *data) {
  sycl::buffer<data_type_t, 1> b{data, 1};
  constexpr data_type_t value = 0xff'ff'ff'ff;
  sycl::queue q;

  q.submit([&b](sycl::handler &h) {
    sycl::accessor a{b, h, sycl::read_write};
    h.parallel_for(sycl::range<1>{1}, [=](auto i) { a[i] += value; });
  }).wait();
}

int main() {
  data_type_t anyData[] = {1, 2};
  data_type_t before{}, after{};
  auto unaligned = (data_type_t *)(((uint8_t *)anyData) + 1);

  std::memcpy(&before, unaligned, sizeof(before));
  overflow(unaligned);
  std::memcpy(&after, unaligned, sizeof(after));

  assert(after == before - 1);
}
