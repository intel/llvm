//===---------------discard_events_kernel_using_assert.hpp-----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Verifies that the last parameter is not `nullptr` for
/// urEnqueueKernelLaunchWithArgsExp when the queue has the discard_events
/// property. The lit RUN directives are in discard_events_using_assert.cpp.
///
//===----------------------------------------------------------------------===//
#include <cassert>
#include <iostream>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

using namespace sycl;
static constexpr int MAGIC_NUM = -1;
static constexpr size_t BUFFER_SIZE = 16;

int main() {

  sycl::property_list Props{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::discard_events{}};
  sycl::queue Q(Props);

  int *Harray = sycl::malloc_host<int>(BUFFER_SIZE, Q);
  if (Harray == nullptr) {
    return -1;
  }
  for (size_t i = 0; i < BUFFER_SIZE; ++i) {
    Harray[i] = MAGIC_NUM;
  }

  Q.parallel_for<class kernel_using_assert>(BUFFER_SIZE, [=](auto itemID) {
     size_t i = itemID.get_id(0);
     Harray[i] = i + 10;
     assert(Harray[i] == i + 10 && "assert message");
   }).wait();

  // Checks result
  for (size_t i = 0; i < BUFFER_SIZE; ++i) {
    size_t expected = i + 10;
    if (Harray[i] != expected)
      return -1;
  }
  free(Harray, Q);

  std::cout << "The test passed." << std::endl;
  return 0;
}
