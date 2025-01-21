// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==----- in_order_get_property.cpp - queue.get_property<in_order> test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>

#include <sycl/properties/all_properties.hpp>

using namespace sycl;

int main() {
  queue Queue1;
  try {
    assert(!Queue1.has_property<property::queue::in_order>() &&
           "Queue1 was created without any properties therefore has property "
           "should return 0");
    Queue1.get_property<property::queue::in_order>();
    assert(false && "Queue1 was created without any properties therefore get "
                    "property should fail.");
  } catch (const exception &e) {
    std::string ErrorMessage = e.what();
    assert(
        (e.code() == errc::invalid &&
         ErrorMessage.find("The property is not found") != std::string::npos) &&
        "Caught unexpected error!");
  }

  queue Queue2{property::queue::in_order()};
  assert(Queue2.has_property<property::queue::in_order>() &&
         "Queue2 should have property::queue::in_order.");
  property::queue::in_order Property =
      Queue2.get_property<property::queue::in_order>();

  return 0;
}
