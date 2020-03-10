//==------------ MockHandler.hpp --- Scheduler unit tests ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl.hpp>
#include <detail/queue_impl.hpp>

class MockHandler : public sycl::handler {
public:
  MockHandler(sycl::shared_ptr_class<sycl::detail::queue_impl> Queue)
      : sycl::handler(std::move(Queue), true) {}
  sycl::event mockFinalize() { return finalize(); }

  sycl::vector_class<sycl::detail::ArgDesc> &getKernelArgs() { return MArgs; }
  sycl::vector_class<sycl::detail::ArgDesc> &getKernelAccessors() {
    return MAssociatedAccesors;
  }
};
