//==---------- SchedulerTest.hpp --- Scheduler unit tests ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl.hpp>
#include <gtest/gtest.h>

class SchedulerTest : public ::testing::Test {
protected:
  sycl::queue MQueue;
  sycl::async_handler MAsyncHandler = [](sycl::exception_list ExceptionList) {
    for (sycl::exception_ptr_class ExceptionPtr : ExceptionList) {
      try {
        std::rethrow_exception(ExceptionPtr);
      } catch (sycl::exception &E) {
        std::cerr << E.what();
      } catch (...) {
        std::cerr << "Unknown async exception was caught." << std::endl;
      }
    }
  };

public:
  void SetUp() override {
    MQueue = sycl::queue(sycl::host_selector(), MAsyncHandler);
  }
};
