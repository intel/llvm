//==---------- SchedulerTest.hpp --- Scheduler unit tests ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/queue.hpp>

#include <gtest/gtest.h>

class SchedulerTest : public ::testing::Test {
protected:
  cl::sycl::async_handler MAsyncHandler =
      [](cl::sycl::exception_list ExceptionList) {
        for (std::exception_ptr ExceptionPtr : ExceptionList) {
          try {
            std::rethrow_exception(ExceptionPtr);
          } catch (cl::sycl::exception &E) {
            std::cerr << E.what();
          } catch (...) {
            std::cerr << "Unknown async exception was caught." << std::endl;
          }
        }
      };
  cl::sycl::queue MQueue = cl::sycl::queue(cl::sycl::device(), MAsyncHandler);
};
