//==---------- SchedulerTest.hpp --- Scheduler unit tests ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/sycl.hpp>
#include <gtest/gtest.h>

class SchedulerTest : public ::testing::Test {
protected:
  __sycl_internal::__v1::async_handler MAsyncHandler =
      [](__sycl_internal::__v1::exception_list ExceptionList) {
        for (__sycl_internal::__v1::exception_ptr_class ExceptionPtr : ExceptionList) {
          try {
            std::rethrow_exception(ExceptionPtr);
          } catch (__sycl_internal::__v1::exception &E) {
            std::cerr << E.what();
          } catch (...) {
            std::cerr << "Unknown async exception was caught." << std::endl;
          }
        }
      };
  __sycl_internal::__v1::queue MQueue = __sycl_internal::__v1::queue(__sycl_internal::__v1::device(), MAsyncHandler);
};
