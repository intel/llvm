//===---------------------------- Tests.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/device_filter.hpp>
#include <sycl/exception.hpp>

#include <gtest/gtest.h>

#include <vector>
#include <utility>
#include <string>

TEST(OneAPIDeviceSelector, IsCaseInsensitive) {
  ASSERT_EQ(sycl::detail::Parse_ONEAPI_DEVICE_SELECTOR("OPENCL:*"),
            sycl::detail::Parse_ONEAPI_DEVICE_SELECTOR("opencl:*"))
      << " backend should be case-insensitive";

  ASSERT_EQ(sycl::detail::Parse_ONEAPI_DEVICE_SELECTOR("*:GPU"),
            sycl::detail::Parse_ONEAPI_DEVICE_SELECTOR("*:gpu"))
      << " device type should be case-insensitive";
}

TEST(OneAPIDeviceSelector, EmitsErrorIfOnlyBackendIsSpecified) {
  try {
    std::ignore = sycl::detail::Parse_ONEAPI_DEVICE_SELECTOR("level_zero");
    FAIL() << "An exception was expected";
  } catch (const sycl::exception &e) {
    ASSERT_EQ(e.code(), sycl::errc::invalid);
    ASSERT_EQ(std::string(e.what()),
              "Incomplete selector!  Try 'level_zero:*' if all "
              "devices under the backend was original intention.");
  }
}

TEST(OneAPIDeviceSelector, EmitsErrorIfBackendStringIsInvalid) {
  try {
    std::ignore = sycl::detail::Parse_ONEAPI_DEVICE_SELECTOR("macaroni:*");
    FAIL() << "An exception was expected";
  } catch (const sycl::exception &e) {
    ASSERT_EQ(e.code(), sycl::errc::invalid);
    // FIXME: the error below could be better
    ASSERT_EQ(std::string(e.what()),
              "ONEAPI_DEVICE_SELECTOR parsing error. Backend is required but "
              "missing from \"macaroni:*\"");
  }
}

// TODO: test case ":"
// TODO: test case ":cpu"
// TODO: test case "level_zero:cpu:cpu"
// TODO: test case "opencl:"
// TODO: other positive test cases for parsing device, sub-devices, etc.
