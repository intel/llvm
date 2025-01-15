//==---- RuntimeProperties.cpp --- check properties handling in RT --- -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

TEST(AccessorProperties, ValidPropsNoInit) {
  sycl::unittest::UrMock<> Mock;
  sycl::queue Queue;
  try {
    sycl::buffer<int, 1> Buf{1};
    std::ignore = Queue.submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::write, sycl::target::device>
          BuffAcc(Buf, cgh, sycl::property::no_init{});
      cgh.fill(BuffAcc, 1);
    });
    // no explicit checks, we expect no exception to be thrown
  } catch (...) {
    FAIL();
  }
}

TEST(AccessorProperties, SetUnsupportedParam) {
  sycl::unittest::UrMock<> Mock;
  sycl::queue Queue;
  try {
    sycl::buffer<int, 1> Buf{1};
    std::ignore = Queue.submit([&](sycl::handler &cgh) {
      // compile-time property is not supported in runtime
      sycl::accessor<int, 1, sycl::access_mode::write, sycl::target::device>
          BuffAcc(Buf, cgh, sycl::ext::oneapi::property::no_alias{});
      cgh.fill(BuffAcc, 1);
    });
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
    return;
  }
}

TEST(HostAccessorProperties, ValidPropsNoInit) {
  sycl::unittest::UrMock<> Mock;
  sycl::queue Queue;
  try {
    sycl::buffer<int, 1> Buf{1};
    sycl::host_accessor<int, 1, sycl::access_mode::write> BuffAcc(
        Buf, sycl::property::no_init{});
    // no explicit checks, we expect no exception to be thrown
  } catch (...) {
    FAIL();
  }
}

TEST(HostAccessorProperties, SetUnsupportedParam) {
  sycl::unittest::UrMock<> Mock;
  sycl::queue Queue;
  try {
    sycl::buffer<int, 1> Buf{1};
    sycl::host_accessor<int, 1, sycl::access_mode::write> BuffAcc(
        Buf, sycl::ext::oneapi::property::no_alias{});
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
    return;
  }
}

// no test for image_accesor since it doesn't have property parameter in
// constructor.

TEST(SampledAccessorProperties, NoSupportedProps) {
  sycl::unittest::UrMock<> Mock;
  sycl::queue Queue;
  try {
    sycl::image_sampler Sampler{
        sycl::addressing_mode::none,
        sycl::coordinate_normalization_mode::unnormalized,
        sycl::filtering_mode::linear};

    constexpr size_t ElementsCount = 4;
    constexpr size_t ChannelsCount = 4;
    int InitValue[ElementsCount * ChannelsCount];
    sycl::sampled_image<1> Image(&InitValue, sycl::image_format::r8g8b8a8_unorm,
                                 Sampler, sycl::range<1>(ElementsCount),
                                 sycl::property::image::use_host_ptr{});
    std::ignore = Queue.submit([&](sycl::handler &cgh) {
      sycl::sampled_image_accessor<sycl::uint4, 1, sycl::image_target::device>
          ImageAcc(Image, cgh, sycl::property::no_init{});
      cgh.single_task([=]() {});
    });
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
    return;
  }
  FAIL() << "Test must exit in exception handler. Exception is not thrown.";
}

TEST(UnsampledImageAccessor, ValidPropsNoInit) {
  sycl::unittest::UrMock<> Mock;
  sycl::queue Queue;
  try {
    constexpr size_t ElementsCount = 4;
    constexpr size_t ChannelsCount = 4;
    int InitValue[ElementsCount * ChannelsCount];
    sycl::unsampled_image<1> Image(
        &InitValue, sycl::image_format::r8g8b8a8_unorm,
        sycl::range<1>(ElementsCount), sycl::property::image::use_host_ptr{});
    std::ignore = Queue.submit([&](sycl::handler &cgh) {
      sycl::unsampled_image_accessor<sycl::uint4, 1, sycl::access_mode::write,
                                     sycl::image_target::device>
          ImageAcc(Image, cgh, sycl::property::no_init{});
      cgh.single_task([=]() {});
    });
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::feature_not_supported);
    EXPECT_STREQ(e.what(), "Device associated with command group handler does "
                           "not have aspect::image.");
    return;
  }
}

TEST(UnsampledImageAccessor, SetUnsupportedParam) {
  sycl::unittest::UrMock<> Mock;
  sycl::queue Queue;
  try {
    constexpr size_t ElementsCount = 4;
    sycl::unsampled_image<1> Image(sycl::image_format::r8g8b8a8_unorm,
                                   sycl::range<1>(ElementsCount),
                                   sycl::property::buffer::use_host_ptr{});
    std::ignore = Queue.submit([&](sycl::handler &cgh) {
      sycl::unsampled_image_accessor<sycl::uint4, 1, sycl::access_mode::write,
                                     sycl::image_target::device>
          ImageAcc(Image, cgh, sycl::ext::oneapi::property::no_alias{});
      cgh.single_task([=]() {});
    });
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
    return;
  }

  FAIL() << "Test must exit in exception handler. Exception is not thrown.";
}
