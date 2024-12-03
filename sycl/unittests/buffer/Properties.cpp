//==-------- Properties.cpp --- check properties handling in RT --- --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

TEST(BufferProps, ValidPropsHostPtr) {
  try {
    int HostPtr[1];
    sycl::buffer<int, 1> Buf{HostPtr, 1,
                             sycl::property::buffer::use_host_ptr{}};
    // no explicit checks, we expect no exception to be thrown
  } catch (...) {
    FAIL();
  }
}

TEST(BufferProps, ValidPropsContextBound) {
  try {
    sycl::unittest::UrMock<> Mock;
    sycl::context Context;
    sycl::buffer<int, 1> Buf{1, sycl::property::buffer::context_bound{Context}};
    // no explicit checks, we expect no exception to be thrown
  } catch (...) {
    FAIL();
  }
}

TEST(BufferProps, ValidPropsMutex) {
  try {
    std::mutex Mutex;
    sycl::buffer<int, 1> Buf{1, sycl::property::buffer::use_mutex{Mutex}};
    // no explicit checks, we expect no exception to be thrown
  } catch (...) {
    FAIL();
  }
}

TEST(BufferProps, ValidPropsPinnedHostMem) {
  try {
    sycl::buffer<int, 1> Buf(
        1, {sycl::ext::oneapi::property::buffer::use_pinned_host_memory()});
    // no explicit checks, we expect no exception to be thrown
  } catch (...) {
    FAIL();
  }
}

TEST(BufferProps, ValidPropsMemChannel) {
  try {
    sycl::buffer<int, 1> Buf(1, sycl::property::buffer::mem_channel{1});
    // no explicit checks, we expect no exception to be thrown
  } catch (...) {
    FAIL();
  }
}

TEST(BufferProps, SetAndQueryMatch) {
  try {
    sycl::unittest::UrMock<> Mock;
    std::mutex Mutex;
    sycl::context Context;
    int HostPtr[1];

    sycl::buffer<int, 1> buf{HostPtr,
                             1,
                             {sycl::property::buffer::use_host_ptr{},
                              sycl::property::buffer::context_bound{Context},
                              sycl::property::buffer::use_mutex{Mutex}}};

    ASSERT_TRUE(buf.has_property<sycl::property::buffer::context_bound>());
    EXPECT_EQ(
        buf.get_property<sycl::property::buffer::context_bound>().get_context(),
        Context);
    ASSERT_TRUE(buf.has_property<sycl::property::buffer::use_mutex>());
    EXPECT_EQ(
        buf.get_property<sycl::property::buffer::use_mutex>().get_mutex_ptr(),
        &Mutex);
    EXPECT_TRUE(buf.has_property<sycl::property::buffer::use_host_ptr>());
    // check some random not supported and not sent param
    EXPECT_FALSE(buf.has_property<sycl::property::image::use_host_ptr>());
  } catch (...) {
    FAIL();
  }
}

TEST(BufferProps, SetUnsupportedParam) {
  try {
    sycl::buffer<int, 1> buf{1, {sycl::property::image::use_host_ptr{}}};
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
    return;
  }

  FAIL() << "Test must exit in exception handler. Exception is not thrown.";
}

TEST(ImageProps, ValidPropsHostPtr) {
  try {
    constexpr size_t ElementsCount = 4;
    constexpr size_t ChannelsCount = 4;
    float InitValue[ElementsCount * ChannelsCount];
    sycl::image<1> Image(&InitValue, sycl::image_channel_order::rgba,
                         sycl::image_channel_type::fp32,
                         sycl::range<1>(ElementsCount),
                         sycl::property::image::use_host_ptr{});
    // no explicit checks, we expect no exception to be thrown
  } catch (...) {
    FAIL();
  }
}

TEST(ImageProps, ValidPropsContextBound) {
  try {
    sycl::unittest::UrMock<> Mock;
    sycl::context Context;
    constexpr size_t ElementsCount = 4;
    sycl::image<1> Image(sycl::image_channel_order::rgba,
                         sycl::image_channel_type::fp32,
                         sycl::range<1>(ElementsCount),
                         sycl::property::image::context_bound{Context});
    // no explicit checks, we expect no exception to be thrown
  } catch (...) {
    FAIL();
  }
}

TEST(ImageProps, ValidPropsMutex) {
  try {
    std::mutex Mutex;
    constexpr size_t ElementsCount = 4;
    sycl::image<1> Image(
        sycl::image_channel_order::rgba, sycl::image_channel_type::fp32,
        sycl::range<1>(ElementsCount), sycl::property::image::use_mutex{Mutex});
    // no explicit checks, we expect no exception to be thrown
  } catch (...) {
    FAIL();
  }
}

TEST(ImageProps, SetUnsupportedParam) {
  try {
    std::mutex Mutex;
    constexpr size_t ElementsCount = 4;
    sycl::image<1> Image(sycl::image_channel_order::rgba,
                         sycl::image_channel_type::fp32,
                         sycl::range<1>(ElementsCount),
                         sycl::property::buffer::use_mutex{Mutex});
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
    return;
  }

  FAIL() << "Test must exit in exception handler. Exception is not thrown.";
}

TEST(USMAllocator, Properties) {
  sycl::unittest::UrMock<> Mock;
  try {
    sycl::queue Q;
    sycl::usm_allocator<int, sycl::usm::alloc::shared> Allocator(
        Q, sycl::property::buffer::use_host_ptr{});
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
    return;
  }

  FAIL() << "Test must exit in exception handler. Exception is not thrown.";
}

TEST(SampledImage, ValidPropsHostPtr) {
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
    // no explicit checks, we expect no exception to be thrown
  } catch (...) {
    FAIL();
  }
}

TEST(SampledImage, SetUnsupportedParam) {
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
                                 sycl::property::buffer::use_host_ptr{});
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
    return;
  }

  FAIL() << "Test must exit in exception handler. Exception is not thrown.";
}

TEST(UnsampledImage, ValidPropsHostPtr) {
  try {
    constexpr size_t ElementsCount = 4;
    constexpr size_t ChannelsCount = 4;
    int InitValue[ElementsCount * ChannelsCount];
    sycl::unsampled_image<1> Image(
        &InitValue, sycl::image_format::r8g8b8a8_unorm,
        sycl::range<1>(ElementsCount), sycl::property::image::use_host_ptr{});
    // no explicit checks, we expect no exception to be thrown
  } catch (...) {
    FAIL();
  }
}

TEST(UnsampledImage, SetUnsupportedParam) {
  try {
    constexpr size_t ElementsCount = 4;
    sycl::unsampled_image<1> Image(sycl::image_format::r8g8b8a8_unorm,
                                   sycl::range<1>(ElementsCount),
                                   sycl::property::buffer::use_host_ptr{});
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
    return;
  }

  FAIL() << "Test must exit in exception handler. Exception is not thrown.";
}
