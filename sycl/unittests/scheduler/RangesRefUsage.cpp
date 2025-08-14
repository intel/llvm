//==---- RangesRefUsage.cpp --- Check RangesRefT --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/ext/oneapi/experimental/rangesref.hpp>

#include <gtest/gtest.h>

TEST(RangesRefUsage, RangesRefUsage) {
  sycl::ext::oneapi::experimental::RangesRefT r0;
  ASSERT_EQ(r0.Dims, size_t{0});

  {
    sycl::range<1> global_range{1024};
    sycl::range<1> local_range{64};
    sycl::id<1> offset{10};
    sycl::nd_range<1> nd_range{global_range, local_range, offset};

    {
      sycl::ext::oneapi::experimental::RangesRefT r{nd_range};
      ASSERT_EQ(r.Dims, size_t{1});
      ASSERT_EQ(*r.GlobalSize, global_range[0]);
      ASSERT_EQ(*r.LocalSize, local_range[0]);
      ASSERT_EQ(*r.GlobalOffset, offset[0]);
    }
    {
      sycl::ext::oneapi::experimental::RangesRefT r{global_range, local_range};
      ASSERT_EQ(r.Dims, size_t{1});
      ASSERT_EQ(*r.GlobalSize, global_range[0]);
      ASSERT_EQ(*r.LocalSize, local_range[0]);
    }
    {
      sycl::ext::oneapi::experimental::RangesRefT r{global_range};
      ASSERT_EQ(r.Dims, size_t{1});
      ASSERT_EQ(*r.GlobalSize, global_range[0]);
      ASSERT_EQ(r.LocalSize, nullptr);
    }
  }
  {
    sycl::range<2> global_range{1024, 512};
    sycl::range<2> local_range{64, 32};
    sycl::id<2> offset{10, 20};
    sycl::nd_range<2> nd_range{global_range, local_range, offset};

    {
      sycl::ext::oneapi::experimental::RangesRefT r{nd_range};
      ASSERT_EQ(r.Dims, size_t{2});
      ASSERT_EQ(r.GlobalSize[0], global_range[0]);
      ASSERT_EQ(r.GlobalSize[1], global_range[1]);
      ASSERT_EQ(r.LocalSize[0], local_range[0]);
      ASSERT_EQ(r.LocalSize[1], local_range[1]);
      ASSERT_EQ(r.GlobalOffset[0], offset[0]);
      ASSERT_EQ(r.GlobalOffset[1], offset[1]);
    }
    {
      sycl::ext::oneapi::experimental::RangesRefT r{global_range, local_range};
      ASSERT_EQ(r.Dims, size_t{2});
      ASSERT_EQ(r.GlobalSize[0], global_range[0]);
      ASSERT_EQ(r.GlobalSize[1], global_range[1]);
      ASSERT_EQ(r.LocalSize[0], local_range[0]);
      ASSERT_EQ(r.LocalSize[1], local_range[1]);
    }
    {
      sycl::ext::oneapi::experimental::RangesRefT r{global_range};
      ASSERT_EQ(r.Dims, size_t{2});
      ASSERT_EQ(r.GlobalSize[0], global_range[0]);
      ASSERT_EQ(r.GlobalSize[1], global_range[1]);
      ASSERT_EQ(r.LocalSize, nullptr);
    }
  }
  {
    sycl::range<3> global_range{1024, 512, 256};
    sycl::range<3> local_range{64, 32, 16};
    sycl::id<3> offset{10, 20, 30};
    sycl::nd_range<3> nd_range{global_range, local_range, offset};

    {
      sycl::ext::oneapi::experimental::RangesRefT r{nd_range};
      ASSERT_EQ(r.Dims, size_t{3});
      ASSERT_EQ(r.GlobalSize[0], global_range[0]);
      ASSERT_EQ(r.GlobalSize[1], global_range[1]);
      ASSERT_EQ(r.GlobalSize[2], global_range[2]);
      ASSERT_EQ(r.LocalSize[0], local_range[0]);
      ASSERT_EQ(r.LocalSize[1], local_range[1]);
      ASSERT_EQ(r.LocalSize[2], local_range[2]);
      ASSERT_EQ(r.GlobalOffset[0], offset[0]);
      ASSERT_EQ(r.GlobalOffset[1], offset[1]);
      ASSERT_EQ(r.GlobalOffset[2], offset[2]);
    }
    {
      sycl::ext::oneapi::experimental::RangesRefT r{global_range, local_range};
      ASSERT_EQ(r.Dims, size_t{3});
      ASSERT_EQ(r.GlobalSize[0], global_range[0]);
      ASSERT_EQ(r.GlobalSize[1], global_range[1]);
      ASSERT_EQ(r.GlobalSize[2], global_range[2]);
      ASSERT_EQ(r.LocalSize[0], local_range[0]);
      ASSERT_EQ(r.LocalSize[1], local_range[1]);
      ASSERT_EQ(r.LocalSize[2], local_range[2]);
    }
    {
      sycl::ext::oneapi::experimental::RangesRefT r{global_range};
      ASSERT_EQ(r.Dims, size_t{3});
      ASSERT_EQ(r.GlobalSize[0], global_range[0]);
      ASSERT_EQ(r.GlobalSize[1], global_range[1]);
      ASSERT_EQ(r.GlobalSize[2], global_range[2]);
      ASSERT_EQ(r.LocalSize, nullptr);
    }
  }
}
