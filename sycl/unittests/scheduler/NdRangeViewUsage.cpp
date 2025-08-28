//==---- NdRangeViewUsage.cpp --- Check nd_range_view ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <detail/cg.hpp>
#include <sycl/detail/nd_range_view.hpp>

#include <gtest/gtest.h>

template <int dims>
void TestNdRangeView(sycl::range<dims> global, sycl::range<dims> local,
                     sycl::id<dims> offset) {
  {
    sycl::nd_range<dims> nd_range{global, local, offset};
    sycl::detail::nd_range_view r{nd_range};
    ASSERT_EQ(r.Dims, size_t{dims});
    for (int d = 0; d < dims; d++) {
      ASSERT_EQ(r.GlobalSize[d], global[d]);
      ASSERT_EQ(r.LocalSize[d], local[d]);
      ASSERT_EQ(r.Offset[d], offset[d]);
    }

    sycl::detail::NDRDescT NDRDesc = r.toNDRDescT();
    ASSERT_EQ(NDRDesc.Dims, size_t{dims});
    for (int d = 0; d < dims; d++) {
      ASSERT_EQ(NDRDesc.GlobalSize[d], global[d]);
      ASSERT_EQ(NDRDesc.LocalSize[d], local[d]);
      ASSERT_EQ(NDRDesc.GlobalOffset[d], offset[d]);
    }
  }
  {
    sycl::detail::nd_range_view r{global, local};
    ASSERT_EQ(r.Dims, size_t{dims});
    for (int d = 0; d < dims; d++) {
      ASSERT_EQ(r.GlobalSize[d], global[d]);
      ASSERT_EQ(r.LocalSize[d], local[d]);
    }
    ASSERT_EQ(r.Offset, nullptr);

    sycl::detail::NDRDescT NDRDesc = r.toNDRDescT();
    ASSERT_EQ(NDRDesc.Dims, size_t{dims});
    for (int d = 0; d < dims; d++) {
      ASSERT_EQ(NDRDesc.GlobalSize[d], global[d]);
      ASSERT_EQ(NDRDesc.LocalSize[d], local[d]);
    }
    for (int d = dims; d < 3; d++) {
      ASSERT_EQ(NDRDesc.GlobalSize[d], 0UL);
      ASSERT_EQ(NDRDesc.LocalSize[d], 0UL);
    }
    for (int d = 0; d < 3; d++) {
      ASSERT_EQ(NDRDesc.GlobalOffset[d], 0UL);
    }
  }
  {
    sycl::detail::nd_range_view r{global};
    ASSERT_EQ(r.Dims, size_t{dims});
    for (int d = 0; d < dims; d++) {
      ASSERT_EQ(r.GlobalSize[d], global[d]);
    }
    ASSERT_EQ(r.LocalSize, nullptr);
    ASSERT_EQ(r.Offset, nullptr);

    sycl::detail::NDRDescT NDRDesc = r.toNDRDescT();
    ASSERT_EQ(NDRDesc.Dims, size_t{dims});
    for (int d = 0; d < dims; d++) {
      ASSERT_EQ(NDRDesc.GlobalSize[d], global[d]);
    }
    for (int d = dims; d < 3; d++) {
      ASSERT_EQ(NDRDesc.GlobalSize[d], 0UL);
    }
    for (int d = 0; d < 3; d++) {
      ASSERT_EQ(NDRDesc.LocalSize[d], 0UL);
      ASSERT_EQ(NDRDesc.GlobalOffset[d], 0UL);
    }
  }
}

TEST(RangesRefUsage, RangesRefUsage) {
  TestNdRangeView(sycl::range<1>{1024}, sycl::range<1>{64}, sycl::id<1>{10});
  TestNdRangeView(sycl::range<2>{1024, 512}, sycl::range<2>{64, 32},
                  sycl::id<2>{10, 5});
  TestNdRangeView(sycl::range<3>{1024, 512, 256}, sycl::range<3>{64, 32, 16},
                  sycl::id<3>{10, 5, 2});
}
