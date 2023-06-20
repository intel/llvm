/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCLcompat
 *
 *  Dim.cpp
 *
 *  Description:
 *     dim3 tests
 **************************************************************************/

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <syclcompat/dims.hpp>

TEST(DimTest, Construct) {
  syclcompat::dim3 d3(512);
  EXPECT_EQ(d3.x, 512);
  EXPECT_EQ(d3.y, 1);
  EXPECT_EQ(d3.z, 1);
}

TEST(DimTest, Convert) {
  syclcompat::dim3 d3(512);
  sycl::range<3> r3 = d3;
  EXPECT_EQ(d3.x, r3[2]);
  EXPECT_EQ(d3.y, r3[1]);
  EXPECT_EQ(d3.z, r3[0]);

  sycl::range<2> r2{1, 2};
  syclcompat::dim3 d3_from_range2(r2);
  EXPECT_EQ(d3_from_range2.x, 2);
  EXPECT_EQ(d3_from_range2.y, 1);
  EXPECT_EQ(d3_from_range2.z, 1);

  sycl::range<1> r1{2};
  syclcompat::dim3 d3_from_range1(r1);
  EXPECT_EQ(d3_from_range2.x, 2);
  EXPECT_EQ(d3_from_range2.y, 1);
  EXPECT_EQ(d3_from_range2.z, 1);
}

TEST(DimTest, ConvertBack) {
  // Dimension-dependent conversions and
  // check that exceptions are thrown when trying to convert
  // higher dimensional dim3 to sycl::range
  {
    syclcompat::dim3 dim_3D(512, 4, 2);

    sycl::range<3> range_3D{dim_3D};
    sycl::range<3> exp_3D{2, 4, 512};
    EXPECT_EQ(range_3D, exp_3D);

    EXPECT_THROW(sycl::range<2> range_2D{dim_3D}, std::invalid_argument);
    EXPECT_THROW(sycl::range<1> range_1D{dim_3D}, std::invalid_argument);
  }
  {
    syclcompat::dim3 dim_2D(512, 2);

    sycl::range<3> range_3D{dim_2D};
    sycl::range<3> exp_3D{1, 2, 512};
    EXPECT_EQ(range_3D, exp_3D);

    sycl::range<2> range_2D{dim_2D};
    sycl::range<2> exp_2D{2, 512};
    EXPECT_EQ(range_2D, exp_2D);

    EXPECT_THROW(sycl::range<1> range_1D{dim_2D}, std::invalid_argument);
  }
  {
    syclcompat::dim3 dim_1D{512};
    sycl::range<3> range_3D{dim_1D};
    sycl::range<3> exp_3D{1, 1, 512};
    EXPECT_EQ(range_3D, exp_3D);

    sycl::range<2> range_2D{dim_1D};
    sycl::range<2> exp_2D{1, 512};
    EXPECT_EQ(range_2D, exp_2D);

    sycl::range<1> range_1D{dim_1D};
    sycl::range<1> exp_1D{512};
    EXPECT_EQ(range_1D, exp_1D);
  }
}

// Check that an nd_range is correctly constructed
// from pair of dim3
TEST(DimTest, ConvertMulti) {
  syclcompat::dim3 threads(32, 4, 2);
  syclcompat::dim3 grid(4, 1, 1);

  sycl::nd_range<3> range{grid * threads, threads};

  EXPECT_EQ(range.get_global_range()[0], 2);
  EXPECT_EQ(range.get_global_range()[1], 4);
  EXPECT_EQ(range.get_global_range()[2], 128);
  EXPECT_EQ(range.get_local_range()[0], 2);
  EXPECT_EQ(range.get_local_range()[1], 4);
  EXPECT_EQ(range.get_local_range()[2], 32);
}
