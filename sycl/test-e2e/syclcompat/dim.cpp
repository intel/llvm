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

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <cassert>
#include <sycl/detail/core.hpp>
#include <syclcompat/dims.hpp>

int main() {
  std::cout << "Testing Construct" << std::endl;
  {
    syclcompat::dim3 d3(512);
    assert(d3.x == 512);
    assert(d3.y == 1);
    assert(d3.z == 1);
  }
  std::cout << "Testing Empty Construct" << std::endl;
  {
    syclcompat::dim3 d3;
    assert(d3.x == 1);
    assert(d3.y == 1);
    assert(d3.z == 1);
  }
  std::cout << "Testing Empty Construct & Update" << std::endl;
  {
    syclcompat::dim3 d3;
    d3.x = 1;
    d3.y = 2;
    d3.z = 3;

    assert(d3.x == 1);
    assert(d3.y == 2);
    assert(d3.z == 3);
  }
  std::cout << "Testing Empty Construct & Update 2" << std::endl;
  {
    syclcompat::dim3 d3;
    d3.x = 32;

    assert(d3.x == 32);
    assert(d3.y == 1);
    assert(d3.z == 1);
  }
  std::cout << "Testing Convert" << std::endl;
  {
    syclcompat::dim3 d3(512);
    sycl::range<3> r3 = d3;
    assert(d3.x == r3[2]);
    assert(d3.y == r3[1]);
    assert(d3.z == r3[0]);

    sycl::range<2> r2{1, 2};
    syclcompat::dim3 d3_from_range2(r2);
    assert(d3_from_range2.x == 2);
    assert(d3_from_range2.y == 1);
    assert(d3_from_range2.z == 1);

    sycl::range<1> r1{2};
    syclcompat::dim3 d3_from_range1(r1);
    assert(d3_from_range2.x == 2);
    assert(d3_from_range2.y == 1);
    assert(d3_from_range2.z == 1);
  }
  std::cout << "Testing ConvertBack" << std::endl;
  // Dimension-dependent conversions and
  // check that exceptions are thrown when trying to convert
  // higher dimensional dim3 to sycl::range
  {
    syclcompat::dim3 dim_3D(512, 4, 2);

    sycl::range<3> range_3D{dim_3D};
    sycl::range<3> exp_3D{2, 4, 512};
    assert(range_3D == exp_3D);

    try {
      sycl::range<2> range_2D{dim_3D};
    } catch (std::invalid_argument const &e) {
      std::cout << "Expected SYCL exception caught: " << e.what();
    }

    try {
      sycl::range<1> range_1D{dim_3D};
    } catch (std::invalid_argument const &e) {
      std::cout << "Expected SYCL exception caught: " << e.what();
    }
  }
  {
    syclcompat::dim3 dim_2D(512, 2);

    sycl::range<3> range_3D{dim_2D};
    sycl::range<3> exp_3D{1, 2, 512};
    assert(range_3D == exp_3D);

    sycl::range<2> range_2D{dim_2D};
    sycl::range<2> exp_2D{2, 512};
    assert(range_2D == exp_2D);

    try {
      sycl::range<1> range_1D{dim_2D};
    } catch (std::invalid_argument const &e) {
      std::cout << "Expected SYCL exception caught: " << e.what();
    }
  }
  {
    syclcompat::dim3 dim_1D{512};
    sycl::range<3> range_3D{dim_1D};
    sycl::range<3> exp_3D{1, 1, 512};
    assert(range_3D == exp_3D);

    sycl::range<2> range_2D{dim_1D};
    sycl::range<2> exp_2D{1, 512};
    assert(range_2D == exp_2D);

    sycl::range<1> range_1D{dim_1D};
    sycl::range<1> exp_1D{512};
    assert(range_1D == exp_1D);
  }

  // Check that an nd_range is correctly constructed
  // from pair of dim3
  std::cout << "Testing ConvertMulti" << std::endl;
  {
    syclcompat::dim3 threads(32, 4, 2);
    syclcompat::dim3 grid(4, 1, 1);

    sycl::nd_range<3> range{grid * threads, threads};

    assert(range.get_global_range()[0] == 2);
    assert(range.get_global_range()[1] == 4);
    assert(range.get_global_range()[2] == 128);
    assert(range.get_local_range()[0] == 2);
    assert(range.get_local_range()[1] == 4);
    assert(range.get_local_range()[2] == 32);
  }

  return 0;
}
