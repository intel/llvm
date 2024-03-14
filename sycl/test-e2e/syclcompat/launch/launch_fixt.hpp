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
 *  launch_fixt.hpp
 *
 *  Description:
 *     Fixtures and helpers for to tests the launch functionality
 **************************************************************************/

#pragma once

#include <sycl/sycl.hpp>

#include <syclcompat/device.hpp>
#include <syclcompat/dims.hpp>

// Struct containing test case data (local & global ranges)
template <int Dim> struct RangeParams {
  RangeParams(sycl::range<Dim> global_range_in, sycl::range<Dim> local_range,
              sycl::range<Dim> expect_global_range_out, bool pass)
      : global_range_in_{global_range_in}, local_range_in_{local_range},
        expect_global_range_out_{expect_global_range_out}, shouldPass_{pass} {}

  sycl::range<Dim> local_range_in_;
  sycl::range<Dim> global_range_in_;
  sycl::range<Dim> expect_global_range_out_;
  bool shouldPass_;

  // Pretty printing of RangeParams
  friend std::ostream &operator<<(std::ostream &os, const RangeParams &range) {
    auto print_range = [](std::ostream &os, const sycl::range<Dim> range) {
      os << " {";
      for (int i = 0; i < Dim; ++i) {
        os << range[i];
        os << ((Dim - i == 1) ? "} " : ", ");
      }
    };
    os << "Local:";
    print_range(os, range.local_range_in_);
    os << "Global (in): ";
    print_range(os, range.global_range_in_);
    os << "Global (out): ";
    print_range(os, range.expect_global_range_out_);
    os << (range.shouldPass_ ? "Should Work" : "Should Throw");
    return os;
  }
};

// Fixture for launch tests - initializes a few different
// range-like members & a queue.
struct LaunchTest {
  LaunchTest()
      : q_{syclcompat::get_default_queue()}, grid_{4, 2, 2}, thread_{32, 2, 2},
        range_1_{128, 32}, range_2_{{4, 128}, {2, 32}},
        range_3_{{2, 4, 64}, {2, 2, 32}} {}
  sycl::queue const q_;
  syclcompat::dim3 const grid_;
  syclcompat::dim3 const thread_;
  sycl::nd_range<1> const range_1_;
  sycl::nd_range<2> const range_2_;
  sycl::nd_range<3> const range_3_;
};

// Typed tests
template <typename T> struct LaunchTestWithArgs : public LaunchTest {
  LaunchTestWithArgs()
      : LaunchTest(), memsize_{LOCAL_MEM_SIZE},
        in_order_q_{{sycl::property::queue::in_order()}}, skip_{false} {
    should_skip();
  }

  void should_skip() {
    if (!syclcompat::get_current_device().has(sycl::aspect::fp64) &&
        std::is_same_v<T, double>) {
      std::cout << "  sycl::aspect::fp64 not supported by the SYCL device."
                << std::endl;
      skip_ = true;
    }
    if (!syclcompat::get_current_device().has(sycl::aspect::fp16) &&
        std::is_same_v<T, sycl::half>) {

      std::cout << "  sycl::aspect::fp16 not supported by the SYCL device."
                << std::endl;
      skip_ = true;
    }
  }

  constexpr static size_t LOCAL_MEM_SIZE = 64;

  size_t const memsize_;
  sycl::queue const in_order_q_;
  bool skip_;
};

using memsize_type_list =
    std::tuple<int, unsigned int, short, unsigned short, long, unsigned long>;
