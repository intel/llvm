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
 *  math_fixt.hpp
 *
 *  Description:
 *     Fixtures and helpers for to tests the math functionalities
 **************************************************************************/

#pragma once

#include <type_traits>

#include <sycl/sycl.hpp>
#include <syclcompat.hpp>

#include "../common.hpp"

template <typename Container, typename ValueT, typename = void>
static constexpr bool contained_is_same_v = false;

template <typename Container, typename ValueT>
static constexpr bool contained_is_same_v<
    Container, ValueT, std::void_t<typename Container::value_type>> =
    std::is_same_v<typename Container::value_type, ValueT>;

template <typename Container, typename = void>
static constexpr bool contained_is_integral_v = false;

template <typename Container>
static constexpr bool contained_is_integral_v<
    Container, std::void_t<typename Container::value_type>> =
    std::is_integral_v<typename Container::value_type>;

template <typename Container, typename = void>
static constexpr bool contained_is_floating_point_v = false;

template <typename Container>
static constexpr bool contained_is_floating_point_v<
    Container, std::void_t<typename Container::value_type>> =
    std::is_floating_point_v<typename Container::value_type> ||
    std::is_same_v<typename Container::value_type, sycl::half>;

template <typename ValueT> struct should_skip {
  bool operator()(const sycl::device &dev) const {
    if constexpr (std::is_same_v<ValueT, double> ||
                  contained_is_same_v<ValueT, double>) {
      if (!dev.has(sycl::aspect::fp64)) {
        std::cout << "  sycl::aspect::fp64 not supported by the SYCL device."
                  << std::endl;
        return true;
      }
    }
    if constexpr (std::is_same_v<ValueT, sycl::half> ||
                  contained_is_same_v<ValueT, sycl::half>) {
      if (!dev.has(sycl::aspect::fp16)) {
        std::cout << "  sycl::aspect::fp16 not supported by the SYCL device."
                  << std::endl;
        return true;
      }
    }
    return false;
  }
};

#define CHECK(ResultT, RESULT, EXPECTED)                                       \
  if constexpr (std::is_integral_v<ResultT>) {                                 \
    assert(RESULT == EXPECTED);                                                \
  } else if constexpr (std::is_floating_point_v<ResultT> ||                    \
                       std::is_same_v<ResultT, sycl::half>) {                  \
    if (sycl::isnan(RESULT))                                                   \
      assert(sycl::isnan(EXPECTED));                                           \
    else                                                                       \
      assert(fabs(RESULT - EXPECTED) < ERROR_TOLERANCE);                       \
  } else if constexpr (contained_is_floating_point_v<ResultT>) {               \
    for (size_t i = 0; i < RESULT.size(); i++)                                 \
      assert(fabs(RESULT[i] - EXPECTED[i]) < ERROR_TOLERANCE);                 \
  } else {                                                                     \
    static_assert(0, "Math_fixt.hpp should not have arrived here.");           \
  }

class OpTestLauncher {
protected:
  syclcompat::dim3 grid_;
  syclcompat::dim3 threads_;
  size_t data_size_;
  bool skip_;

public:
  OpTestLauncher(const syclcompat::dim3 &grid, const syclcompat::dim3 &threads,
                 const size_t data_size, const bool skip)
      : grid_{grid}, threads_{threads}, data_size_{data_size}, skip_{skip} {}
};

// Templated ResultT to support both arithmetic and boolean operators
template <typename ValueT, typename ValueU,
          typename ResultT = std::common_type_t<ValueT, ValueU>>
class BinaryOpTestLauncher : OpTestLauncher {
protected:
  ValueT *op1_;
  ValueU *op2_;
  ResultT res_h_, *res_;

public:
  BinaryOpTestLauncher(const syclcompat::dim3 &grid,
                       const syclcompat::dim3 &threads,
                       const size_t data_size = 1)
      : OpTestLauncher{
            grid, threads, data_size,
            should_skip<ValueT>()(syclcompat::get_current_device())} {
    if (skip_)
      return;
    op1_ = syclcompat::malloc<ValueT>(data_size);
    op2_ = syclcompat::malloc<ValueU>(data_size);
    res_ = syclcompat::malloc<ResultT>(data_size);
  };

  virtual ~BinaryOpTestLauncher() {
    if (skip_)
      return;
    syclcompat::free(op1_);
    syclcompat::free(op2_);
    syclcompat::free(res_);
  }

  template <auto Kernel>
  void launch_test(ValueT op1, ValueU op2, ResultT expected) {
    if (skip_)
      return;
    syclcompat::memcpy<ValueT>(op1_, &op1, data_size_);
    syclcompat::memcpy<ValueU>(op2_, &op2, data_size_);
    syclcompat::launch<Kernel>(grid_, threads_, op1_, op2_, res_);
    syclcompat::wait();
    syclcompat::memcpy<ResultT>(&res_h_, res_, data_size_);

    CHECK(ResultT, res_h_, expected);
  };
};

template <typename ValueT, typename ResultT = ValueT>
class UnaryOpTestLauncher : OpTestLauncher {
protected:
  ValueT *op_;
  ResultT res_h_, *res_;

public:
  UnaryOpTestLauncher(const syclcompat::dim3 &grid,
                      const syclcompat::dim3 &threads,
                      const size_t data_size = 1)
      : OpTestLauncher{
            grid, threads, data_size,
            should_skip<ValueT>()(syclcompat::get_current_device())} {
    if (skip_)
      return;
    op_ = syclcompat::malloc<ValueT>(data_size);
    res_ = syclcompat::malloc<ResultT>(data_size);
  };

  virtual ~UnaryOpTestLauncher() {
    if (skip_)
      return;
    syclcompat::free(op_);
    syclcompat::free(res_);
  }

  template <auto Kernel> void launch_test(ValueT op, ResultT expected) {
    if (skip_)
      return;
    syclcompat::memcpy<ValueT>(op_, &op, data_size_);
    syclcompat::launch<Kernel>(grid_, threads_, op_, res_);
    syclcompat::wait();
    syclcompat::memcpy<ResultT>(&res_h_, res_, data_size_);

    CHECK(ResultT, res_h_, expected);
  }
};
