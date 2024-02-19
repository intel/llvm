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

// FIXME: Only needed because sycl::vec does not have the value_type alias
template <typename Contained, int NumElements, typename ValueT>
static constexpr bool
    contained_is_same_v<sycl::vec<Contained, NumElements>, ValueT, void> =
        std::is_same_v<Contained, ValueT>;

template <typename Container, typename = void>
static constexpr bool contained_is_integral_v = false;

template <typename Container>
static constexpr bool contained_is_integral_v<
    Container, std::void_t<typename Container::value_type>> =
    std::is_integral_v<typename Container::value_type>;

// FIXME: Only needed because sycl::vec does not have the value_type alias
template <typename Contained, int NumElements>
static constexpr bool
    contained_is_integral_v<sycl::vec<Contained, NumElements>, void> =
        std::is_integral_v<Contained>;

template <typename Container, typename = void>
static constexpr bool contained_is_floating_point_v = false;

template <typename Container>
static constexpr bool contained_is_floating_point_v<
    Container, std::void_t<typename Container::value_type>> =
    std::is_floating_point_v<typename Container::value_type> ||
    std::is_same_v<typename Container::value_type, sycl::half>;

// FIXME: Only needed because sycl::vec does not have the value_type alias
template <typename Contained, int NumElements>
static constexpr bool
    contained_is_floating_point_v<sycl::vec<Contained, NumElements>, void> =
        std::is_floating_point_v<Contained> ||
        std::is_same_v<Contained, sycl::half>;

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

class OpTestLauncher {
protected:
  syclcompat::dim3 grid_;
  syclcompat::dim3 threads_;
  bool skip_;

public:
  OpTestLauncher(const syclcompat::dim3 &grid, const syclcompat::dim3 &threads,
                 const bool skip)
      : grid_{grid}, threads_{threads}, skip_{skip} {}
};

// Templated TRes to support both arithmetic and boolean operators
template <typename ValueT, typename ValueU,
          typename TRes = std::common_type_t<ValueT, ValueU>>
class BinaryOpTestLauncher : OpTestLauncher {
protected:
  ValueT *op1_;
  ValueU *op2_;
  TRes *res_;

public:
  BinaryOpTestLauncher(const syclcompat::dim3 &grid,
                       const syclcompat::dim3 &threads,
                       const size_t data_size = 1)
      : OpTestLauncher{
            grid, threads,
            should_skip<ValueT>()(syclcompat::get_current_device())} {
    if (skip_)
      return;
    op1_ = syclcompat::malloc_shared<ValueT>(data_size);
    op2_ = syclcompat::malloc_shared<ValueU>(data_size);
    res_ = syclcompat::malloc_shared<TRes>(data_size);
  };

  virtual ~BinaryOpTestLauncher() {
    if (skip_)
      return;
    syclcompat::free(op1_);
    syclcompat::free(op2_);
    syclcompat::free(res_);
  }

  template <auto Kernel>
  void launch_test(ValueT op1, ValueU op2, TRes expected) {
    if (skip_)
      return;
    *op1_ = op1;
    *op2_ = op2;
    syclcompat::launch<Kernel>(grid_, threads_, op1_, op2_, res_);
    syclcompat::wait();

    if constexpr (std::is_integral_v<ValueT>)
      assert(*res_ == expected);
    else if constexpr (std::is_floating_point_v<ValueT> ||
                       std::is_same_v<ValueT, sycl::half>)
      assert(fabs(*res_ - expected) < ERROR_TOLERANCE);
    else if constexpr (contained_is_floating_point_v<ValueT>) // Container
      for (size_t i = 0; i < res_->size(); i++)
        assert(fabs((*res_)[i] - expected[i]) < ERROR_TOLERANCE);
    else
      assert(0); // If arrived here, no results where checked
  }
};

template <typename ValueT, typename TRes = ValueT>
class UnaryOpTestLauncher : OpTestLauncher {
protected:
  ValueT *op_;
  ValueT *res_;

public:
  UnaryOpTestLauncher(const syclcompat::dim3 &grid,
                      const syclcompat::dim3 &threads,
                      const size_t data_size = 1)
      : OpTestLauncher{
            grid, threads,
            should_skip<ValueT>()(syclcompat::get_current_device())} {
    if (skip_)
      return;
    op_ = syclcompat::malloc_shared<ValueT>(data_size);
    res_ = syclcompat::malloc_shared<ValueT>(data_size);
  };

  virtual ~UnaryOpTestLauncher() {
    if (skip_)
      return;
    syclcompat::free(op_);
    syclcompat::free(res_);
  }

  template <auto Kernel> void launch_test(ValueT op, TRes expected) {
    if (skip_)
      return;
    *op_ = op;
    syclcompat::launch<Kernel>(grid_, threads_, op_, res_);
    syclcompat::wait();

    if constexpr (std::is_integral_v<ValueT>)
      assert(*res_ == expected);
    else if constexpr (std::is_floating_point_v<ValueT> ||
                       std::is_same_v<ValueT, sycl::half>)
      assert(fabs(*res_ - expected) < ERROR_TOLERANCE);
    else if constexpr (contained_is_floating_point_v<ValueT>) // Container
      for (size_t i = 0; i < res_->size(); i++)
        assert(fabs((*res_)[i] - expected[i]) < ERROR_TOLERANCE);
    else
      assert(0); // If arrived here, no results where checked
  }
};
