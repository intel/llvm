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
 *  SYCL compatibility API
 *
 *  UsmAllocations.cpp
 *
 *  Description:
 *    USM allocation tests
 **************************************************************************/

#include <gtest/gtest.h>
#include <numeric>
#include <sycl/ext/oneapi/experimental/compat.hpp>
#include <sycl/sycl.hpp>

#define WG_SIZE 256
#define NUM_WG 32

using namespace sycl::ext::oneapi::experimental;

template <typename T> void VectorKernel(T *A) {
  auto id = compat::global_id::x();
  A[id] = id;
}

// Fixture to set up & launch VectorAdd kernel
template <typename T> class USMTest : public ::testing::Test {
protected:
  USMTest()
      : q_{compat::get_default_queue()}, grid_{NUM_WG}, thread_{WG_SIZE},
        size_{WG_SIZE * NUM_WG} {}
  void SetUp() {
    if ((std::is_same_v<T, double>)&&!q_.get_device().has(sycl::aspect::fp64))
      GTEST_SKIP();
    if ((std::is_same_v<T, sycl::half>)&&!q_.get_device().has(
            sycl::aspect::fp16))
      GTEST_SKIP();
  }
  void launch_kernel() {
    return launch<VectorKernel<T>>(grid_, thread_, q_, d_A).wait();
  }

  // Check result is identity vector
  // Handles memcpy for USM device alloc
  void check_result() {
    sycl::usm::alloc ptr_type = sycl::get_pointer_type(d_A, q_.get_context());
    ASSERT_NE(ptr_type, sycl::usm::alloc::unknown);

    T *result;
    if (ptr_type == sycl::usm::alloc::device) {
      result = static_cast<T *>(std::malloc(sizeof(T) * size_));
      compat::memcpy(result, d_A, sizeof(T) * size_);
    } else {
      result = d_A;
    }

    for (size_t i = 0; i < size_; i++) {
      EXPECT_EQ(result[i], static_cast<T>(i));
    }

    if (ptr_type == sycl::usm::alloc::device)
      std::free(result);
  }

  sycl::queue q_;
  compat::dim3 const grid_;
  compat::dim3 const thread_;
  T *d_A;
  size_t size_;
};

// Test template compat::malloc<T> on multiple
// value types
using value_type_list =
    testing::Types<int, unsigned int, short, unsigned short, long,
                   unsigned long, long long, unsigned long long, float, double,
                   sycl::half>;
TYPED_TEST_SUITE(USMTest, value_type_list);

TYPED_TEST(USMTest, malloc) {
  this->d_A = compat::malloc<TypeParam>(this->size_);
  this->launch_kernel();
  this->check_result();
  compat::free(this->d_A);
}

TYPED_TEST(USMTest, host) {
  if (!this->q_.get_device().has(sycl::aspect::usm_host_allocations))
    GTEST_SKIP();
  this->d_A = compat::malloc_host<TypeParam>(this->size_);
  this->launch_kernel();
  this->check_result();
  compat::free(this->d_A);
}

TYPED_TEST(USMTest, shared) {
  if (!this->q_.get_device().has(sycl::aspect::usm_shared_allocations))
    GTEST_SKIP();
  this->d_A = compat::malloc_shared<TypeParam>(this->size_);
  this->launch_kernel();
  this->check_result();
  compat::free(this->d_A);
}

// Avoid combinatorial explosion by only testing non-templated
// compat::malloc with int type
using int_type_list = testing::Types<int>;
template <class T> struct USMTest_noT : public USMTest<T> {};
TYPED_TEST_SUITE(USMTest_noT, int_type_list);

TYPED_TEST(USMTest_noT, malloc) {
  this->d_A =
      static_cast<TypeParam *>(compat::malloc(this->size_ * sizeof(TypeParam)));
  this->launch_kernel();
  this->check_result();
  compat::free(this->d_A);
}

TYPED_TEST(USMTest_noT, host) {
  if (!this->q_.get_device().has(sycl::aspect::usm_host_allocations))
    GTEST_SKIP();
  this->d_A = static_cast<TypeParam *>(
      compat::malloc_host(this->size_ * sizeof(TypeParam)));
  this->launch_kernel();
  this->check_result();
  compat::free(this->d_A);
}

TYPED_TEST(USMTest_noT, shared) {
  if (!this->q_.get_device().has(sycl::aspect::usm_shared_allocations))
    GTEST_SKIP();
  this->d_A = static_cast<TypeParam *>(
      compat::malloc_shared(this->size_ * sizeof(TypeParam)));
  this->launch_kernel();
  this->check_result();
  compat::free(this->d_A);
}

// Test deduce direction
TEST(USM_Deduction, deduce) {
  using memcpy_direction = compat::detail::memcpy_direction;
  auto default_queue = compat::get_default_queue();

  int *h_ptr = (int *)compat::malloc_host(sizeof(int));
  int *sys_ptr = (int *)std::malloc(sizeof(int));
  int *d_ptr = (int *)compat::malloc(sizeof(int));
  int *s_ptr = (int *)compat::malloc_shared(sizeof(int));

  // * to host
  EXPECT_EQ(
      compat::detail::deduce_memcpy_direction(default_queue, h_ptr, h_ptr),
      memcpy_direction::device_to_device);
  EXPECT_EQ(
      compat::detail::deduce_memcpy_direction(default_queue, h_ptr, sys_ptr),
      memcpy_direction::host_to_host);
  EXPECT_EQ(
      compat::detail::deduce_memcpy_direction(default_queue, h_ptr, d_ptr),
      memcpy_direction::device_to_device);
  EXPECT_EQ(
      compat::detail::deduce_memcpy_direction(default_queue, h_ptr, s_ptr),
      memcpy_direction::device_to_device);

  // * to sys
  EXPECT_EQ(
      compat::detail::deduce_memcpy_direction(default_queue, sys_ptr, h_ptr),
      memcpy_direction::host_to_host);
  EXPECT_EQ(
      compat::detail::deduce_memcpy_direction(default_queue, sys_ptr, sys_ptr),
      memcpy_direction::host_to_host);
  EXPECT_EQ(
      compat::detail::deduce_memcpy_direction(default_queue, sys_ptr, d_ptr),
      memcpy_direction::device_to_host);
  EXPECT_EQ(
      compat::detail::deduce_memcpy_direction(default_queue, sys_ptr, s_ptr),
      memcpy_direction::host_to_host);

  // * to dev
  EXPECT_EQ(
      compat::detail::deduce_memcpy_direction(default_queue, d_ptr, h_ptr),
      memcpy_direction::device_to_device);
  EXPECT_EQ(
      compat::detail::deduce_memcpy_direction(default_queue, d_ptr, sys_ptr),
      memcpy_direction::host_to_device);
  EXPECT_EQ(
      compat::detail::deduce_memcpy_direction(default_queue, d_ptr, d_ptr),
      memcpy_direction::device_to_device);
  EXPECT_EQ(
      compat::detail::deduce_memcpy_direction(default_queue, d_ptr, s_ptr),
      memcpy_direction::device_to_device);

  // * to shared
  EXPECT_EQ(
      compat::detail::deduce_memcpy_direction(default_queue, s_ptr, h_ptr),
      memcpy_direction::device_to_device);
  EXPECT_EQ(
      compat::detail::deduce_memcpy_direction(default_queue, s_ptr, sys_ptr),
      memcpy_direction::host_to_host);
  EXPECT_EQ(
      compat::detail::deduce_memcpy_direction(default_queue, s_ptr, d_ptr),
      memcpy_direction::device_to_device);
  EXPECT_EQ(
      compat::detail::deduce_memcpy_direction(default_queue, s_ptr, s_ptr),
      memcpy_direction::device_to_device);
}
