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
 *  Launch.cpp
 *
 *  Description:
 *     launch<F> and launch<F> with dinamyc local memory tests
 **************************************************************************/

#include <gtest/gtest.h>
#include <sycl/ext/oneapi/experimental/compat.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>

using namespace sycl::ext::oneapi::experimental;

// Struct containing test case data (local & global ranges)
template <int Dim> struct RangeParams {
  RangeParams(sycl::range<Dim> global_range_in, sycl::range<Dim> local_range,
              sycl::range<Dim> expect_global_range_out, bool pass)
      : global_range_in_{global_range_in}, local_range_in_{local_range},
        expect_global_range_out_{expect_global_range_out}, shouldPass_{pass} {}
  // Gtest requires explicit default ctor
  RangeParams()
      : global_range_in_{1, 1, 1}, local_range_in_{1, 1, 1},
        expect_global_range_out_{1, 1, 1}, shouldPass_{true} {};

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

class RangeParamsTestFixture3D
    : public ::testing::TestWithParam<RangeParams<3>> {
protected:
  RangeParams<3> params;
};

TEST_P(RangeParamsTestFixture3D, ComputeNDRange3D) {
  RangeParams<3> r = GetParam();
  try {
    auto g_out =
        compat::compute_nd_range(r.global_range_in_, r.local_range_in_);
    sycl::nd_range<3> x_out = {r.expect_global_range_out_, r.local_range_in_};
    if (r.shouldPass_) {
      ASSERT_EQ(g_out, x_out);
    } else {
      FAIL() << "Expected std::invalid_argument";
    }
  } catch (std::invalid_argument const &err) {
    if (r.shouldPass_) {
      FAIL();
    } else {
      SUCCEED();
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    ComputeNDRangeTests, RangeParamsTestFixture3D,
    ::testing::Values(
        // Round up
        RangeParams<3>{{11, 1, 1}, {2, 1, 1}, {12, 1, 1}, true},
        // Even size
        RangeParams<3>{{320, 1, 1}, {32, 1, 1}, {320, 1, 1}, true},
        // Round up
        RangeParams<3>{{32, 193, 1}, {16, 32, 1}, {32, 224, 1}, true},
        // zero size
        RangeParams<3>{{10, 0, 0}, {1, 0, 0}, {10, 0, 0}, false},
        // zero size
        RangeParams<3>{{0, 10, 10}, {0, 10, 10}, {0, 10, 10}, false},
        // local > global
        RangeParams<3>{{2, 1, 1}, {32, 1, 1}, {32, 1, 1}, false},
        // local > global
        RangeParams<3>{{1, 2, 1}, {1, 32, 1}, {1, 32, 1}, false},
        // local > global
        RangeParams<3>{{1, 1, 2}, {1, 1, 32}, {1, 1, 32}, false}));

// Fixture for launch tests - initializes a few different
// range-like members & a queue.
class LaunchTest1 : public testing::Test {
public:
  LaunchTest1()
      : q_{compat::get_default_queue()}, grid_{4, 2, 2}, thread_{32, 2, 2},
        range_1_{128, 32}, range_2_{{4, 128}, {2, 32}}, range_3_{{2, 4, 64},
                                                                 {2, 2, 32}} {}

protected:
  sycl::queue const q_;
  compat::dim3 const grid_;
  compat::dim3 const thread_;
  sycl::nd_range<1> const range_1_;
  sycl::nd_range<2> const range_2_;
  sycl::nd_range<3> const range_3_;
};

// Dummy kernel functions for testing
void empty_kernel(){};
void int_kernel(int a){};
void int_ptr_kernel(int *a){};

TEST_F(LaunchTest1, NoArgLaunch) {
  compat::launch<empty_kernel>(range_1_);
  compat::launch<empty_kernel>(range_2_);
  compat::launch<empty_kernel>(range_3_);
  compat::launch<empty_kernel>(grid_, thread_);

  compat::launch<empty_kernel>(range_1_, q_);
  compat::launch<empty_kernel>(range_2_, q_);
  compat::launch<empty_kernel>(range_3_, q_);
  compat::launch<empty_kernel>(grid_, thread_, q_);
}

TEST_F(LaunchTest1, OneArgLaunch) {
  int my_int;
  compat::launch<int_kernel>(range_1_, my_int);
  compat::launch<int_kernel>(range_2_, my_int);
  compat::launch<int_kernel>(range_3_, my_int);
  compat::launch<int_kernel>(grid_, thread_, my_int);

  compat::launch<int_kernel>(range_1_, q_, my_int);
  compat::launch<int_kernel>(range_2_, q_, my_int);
  compat::launch<int_kernel>(range_3_, q_, my_int);
  compat::launch<int_kernel>(grid_, thread_, q_, my_int);
}

TEST_F(LaunchTest1, PtrArgLaunch) {
  int *int_ptr;
  compat::launch<int_ptr_kernel>(range_1_, int_ptr);
  compat::launch<int_ptr_kernel>(range_2_, int_ptr);
  compat::launch<int_ptr_kernel>(range_3_, int_ptr);
  compat::launch<int_ptr_kernel>(grid_, thread_, int_ptr);

  compat::launch<int_ptr_kernel>(range_1_, q_, int_ptr);
  compat::launch<int_ptr_kernel>(range_2_, q_, int_ptr);
  compat::launch<int_ptr_kernel>(range_3_, q_, int_ptr);
  compat::launch<int_ptr_kernel>(grid_, thread_, q_, int_ptr);
}

template <typename T> class LocalMemLaunchTest1_t : public LaunchTest1 {
public:
  LocalMemLaunchTest1_t()
      : LaunchTest1(), memsize_{LocalMemLaunchTest1_t<T>::LOCAL_MEM_SIZE},
        in_order_q_{{sycl::property::queue::in_order()}} {}

  void SetUp() {
    if (!sycl::ext::oneapi::experimental::compat::get_current_device().has(
            sycl::aspect::fp64) &&
        std::is_same_v<T, double>)
      GTEST_SKIP() << "sycl::aspect::fp64 not supported by the SYCL device.";
    if (!sycl::ext::oneapi::experimental::compat::get_current_device().has(
            sycl::aspect::fp16) &&
        std::is_same_v<T, sycl::half>)
      GTEST_SKIP() << "sycl::aspect::fp16 not supported by the SYCL device.";
  }

  constexpr static size_t LOCAL_MEM_SIZE = 32;

protected:
  size_t const memsize_;
  sycl::queue const in_order_q_;
};

using value_type_list =
    testing::Types<int, unsigned int, short, unsigned short, long,
                   unsigned long, long long, unsigned long long, float, double,
                   sycl::half>;
TYPED_TEST_SUITE(LocalMemLaunchTest1_t, value_type_list);

template <typename T> void local_mem_empty_kernel(T *local_mem){};
template <typename T> void local_mem_basicdt_kernel(T *local_mem, T value){};
template <typename T> void local_mem_typed_kernel(T *local_mem, T *data) {
  constexpr size_t memsize = LocalMemLaunchTest1_t<T>::LOCAL_MEM_SIZE;
  const int id = sycl::ext::oneapi::experimental::this_item<1>();
  if (id < memsize) {
    local_mem[id] = static_cast<T>(id);
    sycl::group_barrier(sycl::ext::oneapi::experimental::this_group<1>());
    data[id] = local_mem[memsize - id - 1];
  }
};

TYPED_TEST(LocalMemLaunchTest1_t, NoArgLaunch) {
  compat::launch<local_mem_empty_kernel<TypeParam>>(this->range_1_,
                                                    this->memsize_);
  compat::launch<local_mem_empty_kernel<TypeParam>>(this->range_2_,
                                                    this->memsize_);
  compat::launch<local_mem_empty_kernel<TypeParam>>(this->range_3_,
                                                    this->memsize_);
  compat::launch<local_mem_empty_kernel<TypeParam>>(this->grid_, this->thread_,
                                                    this->memsize_);
}

TYPED_TEST(LocalMemLaunchTest1_t, NoArgLaunch_q) {
  compat::launch<local_mem_empty_kernel<TypeParam>>(
      this->range_1_, this->memsize_, this->in_order_q_);
  compat::launch<local_mem_empty_kernel<TypeParam>>(
      this->range_2_, this->memsize_, this->in_order_q_);
  compat::launch<local_mem_empty_kernel<TypeParam>>(
      this->range_3_, this->memsize_, this->in_order_q_);
  compat::launch<local_mem_empty_kernel<TypeParam>>(
      this->grid_, this->thread_, this->memsize_, this->in_order_q_);
}

TYPED_TEST(LocalMemLaunchTest1_t, BasicdtLaunch) {
  TypeParam d_a = TypeParam(1);

  compat::launch<local_mem_basicdt_kernel<TypeParam>>(this->range_1_,
                                                      this->memsize_, d_a);
  compat::launch<local_mem_basicdt_kernel<TypeParam>>(this->range_2_,
                                                      this->memsize_, d_a);
  compat::launch<local_mem_basicdt_kernel<TypeParam>>(this->range_3_,
                                                      this->memsize_, d_a);
  compat::launch<local_mem_basicdt_kernel<TypeParam>>(
      this->grid_, this->thread_, this->memsize_, d_a);
}

TYPED_TEST(LocalMemLaunchTest1_t, BasicdtLaunch_q) {
  TypeParam d_a = TypeParam(1);

  compat::launch<local_mem_basicdt_kernel<TypeParam>>(
      this->range_1_, this->memsize_, this->in_order_q_, d_a);
  compat::launch<local_mem_basicdt_kernel<TypeParam>>(
      this->range_2_, this->memsize_, this->in_order_q_, d_a);
  compat::launch<local_mem_basicdt_kernel<TypeParam>>(
      this->range_3_, this->memsize_, this->in_order_q_, d_a);
  compat::launch<local_mem_basicdt_kernel<TypeParam>>(
      this->grid_, this->thread_, this->memsize_, this->in_order_q_, d_a);
}

TYPED_TEST(LocalMemLaunchTest1_t, ArgLaunch) {
  TypeParam *d_a = (TypeParam *)sycl::ext::oneapi::experimental::compat::malloc(
      this->memsize_ * sizeof(TypeParam));

  compat::launch<local_mem_typed_kernel<TypeParam>>(this->range_1_,
                                                    this->memsize_, d_a);
  compat::launch<local_mem_typed_kernel<TypeParam>>(this->range_2_,
                                                    this->memsize_, d_a);
  compat::launch<local_mem_typed_kernel<TypeParam>>(this->range_3_,
                                                    this->memsize_, d_a);
  compat::launch<local_mem_typed_kernel<TypeParam>>(this->grid_, this->thread_,
                                                    this->memsize_, d_a);

  sycl::ext::oneapi::experimental::compat::free((void *)d_a);
}

TYPED_TEST(LocalMemLaunchTest1_t, ArgLaunch_q) {
  TypeParam *d_a =
      (TypeParam *)sycl::ext::oneapi::experimental::compat::malloc<TypeParam>(
          this->memsize_, this->in_order_q_);

  compat::launch<local_mem_typed_kernel<TypeParam>>(
      this->range_1_, this->memsize_, this->in_order_q_, d_a);
  compat::launch<local_mem_typed_kernel<TypeParam>>(
      this->range_2_, this->memsize_, this->in_order_q_, d_a);
  compat::launch<local_mem_typed_kernel<TypeParam>>(
      this->range_3_, this->memsize_, this->in_order_q_, d_a);
  compat::launch<local_mem_typed_kernel<TypeParam>>(
      this->grid_, this->thread_, this->memsize_, this->in_order_q_, d_a);

  sycl::ext::oneapi::experimental::compat::free((void *)d_a, this->in_order_q_);
}

TYPED_TEST(LocalMemLaunchTest1_t, LocalMemUsage) {
  TypeParam *h_a = (TypeParam *)std::malloc(this->memsize_ * sizeof(TypeParam));
  TypeParam *d_a =
      (TypeParam *)sycl::ext::oneapi::experimental::compat::malloc<TypeParam>(
          this->memsize_);

  // d_a is not used in the kernel
  // no init or memcpy to initialize is needed
  compat::launch<local_mem_typed_kernel<TypeParam>>(this->grid_, this->thread_,
                                                    this->memsize_, d_a);

  sycl::ext::oneapi::experimental::compat::memcpy(
      (void *)h_a, (void *)d_a, this->memsize_ * sizeof(TypeParam));
  sycl::ext::oneapi::experimental::compat::free((void *)d_a);

  for (int i = 0; i < this->memsize_; i++) {
    EXPECT_EQ(h_a[i], TypeParam(this->memsize_ - i - 1));
  }

  std::free(h_a);
}

TYPED_TEST(LocalMemLaunchTest1_t, LocalMemUsage_q) {

  TypeParam *h_a = (TypeParam *)std::malloc(this->memsize_ * sizeof(TypeParam));
  TypeParam *d_a =
      (TypeParam *)sycl::ext::oneapi::experimental::compat::malloc<TypeParam>(
          this->memsize_, this->in_order_q_);

  // d_a is not used in the kernel
  // no init or memcpy to initialize is needed
  compat::launch<local_mem_typed_kernel<TypeParam>>(
      this->grid_, this->thread_, this->memsize_, this->in_order_q_, d_a);

  sycl::ext::oneapi::experimental::compat::memcpy(
      (void *)h_a, (void *)d_a, this->memsize_ * sizeof(TypeParam),
      this->in_order_q_);
  sycl::ext::oneapi::experimental::compat::free((void *)d_a, this->in_order_q_);

  for (size_t i = 0; i < this->memsize_; i++) {
    EXPECT_EQ(h_a[i], TypeParam(this->memsize_ - i - 1));
  }

  std::free(h_a);
}

template <typename T> class LocalMemTypesTest1 : public LaunchTest1 {
public:
  LocalMemTypesTest1() : LaunchTest1() {}

  constexpr static size_t LOCAL_MEM_SIZE = 32;
};

using memsize_type_list = testing::Types<int, unsigned int, short,
                                         unsigned short, long, unsigned long>;
TYPED_TEST_SUITE(LocalMemTypesTest1, memsize_type_list);

TYPED_TEST(LocalMemTypesTest1, NoArgLaunch) {
  TypeParam memsize =
      static_cast<TypeParam>(LocalMemTypesTest1<TypeParam>::LOCAL_MEM_SIZE);

  compat::launch<local_mem_empty_kernel<int>>(this->range_1_, memsize);
  compat::launch<local_mem_empty_kernel<int>>(this->range_2_, memsize);
  compat::launch<local_mem_empty_kernel<int>>(this->range_3_, memsize);
  compat::launch<local_mem_empty_kernel<int>>(this->grid_, this->thread_,
                                              memsize);
}

TYPED_TEST(LocalMemTypesTest1, NoArgLaunch_q) {
  TypeParam memsize =
      static_cast<TypeParam>(LocalMemTypesTest1<TypeParam>::LOCAL_MEM_SIZE);

  compat::launch<local_mem_empty_kernel<int>>(this->range_1_, memsize,
                                              this->q_);
  compat::launch<local_mem_empty_kernel<int>>(this->range_2_, memsize,
                                              this->q_);
  compat::launch<local_mem_empty_kernel<int>>(this->range_3_, memsize,
                                              this->q_);
  compat::launch<local_mem_empty_kernel<int>>(this->grid_, this->thread_,
                                              memsize, this->q_);
}

TYPED_TEST(LocalMemTypesTest1, OneArgLaunch) {
  TypeParam memsize =
      static_cast<TypeParam>(LocalMemTypesTest1<TypeParam>::LOCAL_MEM_SIZE);
  int d_a;

  compat::launch<local_mem_basicdt_kernel<int>>(this->range_1_, memsize, d_a);
  compat::launch<local_mem_basicdt_kernel<int>>(this->range_2_, memsize, d_a);
  compat::launch<local_mem_basicdt_kernel<int>>(this->range_3_, memsize, d_a);
  compat::launch<local_mem_basicdt_kernel<int>>(this->grid_, this->thread_,
                                                memsize, d_a);
}

TYPED_TEST(LocalMemTypesTest1, OneArgLaunch_q) {
  TypeParam memsize =
      static_cast<TypeParam>(LocalMemTypesTest1<TypeParam>::LOCAL_MEM_SIZE);
  int d_a;

  compat::launch<local_mem_basicdt_kernel<int>>(this->range_1_, memsize,
                                                this->q_, d_a);
  compat::launch<local_mem_basicdt_kernel<int>>(this->range_2_, memsize,
                                                this->q_, d_a);
  compat::launch<local_mem_basicdt_kernel<int>>(this->range_3_, memsize,
                                                this->q_, d_a);
  compat::launch<local_mem_basicdt_kernel<int>>(this->grid_, this->thread_,
                                                memsize, this->q_, d_a);
}
