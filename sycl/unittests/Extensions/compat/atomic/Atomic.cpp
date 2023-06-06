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
 *  Atomic.cpp
 *
 *  Description:
 *    atomic operations API tests
 **************************************************************************/

// The original source was under the license below:
// ====------ Atomic.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <sycl/ext/oneapi/experimental/compat.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>

using namespace sycl::ext::oneapi::experimental;

// Simple atomic kernels for testing
// In every case we test two API overloads, one taking an explicit runtime
// memory_order argument. We use `relaxed` in every case because these tests
// are *not* checking the memory_order semantics, just the API.
template <typename T, bool orderArg = false>
void atomic_fetch_add_kernel(T *data, compat::arith_t<T> operand) {
  if constexpr (orderArg) {
    compat::atomic_fetch_add(data, operand, sycl::memory_order::relaxed);
  } else {
    compat::atomic_fetch_add(data, operand);
  }
}
template <typename T, bool orderArg = false>
void atomic_fetch_sub_kernel(T *data, compat::arith_t<T> operand) {
  if constexpr (orderArg) {
    compat::atomic_fetch_sub(data, operand, sycl::memory_order::relaxed);
  } else {
    compat::atomic_fetch_sub(data, operand);
  }
}
template <typename T, bool orderArg = false>
void atomic_fetch_and_kernel(T *data, T operand, T operand0) {
  if constexpr (orderArg) {
    compat::atomic_fetch_and(data,
                             (compat::global_id::x() == 0 ? operand0 : operand),
                             sycl::memory_order::relaxed);
  } else {
    compat::atomic_fetch_and(
        data, (compat::global_id::x() == 0 ? operand0 : operand));
  }
}
template <typename T, bool orderArg = false>
void atomic_fetch_or_kernel(T *data, T operand, T operand0) {
  if constexpr (orderArg) {
    compat::atomic_fetch_or(data,
                            (compat::global_id::x() == 0 ? operand0 : operand),
                            sycl::memory_order::relaxed);
  } else {
    compat::atomic_fetch_or(data,
                            (compat::global_id::x() == 0 ? operand0 : operand));
  }
}
template <typename T, bool orderArg = false>
void atomic_fetch_xor_kernel(T *data, T operand, T operand0) {
  if constexpr (orderArg) {
    compat::atomic_fetch_xor(data,
                             (compat::global_id::x() == 0 ? operand0 : operand),
                             sycl::memory_order::relaxed);
  } else {
    compat::atomic_fetch_xor(
        data, (compat::global_id::x() == 0 ? operand0 : operand));
  }
}
template <typename T, bool orderArg = false>
void atomic_fetch_min_kernel(T *data, T operand, T operand0) {
  if constexpr (orderArg) {
    compat::atomic_fetch_min(data,
                             (compat::global_id::x() == 0 ? operand0 : operand),
                             sycl::memory_order::relaxed);
  } else {
    compat::atomic_fetch_min(
        data, (compat::global_id::x() == 0 ? operand0 : operand));
  }
}
template <typename T, bool orderArg = false>
void atomic_fetch_max_kernel(T *data, T operand, T operand0) {
  if constexpr (orderArg) {
    compat::atomic_fetch_max(data,
                             (compat::global_id::x() == 0 ? operand0 : operand),
                             sycl::memory_order::relaxed);
  } else {
    compat::atomic_fetch_max(
        data, (compat::global_id::x() == 0 ? operand0 : operand));
  }
}
template <typename T, bool orderArg = false>
void atomic_fetch_compare_inc_kernel(T *data, T operand) {
  if constexpr (orderArg) {
    compat::atomic_fetch_compare_inc(data, operand,
                                     sycl::memory_order::relaxed);
  } else {
    compat::atomic_fetch_compare_inc(data, operand);
  }
}
template <typename T, bool orderArg = false>
void atomic_exchange_kernel(T *data, T operand) {
  if constexpr (orderArg) {
    compat::atomic_exchange(data, operand, sycl::memory_order::relaxed);
  } else {
    compat::atomic_exchange(data, operand);
  }
}
template <typename T, bool orderArg = false>
void atomic_compare_exchange_strong_kernel(T *data, T expected, T desired) {
  if constexpr (orderArg) {
    compat::atomic_compare_exchange_strong(data, expected, desired,
                                           sycl::memory_order::relaxed);
  } else {
    compat::atomic_compare_exchange_strong(data, expected, desired);
  }
}

template <auto F, typename T> class AtomicLauncher {
protected:
  compat::dim3 grid_;
  compat::dim3 threads_;
  T *data_;

public:
  AtomicLauncher(compat::dim3 grid, compat::dim3 threads)
      : grid_{grid}, threads_{threads} {
    data_ = (T *)compat::malloc(sizeof(T));
  };
  ~AtomicLauncher() { compat::free(data_); }
  template <typename... Args>
  void launch_test(T init_val, T expected_result, Args... args) {
    if (!compat::get_current_device().has(sycl::aspect::fp64) &&
        (std::is_same_v<T, double> || std::is_same_v<T, double *>))
      GTEST_SKIP();
    compat::memcpy(data_, &init_val, sizeof(T));
    compat::launch<F>(grid_, threads_, data_, args...);
    T result_val;
    compat::memcpy(&result_val, data_, sizeof(T));
    compat::wait();
    EXPECT_EQ(result_val, expected_result);
  }
};

using atomic_type_list =
    testing::Types<int, unsigned int, long, unsigned long, long long,
                   unsigned long long, float, double, int *, long *,
                   long long *, float *, double *>;
template <class> struct atomic_suite : testing::Test {};
TYPED_TEST_SUITE(atomic_suite, atomic_type_list);

using atomic_value_type_list =
    testing::Types<int, unsigned int, long, unsigned long, long long,
                   unsigned long long, float, double>;
template <class> struct atomic_value_suite : testing::Test {};
TYPED_TEST_SUITE(atomic_value_suite, atomic_value_type_list);

using atomic_ptr_type_list =
    testing::Types<int *, long *, long long *, float *, double *>;
template <class> struct atomic_ptr_suite : testing::Test {};
TYPED_TEST_SUITE(atomic_ptr_suite, atomic_ptr_type_list);

using signed_type_list = testing::Types<int, long, long long, float, double>;
template <class> struct atomic_signed_suite : testing::Test {};
TYPED_TEST_SUITE(atomic_signed_suite, signed_type_list);

using integral_type_list =
    testing::Types<int, unsigned int, long, unsigned long, long long,
                   unsigned long long>;
template <class> struct atomic_logic_suite : testing::Test {};
TYPED_TEST_SUITE(atomic_logic_suite, integral_type_list);

TYPED_TEST(atomic_value_suite, atomic_arith) {
  constexpr compat::dim3 grid{4};
  constexpr compat::dim3 threads{32};
  constexpr TypeParam sum = static_cast<TypeParam>(grid.x * threads.x);
  constexpr TypeParam init = static_cast<TypeParam>(0);
  constexpr TypeParam operand = static_cast<TypeParam>(1);

  AtomicLauncher<atomic_fetch_add_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(init, sum, operand);
  AtomicLauncher<atomic_fetch_sub_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(sum, init, operand);
  AtomicLauncher<atomic_fetch_add_kernel<TypeParam, true>, TypeParam>(grid,
                                                                      threads)
      .launch_test(init, sum, operand);
  AtomicLauncher<atomic_fetch_sub_kernel<TypeParam, true>, TypeParam>(grid,
                                                                      threads)
      .launch_test(sum, init, operand);
}

TYPED_TEST(atomic_ptr_suite, atomic_arith) {
  constexpr compat::dim3 grid{4};
  constexpr compat::dim3 threads{32};

  using ValType = std::remove_pointer_t<TypeParam>;

  TypeParam init = (TypeParam)compat::malloc(sizeof(ValType));
  TypeParam final = init + (grid.x * threads.x);
  constexpr std::ptrdiff_t operand = static_cast<std::ptrdiff_t>(1);

  AtomicLauncher<atomic_fetch_add_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(init, final, operand);

  AtomicLauncher<atomic_fetch_sub_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(final, init, operand);

  AtomicLauncher<atomic_fetch_add_kernel<TypeParam, true>, TypeParam>(grid,
                                                                      threads)
      .launch_test(init, final, operand);

  AtomicLauncher<atomic_fetch_sub_kernel<TypeParam, true>, TypeParam>(grid,
                                                                      threads)
      .launch_test(final, init, operand);
  compat::free(init);
}

TYPED_TEST(atomic_value_suite, atomic_minmax) {
  constexpr compat::dim3 grid{4};
  constexpr compat::dim3 threads{32};

  AtomicLauncher<atomic_fetch_min_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(100), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(200), static_cast<TypeParam>(1));
  AtomicLauncher<atomic_fetch_max_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(100), static_cast<TypeParam>(200),
                   static_cast<TypeParam>(200), static_cast<TypeParam>(1));
  AtomicLauncher<atomic_fetch_min_kernel<TypeParam, true>, TypeParam>(grid,
                                                                      threads)
      .launch_test(static_cast<TypeParam>(100), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(200), static_cast<TypeParam>(1));
  AtomicLauncher<atomic_fetch_max_kernel<TypeParam, true>, TypeParam>(grid,
                                                                      threads)
      .launch_test(static_cast<TypeParam>(100), static_cast<TypeParam>(200),
                   static_cast<TypeParam>(200), static_cast<TypeParam>(1));
}

TYPED_TEST(atomic_signed_suite, signed_atomic_minmax) {
  constexpr compat::dim3 grid{4};
  constexpr compat::dim3 threads{32};

  AtomicLauncher<atomic_fetch_min_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(-1), static_cast<TypeParam>(-4),
                   static_cast<TypeParam>(-4), static_cast<TypeParam>(100));
  AtomicLauncher<atomic_fetch_max_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(-40), static_cast<TypeParam>(-30),
                   static_cast<TypeParam>(-30), static_cast<TypeParam>(-100));
  AtomicLauncher<atomic_fetch_min_kernel<TypeParam, true>, TypeParam>(grid,
                                                                      threads)
      .launch_test(static_cast<TypeParam>(-1), static_cast<TypeParam>(-4),
                   static_cast<TypeParam>(-4), static_cast<TypeParam>(100));
  AtomicLauncher<atomic_fetch_max_kernel<TypeParam, true>, TypeParam>(grid,
                                                                      threads)
      .launch_test(static_cast<TypeParam>(-40), static_cast<TypeParam>(-30),
                   static_cast<TypeParam>(-30), static_cast<TypeParam>(-100));
}

TYPED_TEST(atomic_logic_suite, atomic_and) {
  constexpr compat::dim3 grid{4};
  constexpr compat::dim3 threads{32};

  // All 0 -> 0
  AtomicLauncher<atomic_fetch_and_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(0), static_cast<TypeParam>(0),
                   static_cast<TypeParam>(0), static_cast<TypeParam>(0));

  // All 1 -> 1
  AtomicLauncher<atomic_fetch_and_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(1), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(1), static_cast<TypeParam>(1));
  // Most 1, one 0 -> 0
  AtomicLauncher<atomic_fetch_and_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(1), static_cast<TypeParam>(0),
                   static_cast<TypeParam>(1), static_cast<TypeParam>(0));

  // All 0 -> 0
  AtomicLauncher<atomic_fetch_and_kernel<TypeParam, true>, TypeParam>(grid,
                                                                      threads)
      .launch_test(static_cast<TypeParam>(0), static_cast<TypeParam>(0),
                   static_cast<TypeParam>(0), static_cast<TypeParam>(0));

  // All 1 -> 1
  AtomicLauncher<atomic_fetch_and_kernel<TypeParam, true>, TypeParam>(grid,
                                                                      threads)
      .launch_test(static_cast<TypeParam>(1), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(1), static_cast<TypeParam>(1));
  // Most 1, one 0 -> 0
  AtomicLauncher<atomic_fetch_and_kernel<TypeParam, true>, TypeParam>(grid,
                                                                      threads)
      .launch_test(static_cast<TypeParam>(1), static_cast<TypeParam>(0),
                   static_cast<TypeParam>(1), static_cast<TypeParam>(0));
}

TYPED_TEST(atomic_logic_suite, atomic_or) {
  constexpr compat::dim3 grid{4};
  constexpr compat::dim3 threads{32};

  // All 0 -> 0
  AtomicLauncher<atomic_fetch_or_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(0), static_cast<TypeParam>(0),
                   static_cast<TypeParam>(0), static_cast<TypeParam>(0));
  // All 1 -> 1
  AtomicLauncher<atomic_fetch_or_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(1), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(1), static_cast<TypeParam>(1));
  // Most 1, one 0 -> 1
  AtomicLauncher<atomic_fetch_or_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(1), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(1), static_cast<TypeParam>(0));
  // Init 1, all 0 -> 1
  AtomicLauncher<atomic_fetch_or_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(1), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(0), static_cast<TypeParam>(0));

  // All 0 -> 0
  AtomicLauncher<atomic_fetch_or_kernel<TypeParam, true>, TypeParam>(grid,
                                                                     threads)
      .launch_test(static_cast<TypeParam>(0), static_cast<TypeParam>(0),
                   static_cast<TypeParam>(0), static_cast<TypeParam>(0));
  // All 1 -> 1
  AtomicLauncher<atomic_fetch_or_kernel<TypeParam, true>, TypeParam>(grid,
                                                                     threads)
      .launch_test(static_cast<TypeParam>(1), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(1), static_cast<TypeParam>(1));
  // Most 1, one 0 -> 1
  AtomicLauncher<atomic_fetch_or_kernel<TypeParam, true>, TypeParam>(grid,
                                                                     threads)
      .launch_test(static_cast<TypeParam>(1), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(1), static_cast<TypeParam>(0));
  // Init 1, all 0 -> 1
  AtomicLauncher<atomic_fetch_or_kernel<TypeParam, true>, TypeParam>(grid,
                                                                     threads)
      .launch_test(static_cast<TypeParam>(1), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(0), static_cast<TypeParam>(0));
}

TYPED_TEST(atomic_logic_suite, atomic_xor) {
  constexpr compat::dim3 grid{1};
  constexpr compat::dim3 threads{2}; // 2 threads, 3 values inc. init

  // 000 -> 0
  AtomicLauncher<atomic_fetch_xor_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(0), static_cast<TypeParam>(0),
                   static_cast<TypeParam>(0), static_cast<TypeParam>(0));
  // 111 -> 1
  AtomicLauncher<atomic_fetch_xor_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(1), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(1), static_cast<TypeParam>(1));
  // 110 -> 0
  AtomicLauncher<atomic_fetch_xor_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(1), static_cast<TypeParam>(0),
                   static_cast<TypeParam>(1), static_cast<TypeParam>(0));
  // 010 -> 1
  AtomicLauncher<atomic_fetch_xor_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(0), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(1), static_cast<TypeParam>(0));

  // 000 -> 0
  AtomicLauncher<atomic_fetch_xor_kernel<TypeParam, true>, TypeParam>(grid,
                                                                      threads)
      .launch_test(static_cast<TypeParam>(0), static_cast<TypeParam>(0),
                   static_cast<TypeParam>(0), static_cast<TypeParam>(0));
  // 111 -> 1
  AtomicLauncher<atomic_fetch_xor_kernel<TypeParam, true>, TypeParam>(grid,
                                                                      threads)
      .launch_test(static_cast<TypeParam>(1), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(1), static_cast<TypeParam>(1));
  // 110 -> 0
  AtomicLauncher<atomic_fetch_xor_kernel<TypeParam, true>, TypeParam>(grid,
                                                                      threads)
      .launch_test(static_cast<TypeParam>(1), static_cast<TypeParam>(0),
                   static_cast<TypeParam>(1), static_cast<TypeParam>(0));
  // 010 -> 1
  AtomicLauncher<atomic_fetch_xor_kernel<TypeParam, true>, TypeParam>(grid,
                                                                      threads)
      .launch_test(static_cast<TypeParam>(0), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(1), static_cast<TypeParam>(0));
}

TEST(atomic_compex, atomic_comp) {
  constexpr compat::dim3 grid{1};
  constexpr compat::dim3 threads{6};

  AtomicLauncher<atomic_fetch_compare_inc_kernel<unsigned int>, unsigned int>(
      grid, threads)
      .launch_test(0, 6, 6);
  AtomicLauncher<atomic_fetch_compare_inc_kernel<unsigned int>, unsigned int>(
      grid, threads)
      .launch_test(1, 0, 6);

  AtomicLauncher<atomic_fetch_compare_inc_kernel<unsigned int, true>,
                 unsigned int>(grid, threads)
      .launch_test(0, 6, 6);
  AtomicLauncher<atomic_fetch_compare_inc_kernel<unsigned int, true>,
                 unsigned int>(grid, threads)
      .launch_test(1, 0, 6);
}

TYPED_TEST(atomic_value_suite, atomic_exch) {
  constexpr compat::dim3 grid{4};
  constexpr compat::dim3 threads{32};

  AtomicLauncher<atomic_exchange_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(0), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(1));
  AtomicLauncher<atomic_exchange_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(0), static_cast<TypeParam>(0),
                   static_cast<TypeParam>(0));
  AtomicLauncher<atomic_exchange_kernel<TypeParam, true>, TypeParam>(grid,
                                                                     threads)
      .launch_test(static_cast<TypeParam>(0), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(1));
  AtomicLauncher<atomic_exchange_kernel<TypeParam, true>, TypeParam>(grid,
                                                                     threads)
      .launch_test(static_cast<TypeParam>(0), static_cast<TypeParam>(0),
                   static_cast<TypeParam>(0));
}

TYPED_TEST(atomic_ptr_suite, atomic_exch) {
  constexpr compat::dim3 grid{4};
  constexpr compat::dim3 threads{32};

  using ValType = std::remove_pointer_t<TypeParam>;
  TypeParam ptr1 = (TypeParam)compat::malloc(sizeof(ValType));
  TypeParam ptr2 = (TypeParam)compat::malloc(sizeof(ValType));

  AtomicLauncher<atomic_exchange_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(ptr1, ptr2, ptr2);
  AtomicLauncher<atomic_exchange_kernel<TypeParam>, TypeParam>(grid, threads)
      .launch_test(ptr1, ptr1, ptr1);
  AtomicLauncher<atomic_exchange_kernel<TypeParam, true>, TypeParam>(grid,
                                                                     threads)
      .launch_test(ptr1, ptr2, ptr2);
  AtomicLauncher<atomic_exchange_kernel<TypeParam, true>, TypeParam>(grid,
                                                                     threads)
      .launch_test(ptr1, ptr1, ptr1);
  compat::free(ptr1);
  compat::free(ptr2);
}

TYPED_TEST(atomic_value_suite, atomic_exch_strong) {
  constexpr compat::dim3 grid{4};
  constexpr compat::dim3 threads{32};

  AtomicLauncher<atomic_compare_exchange_strong_kernel<TypeParam>, TypeParam>(
      grid, threads)
      .launch_test(static_cast<TypeParam>(0), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(0), static_cast<TypeParam>(1));
  AtomicLauncher<atomic_compare_exchange_strong_kernel<TypeParam>, TypeParam>(
      grid, threads)
      .launch_test(static_cast<TypeParam>(0), static_cast<TypeParam>(0),
                   static_cast<TypeParam>(1), static_cast<TypeParam>(2));
  AtomicLauncher<atomic_compare_exchange_strong_kernel<TypeParam, true>,
                 TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(0), static_cast<TypeParam>(1),
                   static_cast<TypeParam>(0), static_cast<TypeParam>(1));
  AtomicLauncher<atomic_compare_exchange_strong_kernel<TypeParam, true>,
                 TypeParam>(grid, threads)
      .launch_test(static_cast<TypeParam>(0), static_cast<TypeParam>(0),
                   static_cast<TypeParam>(1), static_cast<TypeParam>(2));
}

TYPED_TEST(atomic_ptr_suite, atomic_exch_strong) {
  constexpr compat::dim3 grid{4};
  constexpr compat::dim3 threads{32};

  using ValType = std::remove_pointer_t<TypeParam>;
  TypeParam ptr1 = (TypeParam)compat::malloc(sizeof(ValType));
  TypeParam ptr2 = (TypeParam)compat::malloc(sizeof(ValType));
  TypeParam ptr3 = (TypeParam)compat::malloc(sizeof(ValType));

  AtomicLauncher<atomic_compare_exchange_strong_kernel<TypeParam>, TypeParam>(
      grid, threads)
      .launch_test(ptr1, ptr2, ptr1, ptr2);
  AtomicLauncher<atomic_compare_exchange_strong_kernel<TypeParam>, TypeParam>(
      grid, threads)
      .launch_test(ptr1, ptr1, ptr2, ptr3);
  AtomicLauncher<atomic_compare_exchange_strong_kernel<TypeParam, true>,
                 TypeParam>(grid, threads)
      .launch_test(ptr1, ptr2, ptr1, ptr2);
  AtomicLauncher<atomic_compare_exchange_strong_kernel<TypeParam, true>,
                 TypeParam>(grid, threads)
      .launch_test(ptr1, ptr1, ptr2, ptr3);
  compat::free(ptr1);
  compat::free(ptr2);
  compat::free(ptr3);
}
