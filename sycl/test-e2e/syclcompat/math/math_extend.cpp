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
 *  SYCLcompat API
 *
 *  math_extend.cpp
 *
 *  Description:
 *    math extend helpers tests
 **************************************************************************/

// ===----------- math_extend_func.cpp ---------- -*- C++ -* --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdio.h>
#include <sycl/detail/core.hpp>

#include <syclcompat/device.hpp>
#include <syclcompat/math.hpp>
#include <syclcompat/memory.hpp>

#define CHECK(S, REF)                                                          \
  {                                                                            \
    ++test_id;                                                                 \
    auto ret = S;                                                              \
    if (ret != REF) {                                                          \
      errc = test_id;                                                          \
    }                                                                          \
  }

const auto INT32MAX = std::numeric_limits<int32_t>::max();
const auto INT32MIN = std::numeric_limits<int32_t>::min();
const auto UINT32MAX = std::numeric_limits<uint32_t>::max();
const auto UINT32MIN = std::numeric_limits<uint32_t>::min();
const int b = 4, c = 5, d = 6;

int vadd() {
  int errc{};
  int test_id{};
  CHECK(syclcompat::extend_add<int32_t>(3, 4), 7);
  CHECK(syclcompat::extend_add<uint32_t>(b, c), 9);
  CHECK(syclcompat::extend_add_sat<int32_t>(b, INT32MAX), INT32MAX);
  CHECK(syclcompat::extend_add_sat<uint32_t>(UINT32MAX, INT32MAX), UINT32MAX);
  CHECK(syclcompat::extend_add_sat<int32_t>(b, -20, d, sycl::plus<>()), -10);
  CHECK(syclcompat::extend_add_sat<int32_t>(b, c, -20, sycl::minimum<>()), -20);
  CHECK(syclcompat::extend_add_sat<int32_t>(b, (-33), 9, sycl::maximum<>()), 9);

  return errc;
}

int vsub() {
  int errc{};
  int test_id{};
  CHECK(syclcompat::extend_sub<int32_t>(3, 4), -1);
  CHECK(syclcompat::extend_sub<uint32_t>(c, b), 1);
  CHECK(syclcompat::extend_sub_sat<int32_t>(10, INT32MIN), INT32MAX);
  CHECK(syclcompat::extend_sub_sat<uint32_t>(UINT32MIN, 1), UINT32MIN);
  CHECK(syclcompat::extend_sub_sat<int32_t>(b, -20, d, sycl::plus<>()), 30);
  CHECK(syclcompat::extend_sub_sat<int32_t>(b, c, -20, sycl::minimum<>()), -20);
  CHECK(syclcompat::extend_sub_sat<int32_t>(b, (-33), 9, sycl::maximum<>()),
        37);

  return errc;
}

int vabsdiff() {
  int errc{};
  int test_id{};
  CHECK(syclcompat::extend_absdiff<int32_t>(3, 4), 1);
  CHECK(syclcompat::extend_absdiff<uint32_t>(c, b), 1);
  CHECK(syclcompat::extend_absdiff_sat<int32_t>(10, INT32MIN), INT32MAX);
  CHECK(syclcompat::extend_absdiff_sat<uint32_t>(UINT32MIN, 1), 1);
  CHECK(syclcompat::extend_absdiff_sat<int32_t>(b, -20, d, sycl::plus<>()), 30);
  CHECK(syclcompat::extend_absdiff_sat<int32_t>(b, c, -20, sycl::minimum<>()),
        -20);
  CHECK(syclcompat::extend_absdiff_sat<int32_t>(b, (-33), 9, sycl::maximum<>()),
        37);

  return errc;
}

int vmin() {
  int errc{};
  int test_id{};
  CHECK(syclcompat::extend_min<int32_t>(3, 4), 3);
  CHECK(syclcompat::extend_min<uint32_t>(c, b), 4);
  CHECK(syclcompat::extend_min_sat<int32_t>(UINT32MAX, 1), 1);
  CHECK(syclcompat::extend_min_sat<uint32_t>(10, (-1)), 0);
  CHECK(syclcompat::extend_min_sat<int32_t>(b, -20, d, sycl::plus<>()), -14);
  CHECK(syclcompat::extend_min_sat<int32_t>(b, c, -20, sycl::minimum<>()), -20);
  CHECK(syclcompat::extend_min_sat<int32_t>(b, (-33), 9, sycl::maximum<>()), 9);

  return errc;
}

int vmax() {
  int errc{};
  int test_id{};
  CHECK(syclcompat::extend_max<int32_t>(3, 4), 4);
  CHECK(syclcompat::extend_max<uint32_t>(c, b), 5);
  CHECK(syclcompat::extend_max_sat<int32_t>(UINT32MAX, 1), INT32MAX);
  CHECK(syclcompat::extend_max_sat<uint32_t>(UINT32MAX, 1), UINT32MAX);
  CHECK(syclcompat::extend_max_sat<int32_t>(b, -20, d, sycl::plus<>()), 10);
  CHECK(syclcompat::extend_max_sat<int32_t>(b, c, -20, sycl::minimum<>()), -20);
  CHECK(syclcompat::extend_max_sat<int32_t>(b, (-33), 9, sycl::maximum<>()), 9);

  return errc;
}

template <typename Tp> struct scale {
  Tp operator()(Tp val, Tp scaler) { return val * scaler; }
};

template <typename Tp> struct noop {
  Tp operator()(Tp val, Tp /*scaler*/) { return val; }
};

int shl_clamp() {
  int errc{};
  int test_id{};
  CHECK(syclcompat::extend_shl_clamp<int32_t>(3, 4), 48);
  CHECK(syclcompat::extend_shl_clamp<int32_t>(6, 33), 0);
  CHECK(syclcompat::extend_shl_clamp<int32_t>(3, 4, 4, scale<int32_t>()), 192);
  CHECK(syclcompat::extend_shl_clamp<int32_t>(3, 4, 4, noop<int32_t>()), 48);
  CHECK(syclcompat::extend_shl_sat_clamp<int8_t>(9, 5), 127);
  CHECK(syclcompat::extend_shl_sat_clamp<int8_t>(-9, 5), -128);
  CHECK(syclcompat::extend_shl_sat_clamp<int8_t>(9, 5, -1, scale<int8_t>()),
        -127);
  CHECK(syclcompat::extend_shl_sat_clamp<int8_t>(9, 5, -1, noop<int8_t>()),
        127);

  return errc;
}

int shl_wrap() {
  int errc{};
  int test_id{};
  CHECK(syclcompat::extend_shl_wrap<int32_t>(3, 4), 48);
  CHECK(syclcompat::extend_shl_wrap<int32_t>(6, 32), 6);
  CHECK(syclcompat::extend_shl_wrap<int32_t>(6, 33), 12);
  CHECK(syclcompat::extend_shl_wrap<int32_t>(6, 64), 6);
  CHECK(syclcompat::extend_shl_wrap<int32_t>(3, 4, 4, scale<int32_t>()), 192);
  CHECK(syclcompat::extend_shl_wrap<int32_t>(6, 32, 4, noop<int32_t>()), 6);
  CHECK(syclcompat::extend_shl_sat_wrap<int8_t>(9, 5), 127);
  CHECK(syclcompat::extend_shl_sat_wrap<int8_t>(-9, 5), -128);
  CHECK(syclcompat::extend_shl_sat_wrap<int8_t>(9, 5, -1, scale<int8_t>()),
        -127);
  CHECK(syclcompat::extend_shl_sat_wrap<int8_t>(9, 5, -1, noop<int8_t>()), 127);

  return errc;
}

int shr_clamp() {
  int errc{};
  int test_id{};
  CHECK(syclcompat::extend_shr_clamp<int32_t>(128, 5), 4);
  CHECK(syclcompat::extend_shr_clamp<int32_t>(INT32MAX, 33), 0);
  CHECK(syclcompat::extend_shr_clamp<int32_t>(128, 5, 4, scale<int32_t>()), 16);
  CHECK(syclcompat::extend_shr_clamp<int32_t>(128, 5, 4, noop<int32_t>()), 4);
  CHECK(syclcompat::extend_shr_sat_clamp<int8_t>(512, 1), 127);
  CHECK(syclcompat::extend_shr_sat_clamp<int8_t>(-512, 1), -128);
  CHECK(syclcompat::extend_shr_sat_clamp<int8_t>(512, 1, -1, scale<int8_t>()),
        -127);
  CHECK(syclcompat::extend_shr_sat_clamp<int8_t>(512, 1, -1, noop<int8_t>()),
        127);

  return errc;
}

int shr_wrap() {
  int errc{};
  int test_id{};
  CHECK(syclcompat::extend_shr_wrap<int32_t>(128, 5), 4);
  CHECK(syclcompat::extend_shr_wrap<int32_t>(128, 32), 128);
  CHECK(syclcompat::extend_shr_wrap<int32_t>(128, 33), 64);
  CHECK(syclcompat::extend_shr_wrap<int32_t>(128, 64), 128);
  CHECK(syclcompat::extend_shr_wrap<int32_t>(128, 5, 4, scale<int32_t>()), 16);
  CHECK(syclcompat::extend_shr_wrap<int32_t>(128, 5, 4, noop<int32_t>()), 4);
  CHECK(syclcompat::extend_shr_sat_wrap<int8_t>(512, 1), 127);
  CHECK(syclcompat::extend_shr_sat_wrap<int8_t>(-512, 1), -128);
  CHECK(syclcompat::extend_shr_sat_wrap<int8_t>(512, 1, -1, scale<int8_t>()),
        -127);
  CHECK(syclcompat::extend_shr_sat_wrap<int8_t>(512, 1, -1, noop<int8_t>()),
        127);

  return errc;
}

template <auto F> void test_fn(sycl::queue q, int *ec) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  q.submit([&](sycl::handler &cgh) {
    cgh.single_task([=]() {
      auto res = F();
      if(res != 0) *ec = res;
    });
  });
  int ec_h{};
  syclcompat::memcpy<int>(&ec_h, ec, 1, q);
  q.wait_and_throw();

  if (ec_h != 0) {
    std::cout << "Test " << ec_h << " failed." << std::endl;
    syclcompat::free(ec, q);
    assert(false);
  }
}

int main() {
  sycl::queue q = syclcompat::get_default_queue();
  int *ec = syclcompat::malloc<int>(1, q);
  syclcompat::fill<int>(ec, 0, 1, q);

  test_fn<vadd>(q, ec);
  test_fn<vsub>(q, ec);
  test_fn<vabsdiff>(q, ec);
  test_fn<vmin>(q, ec);
  test_fn<vmax>(q, ec);
  test_fn<shl_clamp>(q, ec);
  test_fn<shl_wrap>(q, ec);
  test_fn<shr_clamp>(q, ec);
  test_fn<shr_wrap>(q, ec);

  syclcompat::free(ec, q);
}
