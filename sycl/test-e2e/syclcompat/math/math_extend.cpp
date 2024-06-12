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

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
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
    auto ret = S;                                                              \
    if (ret != REF) {                                                          \
      return {#S, REF};                                                        \
    }                                                                          \
  }

const auto INT32MAX = std::numeric_limits<int32_t>::max();
const auto INT32MIN = std::numeric_limits<int32_t>::min();
const auto UINT32MAX = std::numeric_limits<uint32_t>::max();
const auto UINT32MIN = std::numeric_limits<uint32_t>::min();
const int b = 4, c = 5, d = 6;

std::pair<const char *, int> vadd() {
  CHECK(syclcompat::extend_add<int32_t>(3, 4), 7);
  CHECK(syclcompat::extend_add<uint32_t>(b, c), 9);
  CHECK(syclcompat::extend_add_sat<int32_t>(b, INT32MAX), INT32MAX);
  CHECK(syclcompat::extend_add_sat<uint32_t>(UINT32MAX, INT32MAX), UINT32MAX);
  CHECK(syclcompat::extend_add_sat<int32_t>(b, -20, d, sycl::plus<>()), -10);
  CHECK(syclcompat::extend_add_sat<int32_t>(b, c, -20, sycl::minimum<>()), -20);
  CHECK(syclcompat::extend_add_sat<int32_t>(b, (-33), 9, sycl::maximum<>()), 9);

  return {nullptr, 0};
}

std::pair<const char *, int> vsub() {
  CHECK(syclcompat::extend_sub<int32_t>(3, 4), -1);
  CHECK(syclcompat::extend_sub<uint32_t>(c, b), 1);
  CHECK(syclcompat::extend_sub_sat<int32_t>(10, INT32MIN), INT32MAX);
  CHECK(syclcompat::extend_sub_sat<uint32_t>(UINT32MIN, 1), UINT32MIN);
  CHECK(syclcompat::extend_sub_sat<int32_t>(b, -20, d, sycl::plus<>()), 30);
  CHECK(syclcompat::extend_sub_sat<int32_t>(b, c, -20, sycl::minimum<>()), -20);
  CHECK(syclcompat::extend_sub_sat<int32_t>(b, (-33), 9, sycl::maximum<>()),
        37);

  return {nullptr, 0};
}

std::pair<const char *, int> vabsdiff() {
  CHECK(syclcompat::extend_absdiff<int32_t>(3, 4), 1);
  CHECK(syclcompat::extend_absdiff<uint32_t>(c, b), 1);
  CHECK(syclcompat::extend_absdiff_sat<int32_t>(10, INT32MIN), INT32MAX);
  CHECK(syclcompat::extend_absdiff_sat<uint32_t>(UINT32MIN, 1), 1);
  CHECK(syclcompat::extend_absdiff_sat<int32_t>(b, -20, d, sycl::plus<>()), 30);
  CHECK(syclcompat::extend_absdiff_sat<int32_t>(b, c, -20, sycl::minimum<>()),
        -20);
  CHECK(syclcompat::extend_absdiff_sat<int32_t>(b, (-33), 9, sycl::maximum<>()),
        37);

  return {nullptr, 0};
}

std::pair<const char *, int> vmin() {
  CHECK(syclcompat::extend_min<int32_t>(3, 4), 3);
  CHECK(syclcompat::extend_min<uint32_t>(c, b), 4);
  CHECK(syclcompat::extend_min_sat<int32_t>(UINT32MAX, 1), 1);
  CHECK(syclcompat::extend_min_sat<uint32_t>(10, (-1)), 0);
  CHECK(syclcompat::extend_min_sat<int32_t>(b, -20, d, sycl::plus<>()), -14);
  CHECK(syclcompat::extend_min_sat<int32_t>(b, c, -20, sycl::minimum<>()), -20);
  CHECK(syclcompat::extend_min_sat<int32_t>(b, (-33), 9, sycl::maximum<>()), 9);

  return {nullptr, 0};
}

std::pair<const char *, int> vmax() {
  CHECK(syclcompat::extend_max<int32_t>(3, 4), 4);
  CHECK(syclcompat::extend_max<uint32_t>(c, b), 5);
  CHECK(syclcompat::extend_max_sat<int32_t>(UINT32MAX, 1), INT32MAX);
  CHECK(syclcompat::extend_max_sat<uint32_t>(UINT32MAX, 1), UINT32MAX);
  CHECK(syclcompat::extend_max_sat<int32_t>(b, -20, d, sycl::plus<>()), 10);
  CHECK(syclcompat::extend_max_sat<int32_t>(b, c, -20, sycl::minimum<>()), -20);
  CHECK(syclcompat::extend_max_sat<int32_t>(b, (-33), 9, sycl::maximum<>()), 9);

  return {nullptr, 0};
}

template <typename Tp> struct scale {
  Tp operator()(Tp val, Tp scaler) { return val * scaler; }
};

template <typename Tp> struct noop {
  Tp operator()(Tp val, Tp /*scaler*/) { return val; }
};

std::pair<const char *, int> shl_clamp() {
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

  return {nullptr, 0};
}

std::pair<const char *, int> shl_wrap() {
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

  return {nullptr, 0};
}

std::pair<const char *, int> shr_clamp() {
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

  return {nullptr, 0};
}

std::pair<const char *, int> shr_wrap() {
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

  return {nullptr, 0};
}

void test(const sycl::stream &s, int *ec) {
  {
    auto res = vadd();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 1;
      return;
    }
    s << "vadd check passed!\n";
  }
  {
    auto res = vsub();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 2;
      return;
    }
    s << "vsub check passed!\n";
  }
  {
    auto res = vabsdiff();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 3;
      return;
    }
    s << "vabsdiff check passed!\n";
  }
  {
    auto res = vmin();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 4;
      return;
    }
    s << "vmin check passed!\n";
  }
  {
    auto res = vmax();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 5;
      return;
    }
    s << "vmax check passed!\n";
  }
  {
    auto res = shl_clamp();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 6;
      return;
    }
    s << "shl_clamp check passed!\n";
  }
  {
    auto res = shl_wrap();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 7;
      return;
    }
    s << "shl_wrap check passed!\n";
  }
  {
    auto res = shr_clamp();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 8;
      return;
    }
    s << "shr_clamp check passed!\n";
  }
  {
    auto res = shr_wrap();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 9;
      return;
    }
    s << "shr_wrap check passed!\n";
  }
  *ec = 0;
}

int main() {
  sycl::queue q = syclcompat::get_default_queue();
  int *ec = syclcompat::malloc<int>(1);
  syclcompat::fill<int>(ec, 0, 1);
  q.submit([&](sycl::handler &cgh) {
    sycl::stream out(1024, 256, cgh);
    cgh.parallel_for(1, [=](sycl::item<1> it) { test(out, ec); });
  });
  q.wait_and_throw();

  int ec_h;
  syclcompat::memcpy<int>(&ec_h, ec, 1);

  return ec_h;
}
