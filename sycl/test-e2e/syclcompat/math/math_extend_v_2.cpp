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
 *  math_extend_v_2.cpp
 *
 *  Description:
 *    math extend 2-vectorized helpers tests
 **************************************************************************/

// ===------------- math_extend_vfunc_2.cpp ----------------*- C++ -*-----===//
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

std::pair<const char *, int> vadd2() {
  CHECK(syclcompat::extend_vadd2<int32_t>(0x0001FFFF, 0x00010005, 0),
        0x00020004);
  CHECK(syclcompat::extend_vadd2<int32_t>(0x7FFF7FFF, 0x00010001, 0),
        0x80008000);
  CHECK(syclcompat::extend_vadd2_sat<int32_t>(0x7FFF7FFF, 0x00010001, 0),
        0x7FFF7FFF);

  CHECK(syclcompat::extend_vadd2<uint32_t>(0x00010002, 0x00020003, 0),
        0x00030005);
  CHECK(syclcompat::extend_vadd2<uint32_t>(0xFFFEFFFF, 0x00030003, 0),
        0x00010002);
  CHECK(syclcompat::extend_vadd2_sat<uint32_t>((uint32_t)0xFFFEFFFF,
                                               (uint32_t)0x00030003, 0),
        0xFFFFFFFF);
  return {nullptr, 0};
}

std::pair<const char *, int> vsub2() {

  CHECK(syclcompat::extend_vsub2<int32_t>(0x0001FFFF, 0xFFFF0001, 0),
        0x0002FFFE);
  // Testing API & Saturated API with mixed types
  CHECK(syclcompat::extend_vsub2<int32_t>((int32_t)0x7FFFFFFD,
                                          (int32_t)0xFFFA7FFF, 0),
        0x80057FFE);
  CHECK(syclcompat::extend_vsub2<int32_t>((uint32_t)0x7FFFFFFD,
                                          (uint32_t)0xFFFA7FFF, 0),
        0x80057FFE);
  CHECK(syclcompat::extend_vsub2<int32_t>((uint32_t)0x7FFFFFFD,
                                          (int32_t)0xFFFA7FFF, 0),
        0x80057FFE);
  CHECK(syclcompat::extend_vsub2<int32_t>((int32_t)0x7FFFFFFD,
                                          (uint32_t)0xFFFA7FFF, 0),
        0x80057FFE);
  CHECK(syclcompat::extend_vsub2_sat<int32_t>((int32_t)0x7FFFFFFD,
                                              (int32_t)0xFFFA7FFF, 0),
        0x7FFF8000);
  CHECK(syclcompat::extend_vsub2_sat<int32_t>((uint32_t)0x7FFFFFFD,
                                              (uint32_t)0xFFFA7FFF, 0),
        0x80057FFE);
  CHECK(syclcompat::extend_vsub2_sat<int32_t>((int32_t)0x7FFFFFFD,
                                              (uint32_t)0xFFFA7FFF, 0),
        0x80058000);
  CHECK(syclcompat::extend_vsub2_sat<int32_t>((uint32_t)0x7FFFFFFD,
                                              (int32_t)0xFFFA7FFF, 0),
        0x7FFF7FFE);

  CHECK(syclcompat::extend_vsub2<uint32_t>(0x0002000B, 0x0001000A, 0),
        0x00010001);
  CHECK(syclcompat::extend_vsub2<uint32_t>((uint32_t)0x00010001,
                                           (uint32_t)0x0002FFFF, 0),
        0xFFFF0002);
  CHECK(syclcompat::extend_vsub2<uint32_t>((int32_t)0x00010001,
                                           (int32_t)0x0002FFFF, 0),
        0xFFFF0002);
  CHECK(syclcompat::extend_vsub2_sat<uint32_t>((uint32_t)0x00010001,
                                               (uint32_t)0x0002FFFF, 0),
        0x00000000);
  CHECK(syclcompat::extend_vsub2_sat<uint32_t>((int32_t)0x00010001,
                                               (int32_t)0x0002FFFF, 0),
        0x00000002);

  return {nullptr, 0};
}

std::pair<const char *, int> vadd2_add() {

  CHECK(syclcompat::extend_vadd2_add<int32_t>(0x00010002, 0x00030004, 1),
        0x0000000B);
  CHECK(syclcompat::extend_vadd2_add<int32_t>(0x0001FFFF, 0x0002FFFE, -1),
        0xFFFFFFFF);
  CHECK(syclcompat::extend_vadd2_add<int32_t>(0x00017FFF, 0x00017FFF, 1),
        0x00010001);

  CHECK(syclcompat::extend_vadd2_add<uint32_t>(0x00010002, 0x00030004, 1),
        0x0000000B);
  CHECK(syclcompat::extend_vadd2_add<uint32_t>((uint32_t)0x0001FFFF,
                                               (uint32_t)0x0002FFFF, 1),
        0x00020002);
  CHECK(syclcompat::extend_vadd2_add<uint32_t>(0x0001FFFF, 0x0002FFFF, 1),
        0x00000002);

  return {nullptr, 0};
}

std::pair<const char *, int> vsub2_add() {

  // Testing API with mixed types
  CHECK(syclcompat::extend_vsub2_add<int32_t>((int32_t)0x0001FFFF,
                                              (int32_t)0xFFFF0001, 1),
        1);
  CHECK(syclcompat::extend_vsub2_add<int32_t>((uint32_t)0x7FFFFFFD,
                                              (uint32_t)0xFFFA7FFF, -1),
        0x00000002);
  CHECK(syclcompat::extend_vsub2_add<int32_t>((int32_t)0x7FFFFFFD,
                                              (int32_t)0xFFFA7FFF, -1),
        0x00000002);
  CHECK(syclcompat::extend_vsub2_add<int32_t>((int32_t)0x7FFFFFFD,
                                              (uint32_t)0xFFFA7FFF, -1),
        0xFFFF0002);
  CHECK(syclcompat::extend_vsub2_add<int32_t>((uint32_t)0x7FFFFFFD,
                                              (int32_t)0xFFFA7FFF, -1),
        0x00010002);

  CHECK(syclcompat::extend_vsub2_add<uint32_t>(0x0002000B, 0x0001000A, 1),
        0x00000003);
  CHECK(syclcompat::extend_vsub2_add<uint32_t>(0x00010001, 0x0002FFFF, 3),
        0x00000004);

  return {nullptr, 0};
}

std::pair<const char *, int> vabsdiff2() {

  CHECK(syclcompat::extend_vabsdiff2<int32_t>((int32_t)0xFFFF0001,
                                              (int32_t)0x0003FFFF, 0),
        0x00040002);
  CHECK(syclcompat::extend_vabsdiff2<int32_t>((int32_t)0x80000002,
                                              (int32_t)0x00010001, 0),
        0x80010001);
  CHECK(syclcompat::extend_vabsdiff2_sat<int32_t>((int32_t)0x80000002,
                                                  (int32_t)0x00010001, 0),
        0x7FFF0001);

  CHECK(syclcompat::extend_vabsdiff2<uint32_t>(0x00010004, 0x00030002, 0),
        0x00020002);
  CHECK(syclcompat::extend_vabsdiff2<uint32_t>((uint32_t)0xFFFF0001,
                                               (int32_t)0xFFFE0003, 0),
        0x00010002);
  CHECK(syclcompat::extend_vabsdiff2_sat<uint32_t>((uint32_t)0xFFFF0001,
                                                   (int32_t)0xFFFE0003, 0),
        0xFFFF0002);

  return {nullptr, 0};
}

std::pair<const char *, int> vabsdiff2_add() {

  CHECK(syclcompat::extend_vabsdiff2_add<int32_t>((int32_t)0xFFFF0001,
                                                  (int32_t)0x0003FFFF, -2),
        0x00000004);

  CHECK(syclcompat::extend_vabsdiff2_add<uint32_t>(0x000A000C, 0x000B000A, 1),
        0x00000004);

  return {nullptr, 0};
}

std::pair<const char *, int> vmin2() {

  CHECK(syclcompat::extend_vmin2<int32_t>((int32_t)0xFFFF0002, 0x00010001, 0),
        (int32_t)0xFFFF0001);
  CHECK(syclcompat::extend_vmin2_sat<int32_t>(0x0002FFF1, 0x0001FFF2, 0),
        0x0001FFF1);

  CHECK(syclcompat::extend_vmin2<uint32_t>(0x000A000D, 0x000B000C, 0),
        0x000A000C);
  CHECK(syclcompat::extend_vmin2_sat<uint32_t>(0x0002FFF1, 0x0001FFF2, 0),
        0x00010000);

  return {nullptr, 0};
}

std::pair<const char *, int> vmax2() {

  CHECK(syclcompat::extend_vmax2<int32_t>((int32_t)0xFFFF0002, 0x00010001, 0),
        0x00010002);
  CHECK(syclcompat::extend_vmax2_sat<int32_t>(0x80008000, 0x00018001, 0),
        0x7FFF7FFF);

  CHECK(syclcompat::extend_vmax2<uint32_t>(0x000A000D, 0x000B000C, 0),
        0x000B000D);
  CHECK(syclcompat::extend_vmax2_sat<uint32_t>(0x0002FFF1, 0x0001FFF2, 0),
        0x00020000);

  return {nullptr, 0};
}

std::pair<const char *, int> vmin2_vmax2_add() {

  CHECK(
      syclcompat::extend_vmin2_add<int32_t>((int32_t)0xFFFF0002, 0x00010001, 2),
      0x00000002);
  CHECK(syclcompat::extend_vmin2_add<uint32_t>(0x000A000D, 0x000B000C, 2),
        0x00000018);

  CHECK(syclcompat::extend_vmax2_add<int32_t>((int32_t)0xFFFF0002, 0x00010001,
                                              -2),
        0x00000001);
  CHECK(syclcompat::extend_vmax2_add<uint32_t>(0x000A000D, 0x000B000C, 2),
        0x0000001A);

  return {nullptr, 0};
}

std::pair<const char *, int> vavrg2() {

  CHECK(syclcompat::extend_vavrg2<int32_t>((int32_t)0xFFFFFFF6, 0x0005FFFA, 0),
        0x0002FFF8);
  CHECK(syclcompat::extend_vavrg2_sat<int32_t>((int32_t)0xFFFFFFF6, 0x0005FFFA,
                                               0),
        0x0002FFF8);

  CHECK(syclcompat::extend_vavrg2<uint32_t>(0x00010006, 0x00030001, 0),
        0x00020004);
  CHECK(syclcompat::extend_vavrg2_sat<uint32_t>(0x00010006, 0x00030001, 0),
        0x00020004);

  return {nullptr, 0};
}

std::pair<const char *, int> vavrg2_add() {

  CHECK(syclcompat::extend_vavrg2_add<int32_t>((int32_t)0xFFFFFFF6, 0x0005FFFA,
                                               -2),
        0xFFFFFFF8);

  CHECK(syclcompat::extend_vavrg2_add<uint32_t>(0x00010006, 0x00030002, 2),
        0x00000008);

  return {nullptr, 0};
}

std::pair<const char *, int> vcompare2() {

  CHECK(syclcompat::extend_vcompare2(0x0002FFFF, 0x0001FFFF, std::greater<>()),
        (unsigned)0x00010000);
  CHECK(syclcompat::extend_vcompare2((uint32_t)0x0002FFFF, (int32_t)0x0001FFFF,
                                     std::greater<>()),
        (unsigned)0x00010001);
  CHECK(syclcompat::extend_vcompare2((int32_t)0x0002FFFF, (uint32_t)0x0001FFFF,
                                     std::greater<>()),
        (unsigned)0x00010000);

  CHECK(syclcompat::extend_vcompare2(0x0002FFFF, 0x0001FFFF, std::less<>()),
        (unsigned)0x00000000);
  CHECK(syclcompat::extend_vcompare2(0x0002FFFF, 0x0002FFFF,
                                     std::greater_equal<>()),
        (unsigned)0x00010001);
  CHECK(
      syclcompat::extend_vcompare2(0x0002FFFF, 0x0001FFFF, std::less_equal<>()),
      (unsigned)0x00000001);
  CHECK(syclcompat::extend_vcompare2(0xFFFE0002, 0xFFFF0002, std::equal_to<>()),
        (unsigned)0x00000001);
  CHECK(syclcompat::extend_vcompare2(0xFFFE0002, 0xFFFF0002,
                                     std::not_equal_to<>()),
        (unsigned)0x00010000);

  return {nullptr, 0};
}

std::pair<const char *, int> vcompare2_add() {

  CHECK(syclcompat::extend_vcompare2_add(0x0002FFFF, 0x0001FFFF, 1,
                                         std::greater<>()),
        (unsigned)0x00000002);
  CHECK(syclcompat::extend_vcompare2_add(0x0002FFFF, 0x0001FFFF, 2,
                                         std::less<>()),
        (unsigned)0x00000002);
  CHECK(syclcompat::extend_vcompare2_add(0x0002FFFF, 0x0002FFFF, 1,
                                         std::greater_equal<>()),
        (unsigned)0x00000003);
  CHECK(syclcompat::extend_vcompare2_add(0x0002FFFF, 0x0001FFFF, 2,
                                         std::less_equal<>()),
        (unsigned)0x00000003);
  CHECK(syclcompat::extend_vcompare2_add(0xFFFE0002, 0xFFFF0002, 0xFFFF,
                                         std::equal_to<>()),
        (unsigned)0x00010000);
  CHECK(syclcompat::extend_vcompare2_add(0xFFFE0002, 0xFFFF0002, 0xFF,
                                         std::not_equal_to<>()),
        (unsigned)0x00000100);

  return {nullptr, 0};
}

void test(const sycl::stream &s, int *ec) {
  {
    auto res = vadd2();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 1;
      return;
    }
    s << "vadd2 check passed!\n";
  }
  {
    auto res = vsub2();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 2;
      return;
    }
    s << "vsub2 check passed!\n";
  }
  {
    auto res = vadd2_add();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 3;
      return;
    }
    s << "vadd2_add check passed!\n";
  }
  {
    auto res = vsub2_add();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 4;
      return;
    }
    s << "vsub2_add check passed!\n";
  }
  {
    auto res = vabsdiff2();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 5;
      return;
    }
    s << "vabsdiff2 check passed!\n";
  }
  {
    auto res = vmin2();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 6;
      return;
    }
    s << "vmin2 check passed!\n";
  }
  {
    auto res = vmax2();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 7;
      return;
    }
    s << "vmax2 check passed!\n";
  }
  {
    auto res = vmin2_vmax2_add();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 8;
      return;
    }
    s << "vmin2_add/vmax2_add check passed!\n";
  }
  {
    auto res = vavrg2();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 9;
      return;
    }
    s << "vavrg2 check passed!\n";
  }
  {
    auto res = vavrg2_add();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 10;
      return;
    }
    s << "vavrg2_add check passed!\n";
  }
  {
    auto res = vabsdiff2_add();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 11;
      return;
    }
    s << "vabsdiff2_add check passed!\n";
  }
  {
    auto res = vcompare2();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 12;
      return;
    }
    s << "vcompare2 check passed!\n";
  }
  {
    auto res = vcompare2_add();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 13;
      return;
    }
    s << "vcompare2_add check passed!\n";
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
