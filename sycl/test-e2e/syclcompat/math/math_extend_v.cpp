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
 *  math_extend_v.cpp
 *
 *  Description:
 *    math extend-vectorized helpers tests
 **************************************************************************/

// ===------------- math_extend_vfunc[2/4].cpp --------------*- C++ -*-----===//
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

// v4
std::pair<const char *, int> vadd4() {
  CHECK(syclcompat::extend_vadd4<int32_t>(0x0102FFFE, 0x01FF02FF, 0),
        0x020101FD);
  CHECK(syclcompat::extend_vadd4<int32_t>((int32_t)0x7E81FEFF,
                                          (int32_t)0x02FD03FF, 0),
        0x807E01FE);
  CHECK(syclcompat::extend_vadd4<int32_t>((uint32_t)0x7E81FEFF,
                                          (uint32_t)0x02FD03FF, 0),
        0x807E01FE);
  CHECK(syclcompat::extend_vadd4<int32_t>((uint32_t)0x7E81FEFF,
                                          (int32_t)0x02FD03FF, 0),
        0x807E01FE);
  CHECK(syclcompat::extend_vadd4<int32_t>((int32_t)0x7E81FEFF,
                                          (uint32_t)0x02FD03FF, 0),
        0x807E01FE);
  CHECK(syclcompat::extend_vadd4_sat<int32_t>((int32_t)0x7E81FEFF,
                                              (int32_t)0x02FD03FF, 0),
        0x7F8001FE);
  CHECK(syclcompat::extend_vadd4_sat<int32_t>((uint32_t)0x7E81FEFF,
                                              (uint32_t)0x02FD03FF, 0),
        0x7F7F7F7F);
  CHECK(syclcompat::extend_vadd4_sat<int32_t>((uint32_t)0x7E81FEFF,
                                              (int32_t)0x02FD03FF, 0),
        0x7F7E7F7F);
  CHECK(syclcompat::extend_vadd4_sat<int32_t>((int32_t)0x7E81FEFF,
                                              (uint32_t)0x02FD03FF, 0),
        0x7F7E017F);

  CHECK(syclcompat::extend_vadd4<uint32_t>(0x01020304, 0x0A0B0C0D, 0),
        0x0B0D0F11);
  CHECK(syclcompat::extend_vadd4<uint32_t>((uint32_t)0x000100FF,
                                           (uint32_t)0x00FE0001, 0),
        0x00FF0000);
  CHECK(syclcompat::extend_vadd4_sat<uint32_t>((uint32_t)0x000100FF,
                                               (uint32_t)0x00FE0001, 0),
        0x00FF00FF);

  return {nullptr, 0};
}

std::pair<const char *, int> vadd4_add() {

  CHECK(syclcompat::extend_vadd4_add<int32_t>(0x0102FFFE, 0x01FF02FF, 1),
        0x00000002);
  CHECK(syclcompat::extend_vadd4_add<int32_t>((int32_t)0x7E81FEFF,
                                              (int32_t)0x02FD03FF, -1),
        0xFFFFFFFC);
  CHECK(syclcompat::extend_vadd4_add<int32_t>((uint32_t)0x7E81FEFF,
                                              (uint32_t)0x02FD03FF, -1),
        0x000004FC);
  CHECK(syclcompat::extend_vadd4_add<int32_t>((uint32_t)0x7E81FEFF,
                                              (int32_t)0x02FD03FF, -1),
        0x000002FC);
  CHECK(syclcompat::extend_vadd4_add<int32_t>((int32_t)0x7E81FEFF,
                                              (uint32_t)0x02FD03FF, -1),
        0x000001FC);

  CHECK(syclcompat::extend_vadd4_add<uint32_t>(0x01020304, 0x01000100, 1),
        0x0000000D);
  CHECK(syclcompat::extend_vadd4_add<uint32_t>((uint32_t)0x000100FF,
                                               (uint32_t)0x00FE0001, 1),
        0x0000000200);

  return {nullptr, 0};
}

std::pair<const char *, int> vsub4() {

  CHECK(syclcompat::extend_vsub4<int32_t>((int32_t)0x0102FFFF,
                                          (int32_t)0x020101FE, 0),
        0xFF01FE01);
  CHECK(syclcompat::extend_vsub4<int32_t>((int32_t)0x01807F10, 0x0102FE10, 0),
        0x007E8100);
  CHECK(
      syclcompat::extend_vsub4_sat<int32_t>((int32_t)0x01807F10, 0x0102FE10, 0),
      0x00807F00);

  CHECK(syclcompat::extend_vsub4<uint32_t>(0x02020C0B, 0x02010A0A, 0),
        0x00010201);
  CHECK(syclcompat::extend_vsub4<uint32_t>(0x01020304, 0x02040608, 0),
        0xFFFEFDFC);
  CHECK(syclcompat::extend_vsub4_sat<uint32_t>(0x01020304, 0x02040608, 0),
        0x00000000);

  return {nullptr, 0};
}

std::pair<const char *, int> vsub4_add() {

  CHECK(syclcompat::extend_vsub4_add<int32_t>((int32_t)0x0102FFFF,
                                              (int32_t)0x020101FE, -1),
        0xFFFFFFFE);
  CHECK(
      syclcompat::extend_vsub4_add<int32_t>((int32_t)0x01807F10, 0x0102FE10, 2),
      0x00000001);

  CHECK(syclcompat::extend_vsub4_add<uint32_t>(0x02020C0B, 0x02010A0A, 2),
        0x00000006);
  CHECK(syclcompat::extend_vsub4_add<uint32_t>(0x01020304, 0x02040608, 1),
        0xFFFFFFF7);

  CHECK(syclcompat::extend_vsub4_add<uint32_t>((uint32_t)0x01020304,
                                               (uint32_t)0x02040608, 1),
        0xFFFFFFF7);

  return {nullptr, 0};
}

std::pair<const char *, int> vabsdiff4() {

  CHECK(
      syclcompat::extend_vabsdiff4<int32_t>((int32_t)0xFF01FF02, 0x01FF02FF, 0),
      0x02020303);
  CHECK(syclcompat::extend_vabsdiff4<int32_t>((int32_t)0x8002007F,
                                              (int32_t)0x01010080, 0),
        0x810100FF);
  CHECK(syclcompat::extend_vabsdiff4_sat<int32_t>((int32_t)0x8002007F,
                                                  (int32_t)0x01010080, 0),
        0x7F01007F);

  CHECK(syclcompat::extend_vabsdiff4<uint32_t>(0x01020304, 0x04030201, 0),
        0x03010103);
  CHECK(syclcompat::extend_vabsdiff4<uint32_t>((uint32_t)0xFEFF0001,
                                               (int32_t)0xF0FE0003, 0),
        0x0E010002);
  CHECK(syclcompat::extend_vabsdiff4_sat<uint32_t>((uint32_t)0xFEFF0001,
                                                   (int32_t)0xF0FE0003, 0),
        0xFFFF0002);

  return {nullptr, 0};
}

std::pair<const char *, int> vabsdiff4_add() {

  CHECK(syclcompat::extend_vabsdiff4_add<int32_t>((int32_t)0xFF01FF02,
                                                  0x01FF02FF, 1),
        0x0000000B);
  CHECK(syclcompat::extend_vabsdiff4_add<int32_t>((int32_t)0x8002007F,
                                                  (int32_t)0x01010080, -1),
        0x00000180);

  CHECK(syclcompat::extend_vabsdiff4_add<uint32_t>(0x01020304, 0x04030201, 2),
        0x0000000A);
  CHECK(syclcompat::extend_vabsdiff4_add<uint32_t>((uint32_t)0xFEFF0001,
                                                   (int32_t)0xF0FE0003, 1),
        0x00000212);

  return {nullptr, 0};
}

std::pair<const char *, int> vmin4() {

  CHECK(syclcompat::extend_vmin4<int32_t>((int32_t)0xFFFF0102,
                                          (int32_t)0xFE010201, 0),
        0xFEFF0101);

  CHECK(syclcompat::extend_vmin4_sat<int32_t>(0x0102FF00, 0x0201FE00, 0),
        0x0101FE00);

  CHECK(syclcompat::extend_vmin4<uint32_t>(0x010A020D, 0x000B020C, 0),
        0x000A020C);

  CHECK(syclcompat::extend_vmin4_sat<uint32_t>(0x020201FF, 0x0201FFFE, 0),
        0x02010000);

  return {nullptr, 0};
}

std::pair<const char *, int> vmax4() {

  CHECK(syclcompat::extend_vmax4<int32_t>((int32_t)0xFFFF0102,
                                          (int32_t)0xFE010201, 0),
        0xFF010202);
  CHECK(syclcompat::extend_vmax4_sat<int32_t>(0x0102FF00, 0x0201FE00, 0),
        0x0202FF00);

  CHECK(syclcompat::extend_vmax4<uint32_t>(0x010A020D, 0x000B020C, 0),
        0x010B020D);
  CHECK(syclcompat::extend_vmax4_sat<uint32_t>(0x020201FF, 0x0201FFFE, 0),
        0x02020100);

  return {nullptr, 0};
}

std::pair<const char *, int> vmin4_vmax4_add() {

  CHECK(syclcompat::extend_vmin4_add<int32_t>((int32_t)0xFFFF0102,
                                              (int32_t)0xFE010201, -1),
        0xFFFFFFFE);

  CHECK(syclcompat::extend_vmin4_add<uint32_t>(0x010A020D, 0x000B020C, 1),
        0x00000019);

  CHECK(syclcompat::extend_vmax4_add<int32_t>((int32_t)0xFFFF0102,
                                              (int32_t)0xFE010201, 2),
        0x00000006);
  CHECK(syclcompat::extend_vmax4_add<uint32_t>(0x010A020D, 0x000B020C, -1),
        0x0000001A);

  return {nullptr, 0};
}

std::pair<const char *, int> vavrg4() {

  CHECK(syclcompat::extend_vavrg4<int32_t>((int32_t)0xFF01FF01, 0x0505FF00, 0),
        0x0203FF01);
  CHECK(syclcompat::extend_vavrg4_sat<int32_t>((int32_t)0xFF01FF01, 0x0505FF00,
                                               0),
        0x0203FF01);

  CHECK(syclcompat::extend_vavrg4<uint32_t>(0x00010106, (int32_t)0xFC050101, 0),
        (int32_t)0xFE030104);
  CHECK(syclcompat::extend_vavrg4_sat<uint32_t>(0x00010106, (int32_t)0xFC050101,
                                                0),
        (int32_t)0x00030104);

  return {nullptr, 0};
}

std::pair<const char *, int> vavrg4_add() {

  CHECK(syclcompat::extend_vavrg4_add<int32_t>((int32_t)0xFF01FF01, 0x0505FF00,
                                               1),
        0x00000006);
  CHECK(syclcompat::extend_vavrg4_add<int32_t>((int32_t)0xFF01FF01, 0x0505FF00,
                                               -6),
        0xFFFFFFFF);

  CHECK(syclcompat::extend_vavrg4_add<uint32_t>(0x00010106, (int32_t)0xFC050101,
                                                1),
        (int32_t)0x00000007);

  CHECK(syclcompat::extend_vavrg4_add<uint32_t>(0x00010106, (int32_t)0xFC050101,
                                                -1),
        (int32_t)0x00000005);

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
    auto res = vadd4();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 12;
      return;
    }
    s << "vadd4 check passed!\n";
  }
  {
    auto res = vsub4();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 13;
      return;
    }
    s << "vsub4 check passed!\n";
  }
  {
    auto res = vadd4_add();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 14;
      return;
    }
    s << "vadd4_add check passed!\n";
  }
  {
    auto res = vsub4_add();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 15;
      return;
    }
    s << "vsub4_add check passed!\n";
  }
  {
    auto res = vabsdiff4();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 16;
      return;
    }
    s << "vabsdiff4 check passed!\n";
  }
  {
    auto res = vabsdiff4_add();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 17;
      return;
    }
    s << "vabsdiff4_add check passed!\n";
  }
  {
    auto res = vmin4();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 18;
      return;
    }
    s << "vmin4 check passed!\n";
  }
  {
    auto res = vmax4();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 19;
      return;
    }
    s << "vmax4 check passed!\n";
  }
  {
    auto res = vmin4_vmax4_add();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 20;
      return;
    }
    s << "vmin4_add/vmax4_add check passed!\n";
  }
  {
    auto res = vavrg4();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 21;
      return;
    }
    s << "vavrg4 check passed!\n";
  }
  {
    auto res = vavrg4_add();
    if (res.first) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 22;
      return;
    }
    s << "vavrg4_add check passed!\n";
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
