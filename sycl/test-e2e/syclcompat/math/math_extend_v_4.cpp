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
 *  math_extend_v_4.cpp
 *
 *  Description:
 *    math extend 4-vectorized helpers tests
 **************************************************************************/

// ===------------- math_extend_vfunc_4.cpp ----------------*- C++ -*-----===//
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

int vadd4() {
  int errc{};
  int test_id{};
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

  return errc;
}

int vadd4_add() {
  int errc{};
  int test_id{};
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

  return errc;
}

int vsub4() {
  int errc{};
  int test_id{};
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

  return errc;
}

int vsub4_add() {
  int errc{};
  int test_id{};
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

  return errc;
}

int vabsdiff4() {
  int errc{};
  int test_id{};
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

  return errc;
}

int vabsdiff4_add() {
  int errc{};
  int test_id{};
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

  return errc;
}

int vmin4() {
  int errc{};
  int test_id{};
  CHECK(syclcompat::extend_vmin4<int32_t>((int32_t)0xFFFF0102,
                                          (int32_t)0xFE010201, 0),
        0xFEFF0101);

  CHECK(syclcompat::extend_vmin4_sat<int32_t>(0x0102FF00, 0x0201FE00, 0),
        0x0101FE00);

  CHECK(syclcompat::extend_vmin4<uint32_t>(0x010A020D, 0x000B020C, 0),
        0x000A020C);

  CHECK(syclcompat::extend_vmin4_sat<uint32_t>(0x020201FF, 0x0201FFFE, 0),
        0x02010000);

  return errc;
}

int vmax4() {
  int errc{};
  int test_id{};
  CHECK(syclcompat::extend_vmax4<int32_t>((int32_t)0xFFFF0102,
                                          (int32_t)0xFE010201, 0),
        0xFF010202);
  CHECK(syclcompat::extend_vmax4_sat<int32_t>(0x0102FF00, 0x0201FE00, 0),
        0x0202FF00);

  CHECK(syclcompat::extend_vmax4<uint32_t>(0x010A020D, 0x000B020C, 0),
        0x010B020D);
  CHECK(syclcompat::extend_vmax4_sat<uint32_t>(0x020201FF, 0x0201FFFE, 0),
        0x02020100);

  return errc;
}

int vmin4_vmax4_add() {
  int errc{};
  int test_id{};
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

  return errc;
}

int vavrg4() {
  int errc{};
  int test_id{};
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

  return errc;
}

int vavrg4_add() {
  int errc{};
  int test_id{};
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

  return errc;
}

int vcompare4() {
  int errc{};
  int test_id{};
  CHECK(syclcompat::extend_vcompare4(0x0102FEFF, 0x01FFFFFE, std::greater<>()),
        (unsigned)0x00010001);
  CHECK(syclcompat::extend_vcompare4((uint32_t)0x0102FEFF, (int32_t)0x01FFFFFE,
                                     std::greater<>()),
        (unsigned)0x00010101);
  CHECK(syclcompat::extend_vcompare4((int32_t)0x0102FEFF, (uint32_t)0x01FFFFFE,
                                     std::greater<>()),
        (unsigned)0x00000000);

  CHECK(syclcompat::extend_vcompare4(0x0102FEFF, 0x01FFFFFE, std::less<>()),
        (unsigned)0x00000100);
  CHECK(syclcompat::extend_vcompare4(0x0102FEFF, 0x01FFFFFE,
                                     std::greater_equal<>()),
        (unsigned)0x01010001);
  CHECK(
      syclcompat::extend_vcompare4(0x0102FEFF, 0x01FFFFFE, std::less_equal<>()),
      (unsigned)0x01000100);
  CHECK(syclcompat::extend_vcompare4(0xFFFE0102, 0xFFFF0202, std::equal_to<>()),
        (unsigned)0x01000001);
  CHECK(syclcompat::extend_vcompare4(0xFFFE0102, 0xFFFF0202,
                                     std::not_equal_to<>()),
        (unsigned)0x00010100);

  return errc;
}

int vcompare4_add() {
  int errc{};
  int test_id{};
  CHECK(syclcompat::extend_vcompare4_add(0x0102FEFF, 0x01FFFFFE, 1,
                                         std::greater<>()),
        (unsigned)0x00000003);
  CHECK(syclcompat::extend_vcompare4_add(0x0102FEFF, 0x01FFFFFE, 1,
                                         std::less<>()),
        (unsigned)0x00000002);
  CHECK(syclcompat::extend_vcompare4_add(0x0102FEFF, 0x01FFFFFE, 2,
                                         std::greater_equal<>()),
        (unsigned)0x00000005);
  CHECK(syclcompat::extend_vcompare4_add(0x0102FEFF, 0x01FFFFFE, 2,
                                         std::less_equal<>()),
        (unsigned)0x00000004);
  CHECK(syclcompat::extend_vcompare4_add(0xFFFE0102, 0xFFFF0202, 0xFF,
                                         std::equal_to<>()),
        (unsigned)0x00000101);
  CHECK(syclcompat::extend_vcompare4_add(0xFFFE0102, 0xFFFF0202, 0xFFFF,
                                         std::not_equal_to<>()),
        (unsigned)0x00010001);

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
  q.wait_and_throw();

  int ec_h{};
  syclcompat::memcpy<int>(&ec_h, ec, 1, q);
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

  test_fn<vadd4>(q, ec);
  test_fn<vsub4>(q, ec);
  test_fn<vadd4_add>(q, ec);
  test_fn<vsub4_add>(q, ec);
  test_fn<vabsdiff4>(q, ec);
  test_fn<vabsdiff4_add>(q, ec);
  test_fn<vmin4>(q, ec);
  test_fn<vmax4>(q, ec);
  test_fn<vmin4_vmax4_add>(q, ec);
  test_fn<vavrg4>(q, ec);
  test_fn<vavrg4_add>(q, ec);
  test_fn<vcompare4>(q, ec);
  test_fn<vcompare4_add>(q, ec);

  syclcompat::free(ec, q);
}
