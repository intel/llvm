// RUN: %clangxx -fsycl -std=c++17 -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==---------- vector_byte.cpp - SYCL vec<> for std::byte test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL_SIMPLE_SWIZZLES
#include <sycl/sycl.hpp>

int main() {
  std::byte bt{7};
  // constructors
  sycl::vec<std::byte, 1> vb1(bt);
  sycl::vec<std::byte, 2> vb2{bt, bt};
  sycl::vec<std::byte, 3> vb3{bt, bt, bt};
  sycl::vec<std::byte, 4> vb4{bt, bt, bt, bt};
  sycl::vec<std::byte, 8> vb8{bt, bt, bt, bt, bt, bt, bt, bt};
  sycl::vec<std::byte, 16> vb16{bt, bt, bt, std::byte{2}, bt, bt, bt, bt,
                                bt, bt, bt, bt,           bt, bt, bt, bt};

  {
    // operator[]
    assert(vb16[3] == std::byte{2});
    // explicit conversion
    std::byte(vb1.x());
    std::byte b = vb1;

    // operator=
    auto vb4op = vb4;
    vb1 = std::byte{3};
  }

  // convert() and as()
  {
    sycl::vec<int, 2> vi2(1, 1);
    auto cnv = vi2.convert<std::byte>();
    auto cnv2 = vb1.convert<int>();

    auto asint = vb2.template as<sycl::vec<int16_t, 1>>();
    auto asbyte = vi2.template as<sycl::vec<std::byte, 8>>();
  }

  // load() and store()
  std::vector<std::byte> std_vec(8, bt);
  {
    sycl::buffer<std::byte, 1> Buf(std_vec.data(), sycl::range<1>(8));

    sycl::queue Queue;
    Queue
        .submit([&](sycl::handler &cgh) {
          auto Acc = Buf.get_access<sycl::access::mode::read_write>(cgh);
          cgh.single_task<class st>([=]() {
            // load
            sycl::multi_ptr<std::byte,
                            sycl::access::address_space::global_space>
                mp(&Acc[0]);
            sycl::vec<std::byte, 8> sycl_vec;
            sycl_vec.load(0, mp);
            sycl_vec[0] = std::byte{2};
            Acc[1] = std::byte{10};

            // store
            sycl_vec.store(0, mp);
          });
        })
        .wait();
  }
  assert(std_vec[0] == std::byte{2});
  assert(std_vec[1] == std::byte{7});

  // swizzle
  {
    auto swizzled_vec = vb8.lo();

    auto sw = vb8.template swizzle<sycl::elem::s0, sycl::elem::s1>()
                  .template as<sycl::vec<int16_t, 1>>();
    auto swbyte = sw.template as<sycl::vec<std::byte, 2>>();
    auto swbyte2 = swizzled_vec.template as<sycl::vec<int, 1>>();

    // hi/lo, even/odd
    sycl::vec<std::byte, 4> vbsw(std::byte{0}, std::byte{1}, std::byte{2},
                                 std::byte{3});
    sycl::vec<std::byte, 2> vbswhi = vbsw.hi();
    assert(vbswhi[0] == std::byte{2});
    vbswhi = vbsw.lo();
    vbswhi = vbsw.odd();
    vbswhi = vbsw.even();
  }

  // operatorOP for vec and for swizzle
  {
    sycl::vec<std::byte, 3> vop1{std::byte{4}, std::byte{9}, std::byte{25}};
    sycl::vec<std::byte, 3> vop2{std::byte{2}, std::byte{3}, std::byte{5}};
    sycl::vec<std::byte, 4> vop3{std::byte{5}, std::byte{6}, std::byte{2},
                                 std::byte{3}};

    // binary op for 2 vec
    auto vop = vop1 + vop2;
    assert(vop[0] == std::byte{6});
    vop = vop1 - vop2;
    vop = vop1 * vop2;
    vop = vop1 / vop2;
    assert(vop[0] == std::byte{2});
    vop = vop1 % vop2;

    // binary op for 2 swizzle
    auto swlo = vop3.lo();
    auto swhi = vop3.hi();
    auto swplus = swlo + swhi;
    sycl::vec<std::byte, 2> vec_test = swplus;
    assert(vec_test.x() == std::byte{7} && vec_test.y() == std::byte{9});
    auto swominus = swlo - swhi;
    auto swmul = swlo * swhi;
    vec_test = swmul;
    assert(vec_test.x() == std::byte{10} && vec_test.y() == std::byte{18});
    auto swdiv = swlo / swhi;

    // binary op for 1 vec
    vop = vop1 + std::byte{3};
    vop = vop1 - std::byte{3};
    assert(vop[1] == std::byte{6});
    vop = vop1 * std::byte{3};
    vop = vop1 / std::byte{3};
    vop = vop1 % std::byte{3};
    assert(vop[0] == std::byte{1});

    vop = std::byte{3} + vop1;
    assert(vop[0] == std::byte{7});
    vop = std::byte{3} - vop1;
    vop = std::byte{3} * vop1;
    assert(vop[2] == std::byte{75});
    vop = std::byte{3} / vop1;

    // binary op for 1 swizzle
    auto swplus1 = swlo + std::byte{3};
    auto swminus1 = swlo - std::byte{3};
    vec_test = swminus1;
    assert(vec_test.x() == std::byte{2} && vec_test.y() == std::byte{3});
    auto swmul1 = swlo * std::byte{3};
    auto swdiv1 = swlo / std::byte{3};
    vec_test = swdiv1;
    assert(vec_test.x() == std::byte{1} && vec_test.y() == std::byte{2});

    auto swplus2 = std::byte{3} + swlo;
    vec_test = swplus2;
    assert(vec_test.x() == std::byte{8} && vec_test.y() == std::byte{9});
    auto swminus2 = std::byte{3} - swlo;
    auto swmul2 = std::byte{3} * swlo;
    vec_test = swmul2;
    assert(vec_test.x() == std::byte{15} && vec_test.y() == std::byte{18});
    auto swdiv2 = std::byte{3} / swlo;

    // operatorOP= for 2 vec
    sycl::vec<std::byte, 3> vbuf{std::byte{4}, std::byte{5}, std::byte{6}};
    vop = vbuf += vop1;
    assert(vop[0] == std::byte{8});
    vop = vbuf -= vop1;
    vop = vbuf *= vop1;
    vop = vbuf /= vop1;
    vop = vbuf %= vop1;

    // operatorOP= for 2 swizzle
    swlo += swhi;
    swlo -= swhi;
    vec_test = swlo;
    assert(vec_test.x() == std::byte{5} && vec_test.y() == std::byte{6});
    swlo *= swhi;
    swlo /= swhi;
    swlo %= swhi;

    // operatorOP= for 1 vec
    vop = vop1 += std::byte{3};
    assert(vop[0] == std::byte{7});
    vop = vop1 -= std::byte{3};
    vop = vop1 *= std::byte{3};
    vop = vop1 /= std::byte{3};
    vop = vop1 %= std::byte{3};

    // operatorOP= for 1 swizzle

    swlo += std::byte{3};
    swlo -= std::byte{1};
    vec_test = swlo;
    assert(vec_test.x() == std::byte{3} && vec_test.y() == std::byte{2});
    swlo *= std::byte{3};
    swlo /= std::byte{3};
    swlo %= std::byte{3};

    // unary operator++ and -- for vec
    vop1 = sycl::vec<std::byte, 3>(std::byte{4}, std::byte{9}, std::byte{25});
    vop1++;
    vop1--;
    vop = ++vop1;
    assert(vop[2] == std::byte{26});
    --vop1;

    // unary operator++ and -- for swizzle
    swlo++;
    swlo--;
    vec_test = swlo;
    assert(vec_test.x() == std::byte{0} && vec_test.y() == std::byte{2});

    // logical binary op for 2 vec
    vop = vop1 & vop2;
    vop = vop1 | vop2;
    vop = vop1 ^ vop2;

    // logical binary op for 2 swizzle
    auto swand = swlo & swhi;
    auto swor = swlo | swhi;
    auto swxor = swlo ^ swhi;

    // logical binary op for 1 vec
    vop = vop1 & std::byte{3};
    vop = vop1 | std::byte{3};
    vop = vop1 ^ std::byte{3};
    vop = std::byte{3} & vop1;
    vop = std::byte{3} | vop1;
    vop = std::byte{3} ^ vop1;

    // logical binary op for 1 swizzle
    auto swand2 = swlo & std::byte{3};
    auto swor2 = swlo | std::byte{3};
    auto swxor2 = swlo ^ std::byte{3};

    auto swand3 = std::byte{3} & swlo;
    auto swor3 = std::byte{3} | swlo;
    auto swxor3 = std::byte{3} ^ swlo;

    // bit binary op for 2 vec
    vop = vop1 && vop2;
    vop = vop1 || vop2;
    vop = vop1 >> vop2;
    vop = vop1 << vop2;

    vop = vop1 >> std::byte{3};
    vop = vop1 << std::byte{3};
    vop = std::byte{3} >> vop1;
    vop = std::byte{3} << vop1;

    // bit binary op for 2 swizzle
    swlo >> swhi;
    swlo << swhi;
    swlo >> std::byte{3};
    swlo << std::byte{3};
    auto right = std::byte{3} >> swhi;
    auto left = std::byte{3} << swhi;

    // condition op for 2 vec
    auto vres = vop1 == vop2;
    vres = vop1 != vop2;
    vres = vop1 > vop2;
    vres = vop1 < vop2;
    vres = vop1 >= vop2;
    vres = vop1 <= vop2;

    vres = vop1 == std::byte{3};
    vres = vop1 != std::byte{3};
    vres = vop1 > std::byte{3};
    vres = vop1 < std::byte{3};
    vres = vop1 >= std::byte{3};
    vres = vop1 <= std::byte{3};

    vres = std::byte{3} == vop1;
    vres = std::byte{3} != vop1;
    vres = std::byte{3} > vop1;
    vres = std::byte{3} < vop1;
    vres = std::byte{3} >= vop1;
    vres = std::byte{3} <= vop1;

    // condition op for 2 swizzle
    auto swres = swhi == swlo;
    auto swres1 = swhi != swlo;
    auto swres2 = swhi > swlo;
    auto swres3 = swhi < swlo;
    auto swres4 = swhi >= swlo;
    auto swres5 = swhi <= swlo;
    auto swres6 = swhi == std::byte{3};
    auto swres7 = swhi != std::byte{3};
    auto swres8 = swhi > std::byte{3};
    auto swres9 = swhi < std::byte{3};
    auto swres10 = swhi >= std::byte{3};
    auto swres11 = swhi <= std::byte{3};

    sycl::vec<std::byte, 3> voptest{std::byte{4}, std::byte{9}, std::byte{25}};
    auto bitv1 = ~vop3;
    auto bitv2 = !vop3;
    auto bitw = ~swhi;
  }

  return 0;
}