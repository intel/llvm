// RUN: %{build} -std=c++17 -o %t.out
// RUN: %{run} %t.out

// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -std=c++17 -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}

//==---------- vector_byte.cpp - SYCL vec<> for std::byte test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
#include <sycl/types.hpp>

#include <cstddef> // std::byte
#include <tuple>   // std::ignore

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
    std::ignore = std::byte(vb1.x());
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

    assert(cnv[0] == std::byte{1} && cnv[1] == std::byte{1});
    assert(cnv2[0] == 3);

    auto asint = vb2.template as<sycl::vec<int16_t, 1>>();
    auto asbyte = vi2.template as<sycl::vec<std::byte, 8>>();

    // 0000 0111 0000 0111 = 119
    assert(asint[0] == 1799);

    // 0000 0000 0000 0001 0000 0000 0000 0001
    assert(asbyte[0] == std::byte{1} && asbyte[1] == std::byte{0} &&
           asbyte[2] == std::byte{0} && asbyte[3] == std::byte{0} &&
           asbyte[4] == std::byte{1} && asbyte[5] == std::byte{0} &&
           asbyte[6] == std::byte{0} && asbyte[7] == std::byte{0});
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
                            sycl::access::address_space::global_space,
                            sycl::access::decorated::yes>
                mp(Acc);
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
    assert(vbswhi[0] == std::byte{2} && vbswhi[1] == std::byte{3});

    vbswhi = vbsw.lo();
    assert(vbswhi[0] == std::byte{0} && vbswhi[1] == std::byte{1});

    vbswhi = vbsw.odd();
    assert(vbswhi[0] == std::byte{1} && vbswhi[1] == std::byte{3});

    vbswhi = vbsw.even();
    assert(vbswhi[0] == std::byte{0} && vbswhi[1] == std::byte{2});
  }

  // operatorOP for vec and for swizzle
  {
    sycl::vec<std::byte, 3> VecByte3A{std::byte{4}, std::byte{9},
                                      std::byte{25}};
    sycl::vec<std::byte, 3> VecByte3B{std::byte{2}, std::byte{3}, std::byte{5}};
    sycl::vec<std::byte, 4> VecByte4A{std::byte{5}, std::byte{6}, std::byte{2},
                                      std::byte{3}};

    // Test bitwise operations on vec<std::byte> and swizzles.
    {
      auto SwizByte2A = VecByte4A.lo();
      auto SwizByte2B = VecByte4A.hi();

      // logical binary op for 2 vec
      auto VecByte3And = VecByte3A & VecByte3B;
      auto VecByte3Or = VecByte3A | VecByte3B;
      auto VecByte3Xor = VecByte3A ^ VecByte3B;
      assert(VecByte3And[0] == (VecByte3A[0] & VecByte3B[0]));
      assert(VecByte3Or[1] == (VecByte3A[1] | VecByte3B[1]));
      assert(VecByte3Xor[2] == (VecByte3A[2] ^ VecByte3B[2]));

      // logical binary op between swizzle and vec.
      using SwizType = sycl::vec<std::byte, 2>;
      auto SwizByte2And = SwizByte2A & (SwizType)SwizByte2B;
      auto SwizByte2Or = SwizByte2A | (SwizType)SwizByte2B;
      auto SwizByte2Xor = SwizByte2A ^ (SwizType)SwizByte2B;

      assert(SwizByte2And[0] == (VecByte4A[0] & VecByte4A[2]));
      assert(SwizByte2Or[1] == (VecByte4A[1] | VecByte4A[3]));
      assert(SwizByte2Xor[0] == (VecByte4A[0] ^ VecByte4A[2]));

      // Check overloads with scalar argument for bitwise operators.
      auto BitWiseAnd1 = VecByte3A & std::byte{3};
      auto BitWiseOr1 = VecByte3A | std::byte{3};
      auto BitWiseXor1 = VecByte3A ^ std::byte{3};
      auto BitWiseAnd2 = std::byte{3} & VecByte3A;
      auto BitWiseOr2 = std::byte{3} | VecByte3A;
      auto BitWiseXor2 = std::byte{3} ^ VecByte3A;
      assert(BitWiseAnd1[0] == BitWiseAnd2[0]);
      assert(BitWiseOr1[1] == BitWiseOr2[1]);
      assert(BitWiseXor1[2] == BitWiseXor2[2]);

      // logical binary op for 1 swizzle
      auto SwizByte2AndScalarA = SwizByte2A & std::byte{3};
      auto SwizByte2OrScalarA = SwizByte2A | std::byte{3};
      auto SwizByte2XorScalarA = SwizByte2A ^ std::byte{3};
      auto SwizByte2AndScalarB = std::byte{3} & SwizByte2A;
      auto SwizByte2OrScalarB = std::byte{3} | SwizByte2A;
      auto SwizByte2XorScalarB = std::byte{3} ^ SwizByte2A;
      assert(SwizByte2AndScalarA[0] == SwizByte2AndScalarB[0]);
      assert(SwizByte2OrScalarA[1] == SwizByte2OrScalarB[1]);
      assert(SwizByte2XorScalarA[0] == SwizByte2XorScalarB[0]);

      // bit-wise negation test
      auto VecByte4Neg = ~VecByte4A;
      assert(VecByte4Neg[0] == ~VecByte4A[0]);

      auto SwizByte2Neg = ~SwizByte2B;
      assert(SwizByte2Neg[0] == ~SwizByte2B[0]);
    }

    // std::byte is not an arithmetic type or a character type, so std::byte
    // and vec<std::byte> should not support artithmetic operations. In the
    // new implementation of vec<> class, the following will be removed.
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
    {
      // binary op for 2 vec
      auto vop = VecByte3A + VecByte3B;
      assert(vop[0] == std::byte{6});
      vop = VecByte3A - VecByte3B;
      vop = VecByte3A * VecByte3B;
      vop = VecByte3A / VecByte3B;
      assert(vop[0] == std::byte{2});
      vop = VecByte3A % VecByte3B;

      // binary op for 2 swizzle
      auto swlo = VecByte4A.lo();
      auto swhi = VecByte4A.hi();
      auto swplus = swlo + swhi;
      sycl::vec<std::byte, 2> vec_test = swplus;
      assert(vec_test.x() == std::byte{7} && vec_test.y() == std::byte{9});
      auto swominus = swlo - swhi;
      auto swmul = swlo * swhi;
      vec_test = swmul;
      assert(vec_test.x() == std::byte{10} && vec_test.y() == std::byte{18});
      auto swdiv = swlo / swhi;

      // binary op for 1 vec
      vop = VecByte3A + std::byte{3};
      vop = VecByte3A - std::byte{3};
      assert(vop[1] == std::byte{6});
      vop = VecByte3A * std::byte{3};
      vop = VecByte3A / std::byte{3};
      vop = VecByte3A % std::byte{3};
      assert(vop[0] == std::byte{1});

      vop = std::byte{3} + VecByte3A;
      assert(vop[0] == std::byte{7});
      vop = std::byte{3} - VecByte3A;
      vop = std::byte{3} * VecByte3A;
      assert(vop[2] == std::byte{75});
      vop = std::byte{3} / VecByte3A;

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
      vop = vbuf += VecByte3A;
      assert(vop[0] == std::byte{8});
      vop = vbuf -= VecByte3A;
      vop = vbuf *= VecByte3A;
      vop = vbuf /= VecByte3A;
      vop = vbuf %= VecByte3A;

      // operatorOP= for 2 swizzle
      swlo += swhi;
      swlo -= swhi;
      vec_test = swlo;
      assert(vec_test.x() == std::byte{5} && vec_test.y() == std::byte{6});
      swlo *= swhi;
      swlo /= swhi;
      swlo %= swhi;

      // operatorOP= for 1 vec
      vop = VecByte3A += std::byte{3};
      assert(vop[0] == std::byte{7});
      vop = VecByte3A -= std::byte{3};
      vop = VecByte3A *= std::byte{3};
      vop = VecByte3A /= std::byte{3};
      vop = VecByte3A %= std::byte{3};

      // operatorOP= for 1 swizzle
      swlo += std::byte{3};
      swlo -= std::byte{1};
      vec_test = swlo;
      assert(vec_test.x() == std::byte{3} && vec_test.y() == std::byte{2});
      swlo *= std::byte{3};
      swlo /= std::byte{3};
      swlo %= std::byte{3};

      // unary operator++ and -- for vec
      VecByte3A =
          sycl::vec<std::byte, 3>(std::byte{4}, std::byte{9}, std::byte{25});
      VecByte3A++;
      VecByte3A--;
      vop = ++VecByte3A;
      assert(vop[2] == std::byte{26});
      --VecByte3A;

      // unary operator++ and -- for swizzle
      swlo++;
      swlo--;
      vec_test = swlo;
      assert(vec_test.x() == std::byte{0} && vec_test.y() == std::byte{2});
    }

    // Logical operations on vec<byte> and swizzles.
    {
      // condition op for 2 vec
      auto vres = VecByte3A == VecByte3B;
      vres = VecByte3A != VecByte3B;
      vres = VecByte3A > VecByte3B;
      vres = VecByte3A < VecByte3B;
      vres = VecByte3A >= VecByte3B;
      vres = VecByte3A <= VecByte3B;

      vres = VecByte3A == std::byte{3};
      vres = VecByte3A != std::byte{3};
      vres = VecByte3A > std::byte{3};
      vres = VecByte3A < std::byte{3};
      vres = VecByte3A >= std::byte{3};
      vres = VecByte3A <= std::byte{3};

      vres = std::byte{3} == VecByte3A;
      vres = std::byte{3} != VecByte3A;
      vres = std::byte{3} > VecByte3A;
      vres = std::byte{3} < VecByte3A;
      vres = std::byte{3} >= VecByte3A;
      vres = std::byte{3} <= VecByte3A;

      auto swlo = VecByte4A.lo();
      auto swhi = VecByte4A.hi();

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

      // bit binary operations
      auto vop = VecByte3A && VecByte3B;
      vop = VecByte3A || VecByte3B;

      auto vop1 = VecByte3A >> VecByte3B;
      vop1 = VecByte3A << VecByte3B;

      vop1 = VecByte3A >> std::byte{3};
      vop1 = VecByte3A << std::byte{3};
      vop1 = std::byte{3} >> VecByte3A;
      vop1 = std::byte{3} << VecByte3A;

      swlo >> swhi;
      swlo << swhi;
      swlo >> std::byte{3};
      swlo << std::byte{3};
      auto right = std::byte{3} >> swhi;
      auto left = std::byte{3} << swhi;

      auto bitv2 = !VecByte4A;
    }
#else
    {
      // std::byte is not an arithmetic type and it only supports the following
      // overloads of >> and << operators.
      //
      // 1 template <class IntegerType>
      //   constexpr std::byte operator<<( std::byte b, IntegerType shift )
      //   noexcept;
      // 2 template <class IntegerType>
      //   constexpr std::byte operator>>( std::byte b, IntegerType shift )
      //   noexcept;
      auto VecByte3Shift = VecByte3A << 3;
      assert(VecByte3Shift[0] == VecByte3A[0] << 3 &&
             VecByte3Shift[1] == VecByte3A[1] << 3 &&
             VecByte3Shift[2] == VecByte3A[2] << 3);

      VecByte3Shift = VecByte3A >> 1;
      assert(VecByte3Shift[0] == VecByte3A[0] >> 1 &&
             VecByte3Shift[1] == VecByte3A[1] >> 1 &&
             VecByte3Shift[2] == VecByte3A[2] >> 1);

      auto SwizByte2Shift = VecByte4A.lo();
      using VecType = sycl::vec<std::byte, 2>;
      auto SwizShiftRight = (VecType)(SwizByte2Shift >> 3);
      auto SwizShiftLeft = (VecType)(SwizByte2Shift << 3);
      assert(SwizShiftRight[0] == SwizByte2Shift[0] >> 3 &&
             SwizShiftLeft[1] == SwizByte2Shift[1] << 3);
    }
#endif // __INTEL_PREVIEW_BREAKING_CHANGES
  }

  return 0;
}
