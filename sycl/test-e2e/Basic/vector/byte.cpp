// RUN: %{build} %cxx_std_optionc++17 -o %t.out
// RUN: %{run} %t.out

// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes %cxx_std_optionc++17 -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}

//==---------- vector_byte.cpp - SYCL vec<> for std::byte test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
#include <sycl/detail/vector_convert.hpp>
#include <sycl/vector.hpp>

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

      // logical binary assignment op for 2 vec
      auto VecByte3ACopy = VecByte3A;
      VecByte3ACopy &= VecByte3B;
      assert(VecByte3ACopy[0] == (VecByte3A[0] & VecByte3B[0]));
      VecByte3ACopy = VecByte3A;
      VecByte3ACopy |= VecByte3B;
      assert(VecByte3ACopy[1] == (VecByte3A[1] | VecByte3B[1]));
      VecByte3ACopy = VecByte3A;
      VecByte3ACopy ^= VecByte3B;
      assert(VecByte3ACopy[2] == (VecByte3A[2] ^ VecByte3B[2]));

      // logical binary op between swizzle and vec.
      using SwizType = sycl::vec<std::byte, 2>;
      auto SwizByte2And = SwizByte2A & (SwizType)SwizByte2B;
      auto SwizByte2Or = SwizByte2A | (SwizType)SwizByte2B;
      auto SwizByte2Xor = SwizByte2A ^ (SwizType)SwizByte2B;

      assert(SwizByte2And[0] == (VecByte4A[0] & VecByte4A[2]));
      assert(SwizByte2Or[1] == (VecByte4A[1] | VecByte4A[3]));
      assert(SwizByte2Xor[0] == (VecByte4A[0] ^ VecByte4A[2]));

      // logical binary assignment op between swizzle and vec.
      auto VecByte4ACopy = VecByte4A;
      auto SwizByte2ACopy = VecByte4ACopy.lo();
      SwizByte2ACopy &= (SwizType)SwizByte2B;
      assert(SwizByte2ACopy[0] == (SwizByte2A[0] & SwizByte2B[0]));
      VecByte4ACopy = VecByte4A;
      SwizByte2ACopy |= (SwizType)SwizByte2B;
      assert(SwizByte2ACopy[0] == (SwizByte2A[0] | SwizByte2B[0]));
      VecByte4ACopy = VecByte4A;
      SwizByte2ACopy ^= (SwizType)SwizByte2B;
      assert(SwizByte2ACopy[0] == (SwizByte2A[0] ^ SwizByte2B[0]));

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

      // Check overloads with scalar argument for bitwise assign operators.
      VecByte3ACopy = VecByte3A;
      VecByte3ACopy &= std::byte{3};
      assert(VecByte3ACopy[0] == (VecByte3A[0] & std::byte{3}));
      VecByte3ACopy = VecByte3A;
      VecByte3ACopy |= std::byte{3};
      assert(VecByte3ACopy[1] == (VecByte3A[1] | std::byte{3}));
      VecByte3ACopy = VecByte3A;
      VecByte3ACopy ^= std::byte{3};
      assert(VecByte3ACopy[2] == (VecByte3A[2] ^ std::byte{3}));

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

      // logical binary assign op for 1 swizzle
      VecByte4ACopy = VecByte4A;
      SwizByte2ACopy &= std::byte{3};
      assert(SwizByte2ACopy[0] == (SwizByte2A[0] & std::byte{3}));
      VecByte4ACopy = VecByte4A;
      SwizByte2ACopy |= std::byte{3};
      assert(SwizByte2ACopy[0] == (SwizByte2A[0] | std::byte{3}));
      VecByte4ACopy = VecByte4A;
      SwizByte2ACopy ^= std::byte{3};
      assert(SwizByte2ACopy[0] == (SwizByte2A[0] ^ std::byte{3}));

      // bit-wise negation test
      auto VecByte4Neg = ~VecByte4A;
      assert(VecByte4Neg[0] == ~VecByte4A[0]);

      auto SwizByte2Neg = ~SwizByte2B;
      assert(SwizByte2Neg[0] == ~SwizByte2B[0]);
    }
    
    // Test comparison operations on vec<std::byte> and swizzles.
    {
      auto SwizByte2A = VecByte4A.lo();
      auto SwizByte2B = VecByte4A.hi();

      // comparison op for 2 vec
      auto VecByte3Eq = VecByte3A == VecByte3B;
      auto VecByte3Neq = VecByte3A != VecByte3B;
      auto VecByte3Lt = VecByte3A < VecByte3B;
      auto VecByte3Lte = VecByte3A <= VecByte3B;
      auto VecByte3Gt = VecByte3A > VecByte3B;
      auto VecByte3Gte = VecByte3A >= VecByte3B;
      // Cast to bool since the result vector element is defined to be int8_t
      assert(static_cast<bool>(VecByte3Eq[0]) ==
             (VecByte3A[0] == VecByte3B[0]));
      assert(static_cast<bool>(VecByte3Neq[1]) ==
             (VecByte3A[1] != VecByte3B[1]));
      assert(static_cast<bool>(VecByte3Lt[2]) == (VecByte3A[2] < VecByte3B[2]));
      assert(static_cast<bool>(VecByte3Lte[0]) ==
             (VecByte3A[0] <= VecByte3B[0]));
      assert(static_cast<bool>(VecByte3Gt[1]) == (VecByte3A[1] > VecByte3B[1]));
      assert(static_cast<bool>(VecByte3Gte[2]) ==
             (VecByte3A[2] >= VecByte3B[2]));

      // comparison op between swizzle and vec.
      using SwizType = sycl::vec<std::byte, 2>;
      auto SwizByte2Eq = SwizByte2A == (SwizType)SwizByte2B;
      auto SwizByte2Neq = SwizByte2A != (SwizType)SwizByte2B;
      auto SwizByte2Lt = SwizByte2A < (SwizType)SwizByte2B;
      auto SwizByte2Lte = SwizByte2A <= (SwizType)SwizByte2B;
      auto SwizByte2Gt = SwizByte2A  > (SwizType)SwizByte2B;
      auto SwizByte2Gte = SwizByte2A >= (SwizType)SwizByte2B;
      // Cast to bool since the result vector element is defined to be int8_t
      assert(static_cast<bool>(SwizByte2Eq[0]) ==
             (VecByte4A[0] == VecByte4A[2]));
      assert(static_cast<bool>(SwizByte2Neq[0]) ==
             (VecByte4A[0] != VecByte4A[2]));
      assert(static_cast<bool>(SwizByte2Lt[0]) ==
             (VecByte4A[0] < VecByte4A[2]));
      assert(static_cast<bool>(SwizByte2Lte[0]) ==
             (VecByte4A[0] <= VecByte4A[2]));
      assert(static_cast<bool>(SwizByte2Gt[0]) ==
             (VecByte4A[0] > VecByte4A[2]));
      assert(static_cast<bool>(SwizByte2Gte[0]) ==
             (VecByte4A[0] >= VecByte4A[2]));

      // Check overloads with scalar argument for comparison operators.
      auto BitWiseEq1 = VecByte3A == std::byte{3};
      auto BitWiseNeq1 = VecByte3A != std::byte{3};
      auto BitWiseLt1 = VecByte3A < std::byte{3};
      auto BitWiseLte1 = VecByte3A <= std::byte{3};
      auto BitWiseGt1 = VecByte3A > std::byte{3};
      auto BitWiseGte1 = VecByte3A >= std::byte{3};
      auto BitWiseEq2 = std::byte{3} == VecByte3A;
      auto BitWiseNeq2 = std::byte{3} != VecByte3A;
      auto BitWiseLt2 = std::byte{3} < VecByte3A;
      auto BitWiseLte2 = std::byte{3} <= VecByte3A;
      auto BitWiseGt2 = std::byte{3} > VecByte3A;
      auto BitWiseGte2 = std::byte{3} >= VecByte3A;
      // Cast to bool since the result vector element is defined to be int8_t
      assert(static_cast<bool>(BitWiseEq1[0]) ==
             (VecByte3A[0] == std::byte{3}));
      assert(static_cast<bool>(BitWiseNeq1[0]) ==
             (VecByte3A[0] != std::byte{3}));
      assert(static_cast<bool>(BitWiseLt1[0]) == (VecByte3A[0] < std::byte{3}));
      assert(static_cast<bool>(BitWiseLte1[0]) ==
             (VecByte3A[0] <= std::byte{3}));
      assert(static_cast<bool>(BitWiseGt1[0]) == (VecByte3A[0] > std::byte{3}));
      assert(static_cast<bool>(BitWiseGte1[0]) ==
             (VecByte3A[0] >= std::byte{3}));
      assert(static_cast<bool>(BitWiseEq2[0]) ==
             (std::byte{3} == VecByte3A[0]));
      assert(static_cast<bool>(BitWiseNeq2[0]) ==
             (std::byte{3} != VecByte3A[0]));
      assert(static_cast<bool>(BitWiseLt2[0]) == (std::byte{3} < VecByte3A[0]));
      assert(static_cast<bool>(BitWiseLte2[0]) ==
             (std::byte{3} <= VecByte3A[0]));
      assert(static_cast<bool>(BitWiseGt2[0]) == (std::byte{3} > VecByte3A[0]));
      assert(static_cast<bool>(BitWiseGte2[0]) ==
             (std::byte{3} >= VecByte3A[0]));

      // logical binary op for 1 swizzle
      auto SwizByte2EqScalarA = SwizByte2A == std::byte{3};
      auto SwizByte2NeqScalarA = SwizByte2A != std::byte{3};
      auto SwizByte2LtScalarA = SwizByte2A < std::byte{3};
      auto SwizByte2LteScalarA = SwizByte2A <= std::byte{3};
      auto SwizByte2GtScalarA = SwizByte2A > std::byte{3};
      auto SwizByte2GteScalarA = SwizByte2A >= std::byte{3};
      auto SwizByte2EqScalarB = std::byte{3} == SwizByte2A;
      auto SwizByte2NeqScalarB = std::byte{3} != SwizByte2A;
      auto SwizByte2LtScalarB = std::byte{3} < SwizByte2A;
      auto SwizByte2LteScalarB = std::byte{3} <= SwizByte2A;
      auto SwizByte2GtScalarB = std::byte{3} > SwizByte2A;
      auto SwizByte2GteScalarB = std::byte{3} >= SwizByte2A;
      // Cast to bool since the result vector element is defined to be int8_t
      assert(static_cast<bool>(SwizByte2EqScalarA[0]) ==
             (SwizByte2A[0] == std::byte{3}));
      assert(static_cast<bool>(SwizByte2NeqScalarA[0]) ==
             (SwizByte2A[0] != std::byte{3}));
      assert(static_cast<bool>(SwizByte2LtScalarA[0]) ==
             (SwizByte2A[0] < std::byte{3}));
      assert(static_cast<bool>(SwizByte2LteScalarA[0]) ==
             (SwizByte2A[0] <= std::byte{3}));
      assert(static_cast<bool>(SwizByte2GtScalarA[0]) ==
             (SwizByte2A[0] > std::byte{3}));
      assert(static_cast<bool>(SwizByte2GteScalarA[0]) ==
             (SwizByte2A[0] >= std::byte{3}));
      assert(static_cast<bool>(SwizByte2EqScalarB[0]) ==
             (std::byte{3} == SwizByte2A[0]));
      assert(static_cast<bool>(SwizByte2NeqScalarB[0]) ==
             (std::byte{3} != SwizByte2A[0]));
      assert(static_cast<bool>(SwizByte2LtScalarB[0]) ==
             (std::byte{3} < SwizByte2A[0]));
      assert(static_cast<bool>(SwizByte2LteScalarB[0]) ==
             (std::byte{3} <= SwizByte2A[0]));
      assert(static_cast<bool>(SwizByte2GtScalarB[0]) ==
             (std::byte{3} > SwizByte2A[0]));
      assert(static_cast<bool>(SwizByte2GteScalarB[0]) ==
             (std::byte{3} >= SwizByte2A[0]));
    }

#if __SYCL_USE_LIBSYCL8_VEC_IMPL
    {
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
#endif
  }

  return 0;
}
