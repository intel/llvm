// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

//==-------- sub_group_mask.cpp - SYCL sub-group mask test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

int main() {
  for (size_t sgsize = 32; sgsize > 4; sgsize /= 2) {
    std::cout << "Running test for sub-group size = " << sgsize << std::endl;
    auto g = sycl::detail::Builder::createSubGroupMask<
        sycl::ext::oneapi::sub_group_mask>(0, sgsize);
    assert(g.none() && !g.any() && !g.all());
    assert(g[5] == false); // reference::operator[](id) const;
    g[5] = true;           // reference::operator=(bool);
    assert(g[5] == true);
    g[6] = g[5]; // reference::operator=(reference) reference::operator[](id);
    assert(g[5].flip() == false);   // reference::flip()
    assert(~g[5 % sgsize] == true); // refernce::operator~()
    assert(g[5 % sgsize] == false);
    assert(g[6 % sgsize] == true);
    assert(g.test(5 % sgsize) == false && g.test(6 % sgsize) == true);
    g.set(3 % sgsize, 1);
    g.set(6 % sgsize, 0);
    g.set(2 % sgsize, 1);
    assert(!g.none() && g.any() && !g.all());

    assert(g.count() == 2);
    assert(g.find_low() == 2 % sgsize);
    assert(g.find_high() == 3 % sgsize);
    assert(g.size() == sgsize);

    g.reset();
    assert(g.none() && !g.any() && !g.all());
    assert(g.find_low() == g.size() && g.find_high() == g.size());
    g.set();
    assert(!g.none() && g.any() && g.all());
    assert(g.find_low() == 0 && g.find_high() == 31 % sgsize);
    g.flip();
    assert(g.none() && !g.any() && !g.all());

    g.flip(2);
    g.flip(3);
    g.flip(7);
    auto b = g;
    assert(b == g && !(b != g));
    g.flip(7);
    assert(g.find_high() == 3 % sgsize);
    assert(b.find_high() == 7 % sgsize);
    assert(b != g && !(b == g));
    g.flip(7);
    assert(b == g && !(b != g));
    b = g >> 1;
    assert(b[1] && b[2] && b[6]);
    b <<= 1;
    assert(b == g);
    g ^= ~b;
    assert(!g.none() && g.any() && g.all());
    assert((g | ~g).all());
    assert((g & ~g).none());
    assert((g ^ ~g).all());
    b.reset_low();
    b.reset_high();
    assert(!b[2] && b[3] && !b[7]);
    b.insert_bits(0x01020408);
    assert(((b[24] && b[17]) || sgsize < 32) && (b[10] || sgsize < 16) && b[3]);
    b <<= 10;
    assert(((!b[24] && !b[17] && b[27] && b[20]) || sgsize < 32) &&
           ((!b[10] && b[13]) || sgsize < 16) && !b[3]);
    b.insert_bits((char)0b01010101, 6);
    assert(b[6] && ((b[8] && b[10] && b[12] && !b[13]) || sgsize < 16));
    b[3] = true;
    b.insert_bits(sycl::marray<char, 8>{1, 2, 4, 8, 16, 32, 64, 128}, 5);
    assert(
        ((!b[18] && !b[20] && !b[22] && !b[24] && !b[30] && !b[16] && b[23]) ||
         sgsize < 32) &&
        b[3] && b[5] && (b[14] || sgsize < 16));
    b.flip(14);
    b.flip(23);
    char r, rbc;
    const auto b_const{b};
    b.extract_bits(r);
    b_const.extract_bits(rbc);
    assert(r == 0b00101000);
    assert(rbc == 0b00101000);
    long r2 = -1, r2bc = -1;
    b.extract_bits(r2, 3);
    b_const.extract_bits(r2bc, 3);
    assert(r2 == 5);
    assert(r2bc == 5);

    b.insert_bits((uint32_t)0x08040201);
    const auto b_const2{b};
    sycl::marray<char, 6> r3{-1}, r3bc{-1};
    b.extract_bits(r3);
    b_const2.extract_bits(r3bc);
    assert(r3[0] == 1 && r3[1] == (sgsize > 8 ? 2 : 0) &&
           r3[2] == (sgsize > 16 ? 4 : 0) && r3[3] == (sgsize > 16 ? 8 : 0) &&
           !r3[4] && !r3[5]);
    assert(r3bc[0] == 1 && r3bc[1] == (sgsize > 8 ? 2 : 0) &&
           r3bc[2] == (sgsize > 16 ? 4 : 0) &&
           r3bc[3] == (sgsize > 16 ? 8 : 0) && !r3bc[4] && !r3bc[5]);
    int ibits = 0b1010101010101010101010101010101;
    b.insert_bits(ibits);
    for (size_t i = 0; i < sgsize; i++) {
      assert(b[i] != (bool)(i % 2));
    }
    short sbits = 0b0111011101110111;
    b.insert_bits(sbits, 7);
    b.extract_bits(ibits);
    assert(ibits ==
           (0b1010101001110111011101111010101 & ((1ULL << sgsize) - 1ULL)));
    sbits = 0b1100001111000011;
    b.insert_bits(sbits, 23);
    b.extract_bits(ibits);
    if (sgsize >= 32) {
      int64_t lbits = -1;
      b.extract_bits(lbits, 33);
      assert(lbits == 0);
      lbits = -1;
      b.extract_bits(lbits, 5);
      assert(lbits ==
             (0b111000011011101110111011110 & ((1ULL << sgsize) - 1ULL)));
      lbits = -1;
      b.insert_bits(lbits);
      assert(b.all());
    }
  }
}
