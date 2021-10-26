// RUN: %clangxx -g -O0 -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
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
  auto g = sycl::detail::Builder::createSubGroupMask<
      sycl::ext::oneapi::sub_group_mask>(0, 32);
  assert(g.none() && !g.any() && !g.all());
  assert(g[10] == false); // reference::operator[](id) const;
  g[10] = true;           // reference::operator=(bool);
  assert(g[10] == true);
  g[11] = g[10]; // reference::operator=(reference) reference::operator[](id);
  assert(g[10].flip() == false); // reference::flip()
  assert(~g[10] == true);        // refernce::operator~()
  assert(g[10] == false);
  assert(g[11] == true);
  assert(g.test(10) == false && g.test(11) == true);
  g.set(30, 1);
  g.set(11, 0);
  g.set(23, 1);
  assert(!g.none() && g.any() && !g.all());

  assert(g.count() == 2);
  assert(g.find_low() == 23);
  assert(g.find_high() == 30);
  assert(g.size() == 32);

  g.reset();
  assert(g.none() && !g.any() && !g.all());
  assert(g.find_low() == g.size() && g.find_high() == g.size());
  g.set();
  assert(!g.none() && g.any() && g.all());
  assert(g.find_low() == 0 && g.find_high() == 31);
  g.flip();
  assert(g.none() && !g.any() && !g.all());

  g.flip(13);
  g.flip(23);
  g.flip(29);
  auto b = g;
  assert(b == g && !(b != g));
  g.flip(31);
  assert(g.find_high() == 31);
  assert(b.find_high() == 29);
  assert(b != g && !(b == g));
  b.flip(31);
  assert(b == g && !(b != g));
  b = g >> 1;
  assert(b[12] && b[22] && b[28] && b[30]);
  b <<= 1;
  assert(b == g);
  g ^= ~b;
  assert(!g.none() && g.any() && g.all());
  assert((g | ~g).all());
  assert((g & ~g).none());
  assert((g ^ ~g).all());
  b.reset_low();
  b.reset_high();
  assert(!b[13] && b[23] && b[29] && !b[31]);
  b.insert_bits(0x01020408);
  assert(b[24] && b[17] && b[10] && b[3]);
  b <<= 13;
  assert(!b[24] && !b[17] && !b[10] && !b[3] && b[30] && b[23] && b[16]);
  b.insert_bits((char)0b01010101, 18);
  assert(b[18] && b[20] && b[22] && b[24] && b[30] && !b[23] && b[16]);
  b[3] = true;
  b.insert_bits(sycl::marray<char, 8>{1, 2, 4, 8, 16, 32, 64, 128}, 5);
  assert(!b[18] && !b[20] && !b[22] && !b[24] && !b[30] && !b[16] && b[3] &&
         b[5] && b[14] && b[23]);
  char r, rbc;
  const auto b_const{b};
  b.extract_bits(r);
  b_const.extract_bits(rbc);
  assert(r == 0b00101000);
  assert(rbc == 0b00101000);
  long r2 = -1, r2bc = -1;
  b.extract_bits(r2, 16);
  b_const.extract_bits(r2bc, 16);
  assert(r2 == 128);
  assert(r2bc == 128);

  b[31] = true;
  const auto b_const2{b};
  sycl::marray<char, 6> r3{-1}, r3bc{-1};
  b.extract_bits(r3, 14);
  b_const2.extract_bits(r3bc, 14);
  assert(r3[0] == 1 && r3[1] == 2 && r3[2] == 2 && !r3[3] && !r3[4] && !r3[5]);
  assert(r3bc[0] == 1 && r3bc[1] == 2 && r3bc[2] == 2 && !r3bc[3] && !r3bc[4] &&
         !r3bc[5]);
  int ibits = 0b1010101010101010101010101010101;
  b.insert_bits(ibits);
  for (size_t i = 0; i < 32; i++) {
    assert(b[i] != (bool)(i % 2));
  }
  short sbits = 0b0111011101110111;
  b.insert_bits(sbits, 7);
  b.extract_bits(ibits);
  assert(ibits == 0b1010101001110111011101111010101);
  sbits = 0b1100001111000011;
  b.insert_bits(sbits, 23);
  b.extract_bits(ibits);
  assert(ibits == 0b11100001101110111011101111010101);
  int64_t lbits = -1;
  b.extract_bits(lbits, 33);
  assert(lbits == 0);
  lbits = -1;
  b.extract_bits(lbits, 5);
  assert(lbits == 0b111000011011101110111011110);
  lbits = -1;
  b.insert_bits(lbits);
  assert(b.all());
}
