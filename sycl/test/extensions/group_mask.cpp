// RUN: %clangxx -g -O0 -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

//==-------- group_mask.cpp - SYCL group_mask test -------------------------==//
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
  auto g =
      sycl::detail::Builder::createGroupMask<sycl::ext::oneapi::group_mask>(
          sycl::marray<uint32_t, 4>{0});
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
  g.set(101, 1);
  g.set(11, 0);
  g.set(53, 1);
  assert(!g.none() && g.any() && !g.all());

  assert(g.count() == 2);
  assert(g.find_low() == 53);
  assert(g.find_high() == 101);
  assert(g.size() == 128);

  g.reset();
  assert(g.none() && !g.any() && !g.all());
  assert(g.find_low() == g.size() && g.find_high() == g.size());
  g.set();
  assert(!g.none() && g.any() && g.all());
  assert(g.find_low() == 0 && g.find_high() == 127);
  g.flip();
  assert(g.none() && !g.any() && !g.all());

  g.flip(13);
  g.flip(43);
  g.flip(79);
  auto b = g;
  assert(b == g && !(b != g));
  g.flip(101);
  assert(g.find_high() == 101);
  assert(b.find_high() == 79);
  assert(b != g && !(b == g));
  b.flip(101);
  assert(b == g && !(b != g));
  b = g >> 1;
  assert(b[12] && b[42] && b[78] && b[100]);
  b <<= 1;
  assert(b == g);
  g ^= ~b;
  assert(!g.none() && g.any() && g.all());
  assert((g | ~g).all());
  assert((g & ~g).none());
  assert((g ^ ~g).all());
  b.reset_low();
  b.reset_high();
  assert(!b[13] && b[43] && b[79] && !b[101]);
  b.insert_bits(sycl::marray<uint32_t, 4>{1, 2, 4, 8});
  assert(b[96] && b[65] && b[34] && b[3]);
  g = b;
  g <<= 33;
  assert(!g[96] && !g[65] && !g[34] && !g[3] && g[98] && g[67] && g[36]);
  b.insert_bits(sycl::marray<uint32_t, 4>{1, 1, 1, 1}, 15);
  assert(b[111] && !b[96] && b[79] && !b[65] && b[47] && !b[34] && b[15] &&
         b[3]);

  auto r = b.extract_bits<class sycl::marray<uint32_t, 4>>();
  for(size_t i=0; i<b.size();i++) {
    assert(b[i]==(bool)(r[3-(i/32)] & (1<<(i%32))));
  }
  b >>= 79;
  assert(b[32] && b[0]);
  b.flip(32);
  b.flip(0);
  assert(b.none());
  b.insert_bits((int)1);
}
