// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==------------ swizzle_op.cpp - SYCL SwizzleOp basic test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#define SYCL_SIMPLE_SWIZZLES

#include <cassert>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  {
    sycl::cl_float results[3] = {0};
    {
      buffer<sycl::cl_float, 1> b(results, range<1>(3));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_1>([=]() {
          sycl::cl_float2 ab = {4, 2};
          sycl::cl_float c = ab.x() * ab.y();
          B[0] = ab.x();
          B[1] = ab.y();
          B[2] = c;
        });
      });
    }
    assert(results[0] == 4);
    assert(results[1] == 2);
    assert(results[2] == 8);
  }

  {
    sycl::cl_float results[3] = {0};
    {
      buffer<sycl::cl_float, 1> b(results, range<1>(3));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_2>([=]() {
          sycl::cl_float2 ab = {4, 2};
          sycl::cl_float c = ab.x() * 2;
          B[0] = ab.x();
          B[1] = ab.y();
          B[2] = c;
        });
      });
    }
    assert(results[0] == 4);
    assert(results[1] == 2);
    assert(results[2] == 8);
  }

  {
    sycl::cl_float results[3] = {0};
    {
      buffer<sycl::cl_float, 1> b(results, range<1>(3));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_3>([=]() {
          sycl::cl_float2 ab = {4, 2};
          sycl::cl_float c = 4 * ab.y();
          B[0] = ab.x();
          B[1] = ab.y();
          B[2] = c;
        });
      });
    }
    assert(results[0] == 4);
    assert(results[1] == 2);
    assert(results[2] == 8);
  }

  {
    sycl::cl_float results[4] = {0};
    {
      buffer<sycl::cl_float, 1> b(results, range<1>(4));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_4>([=]() {
          sycl::cl_float2 ab = {4, 2};
          sycl::cl_float2 c = {0, 0};
          c.x() = ab.x() * ab.y();
          B[0] = ab.x();
          B[1] = ab.y();
          B[2] = c.x();
          B[4] = c.y();
        });
      });
    }
    assert(results[0] == 4);
    assert(results[1] == 2);
    assert(results[2] == 8);
    assert(results[3] == 0);
  }

  {
    sycl::cl_float results[4] = {0};
    {
      buffer<sycl::cl_float, 1> b(results, range<1>(4));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_5>([=]() {
          sycl::cl_float2 ab = {4, 2};
          sycl::cl_float2 c = {0, 0};
          c.x() = 4 * ab.y();
          B[0] = ab.x();
          B[1] = ab.y();
          B[2] = c.x();
          B[4] = c.y();
        });
      });
    }
    assert(results[0] == 4);
    assert(results[1] == 2);
    assert(results[2] == 8);
    assert(results[3] == 0);
  }

  {
    sycl::cl_float results[4] = {0};
    {
      buffer<sycl::cl_float, 1> b(results, range<1>(4));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_6>([=]() {
          sycl::cl_float2 ab = {4, 2};
          sycl::cl_float2 c = {0, 0};
          c.x() = ab.x() * 2;
          B[0] = ab.x();
          B[1] = ab.y();
          B[2] = c.x();
          B[4] = c.y();
        });
      });
    }
    assert(results[0] == 4);
    assert(results[1] == 2);
    assert(results[2] == 8);
    assert(results[3] == 0);
  }

  {
    sycl::cl_float results[6] = {0};
    {
      buffer<sycl::cl_float, 1> b(results, range<1>(6));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_7>([=]() {
          sycl::uchar4 abc = {4, 2, 1, 0};

          sycl::uchar4 c_each;
          c_each.x() = abc.x();
          c_each.y() = abc.y();
          c_each.z() = abc.z();

          sycl::uchar4 c_full;
          c_full = abc;

          B[0] = c_each.x();
          B[1] = c_each.y();
          B[2] = c_each.z();
          B[3] = c_full.x();
          B[4] = c_full.y();
          B[5] = c_full.z();
        });
      });
    }
    assert(results[0] == 4);
    assert(results[1] == 2);
    assert(results[2] == 1);
    assert(results[3] == 4);
    assert(results[4] == 2);
    assert(results[5] == 1);
  }

  {
    sycl::cl_float results[4] = {0};
    {
      buffer<sycl::cl_float, 1> b(results, range<1>(4));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_8>([=]() {
          sycl::uchar4 cba;
          sycl::uchar x = 1;
          sycl::uchar y = 2;
          sycl::uchar z = 3;
          sycl::uchar w = 4;
          cba.x() = x;
          cba.y() = y;
          cba.z() = z;
          cba.w() = w;

          sycl::uchar4 abc = {1, 2, 3, 4};
          abc.x() = cba.s0();
          abc.y() = cba.s1();
          abc.z() = cba.s2();
          abc.w() = cba.s3();
          if ((cba.x() == abc.x())) {
            abc.xy() = abc.xy() * 3;

            B[0] = abc.x();
            B[1] = abc.y();
            B[2] = abc.z();
            B[3] = abc.w();
          }
        });
      });
    }
    assert(results[0] == 3);
    assert(results[1] == 6);
    assert(results[2] == 3);
    assert(results[3] == 4);
  }

  {
    sycl::cl_uint results[4] = {0};
    {
      buffer<sycl::cl_uint, 1> b(results, range<1>(4));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_9>([=]() {
          sycl::uchar4 vec;
          sycl::uint add = 254;
          sycl::uchar factor = 2;
          vec.x() = 2;
          vec.y() = 4;
          vec.z() = 6;
          vec.w() = 8;

          B[0] = add + vec.x() / factor;
          B[1] = add + vec.y() / factor;
          B[2] = add + vec.z() / factor;
          B[3] = add + vec.w() / factor;
        });
      });
    }
    assert(results[0] == 255);
    assert(results[1] == 256);
    assert(results[2] == 257);
    assert(results[3] == 258);
  }

  {
    int FF[8] = {1, 1, 1, 0, 1, 1, 1, 0};
    {
      buffer<sycl::int3, 1> b((sycl::int3 *)FF, range<1>(2));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class test_10>(sycl::range<1>{2}, [=](sycl::id<1> ID) {
          B[ID] = sycl::int3{ID[0]} / B[ID];
        });
      });
    }
    assert(FF[0] == 0);
    assert(FF[1] == 0);
    assert(FF[2] == 0);
    assert(FF[4] == 1);
    assert(FF[5] == 1);
    assert(FF[6] == 1);
  }
  {
    sycl::int3 result = {0, 0, 0};
    {
      buffer<sycl::int3, 1> b(&result, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_11>([=]() {
          sycl::int3 testVec1 = {2, 2, 2};
          sycl::int3 testVec2 = {1, 1, 1};
          B[0] = testVec1 / testVec2;
        });
      });
    }
    const int r1 = result.x();
    const int r2 = result.y();
    const int r3 = result.z();
    assert(r1 == 2);
    assert(r2 == 2);
    assert(r3 == 2);
  }
  return 0;
}
