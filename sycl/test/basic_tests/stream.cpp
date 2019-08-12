// RUN: %clangxx -fsycl %s -o %t.out -lOpenCL
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out | FileCheck %s
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER
// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows
//==------------------ stream.cpp - SYCL stream basic test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL_SIMPLE_SWIZZLES
#include <CL/sycl.hpp>
#include <CL/sycl/context.hpp>
#include <cassert>

using namespace cl::sycl;
using sm = stream_manipulator;

int main() {
  {
    default_selector Selector;
    queue Queue(Selector);

    // Check constructor and getters
    Queue.submit([&](handler &CGH) {
      stream Out(1024, 80, CGH);
      assert(Out.get_size() == 1024);
      assert(Out.get_max_statement_size() == 80);
    });

    // Check common reference semantics
    hash_class<stream> Hasher;

    Queue.submit([&](handler &CGH) {
      stream Out1(1024, 80, CGH);
      stream Out2(Out1);

      assert(Out1 == Out2);
      assert(Hasher(Out1) == Hasher(Out2));

      stream Out3(std::move(Out1));

      assert(Out2 == Out3);
      assert(Hasher(Out2) == Hasher(Out3));
    });

    // TODO: support cl::sycl::endl. According to specitification endl should be
    // constant global variable in cl::sycl which is initialized with
    // strean_manipulator::endl. This approach doesn't currently work,
    // variable is not initialized in the kernel code, it contains some garbage
    // value.
    Queue.submit([&](handler &CGH) {
      stream Out(1024, 80, CGH);
      CGH.single_task<class integral>([=]() {
        // String
        Out << "Hello, World!\n";
// CHECK: Hello, World!

        // Char
        Out << 'a' << '\n';
// CHECK-NEXT: a

        // Boolean type
        Out << true << sm::endl;
        Out << false << sm::endl;
// CHECK-NEXT: true
// CHECK-NEXT: false

        // Integral types
        Out << static_cast<unsigned char>(123) << sm::endl;
        Out << static_cast<signed char>(-123) << sm::endl;
        Out << static_cast<short>(2344) << sm::endl;
        Out << static_cast<signed short>(-2344) << sm::endl;
        Out << 3454 << sm::endl;
        Out << -3454 << sm::endl;
        Out << 3454U << sm::endl;
        Out << 3454L << sm::endl;
        Out << 12345678901245UL << sm::endl;
        Out << -12345678901245LL << sm::endl;
        Out << 12345678901245ULL << sm::endl;
// CHECK-NEXT: 123
// CHECK-NEXT: -123
// CHECK-NEXT: 2344
// CHECK-NEXT: -2344
// CHECK-NEXT: 3454
// CHECK-NEXT: -3454
// CHECK-NEXT: 3454
// CHECK-NEXT: 3454
// CHECK-NEXT: 12345678901245
// CHECK-NEXT: -12345678901245
// CHECK-NEXT: 12345678901245

        // Floating point types
        Out << 33.4f << sm::endl;
        Out << 5.2 << sm::endl;
        Out << -33.4f << sm::endl;
        Out << -5.2 << sm::endl;
        Out << 0.0003 << sm::endl;
// CHECK-NEXT: 33.4
// CHECK-NEXT: 5.2
// CHECK-NEXT: -33.4
// CHECK-NEXT: -5.2
// CHECK-NEXT: 0.0003

        // Manipulators for integral types
        Out << sm::dec << 0213 << sm::endl;
        Out << sm::dec << 0x213A << sm::endl;
        Out << sm::oct << 139 << sm::endl;
        Out << sm::hex << 8506 << sm::endl;
// CHECK-NEXT: 139
// CHECK-NEXT: 8506
// CHECK-NEXT: 213
// CHECK-NEXT: 213a

        Out << sm::oct << sm::showbase << 8506 << ' ' << sm::noshowbase << 8506
            << sm::endl;
        Out << sm::hex << sm::showbase << 8506 << ' ' << sm::noshowbase << 8506
            << sm::endl;
        Out << sm::dec << sm::showbase << 8506 << ' ' << sm::noshowbase << 8506
            << sm::endl;
// CHECK-NEXT: 020472 20472
// CHECK-NEXT: 0x213a 213a
// CHECK-NEXT: 8506 8506

        Out << sm::dec << sm::showpos << 234 << ' ' << sm::noshowpos << 234
            << sm::endl;
        Out << sm::hex << sm::showpos << 234 << ' ' << sm::noshowpos << 234
            << sm::endl;
        Out << sm::oct << sm::showpos << 234 << ' ' << sm::noshowpos << 234
            << sm::endl;
// CHECK-NEXT: +234 234
// CHECK-NEXT: ea ea
// CHECK-NEXT: 352 352

        Out << sm::hex << sm::showpos << -1 << ' ' << sm::noshowpos << -1
            << sm::endl;
        Out << sm::oct << sm::showpos << -1 << ' ' << sm::noshowpos << -1
            << sm::endl;
        Out << sm::dec << sm::showpos << -1 << ' ' << sm::noshowpos << -1
            << sm::endl;
// CHECK-NEXT: ffffffff ffffffff
// CHECK-NEXT: 37777777777 37777777777
// CHECK-NEXT: -1 -1

        // Pointers
        int a = 5;
        int *Ptr = &a;
        Out << Ptr << sm::endl;
        const int *const ConstPtr = &a;
        Out << ConstPtr << sm::endl;
        auto multiPtr = private_ptr<int>(Ptr);
        Out << multiPtr << sm::endl;
// CHECK-NEXT: 0x{{[0-9a-fA-F]*$}}
// CHECK-NEXT: 0x{{[0-9a-fA-F]*$}}
// CHECK-NEXT: 0x{{[0-9a-fA-F]*$}}

        // Vectors
        vec<int, 1> f1(545);
        Out << f1 << sm::endl;
        vec<int, 2> f2(545, 645);
        Out << f2 << sm::endl;
        vec<int, 3> f3(545, 645, 771);
        Out << f3 << sm::endl;
        vec<int, 4> f4(542325, 645, 771, 1024);
        Out << sm::hex << sm::showbase << f4 << sm::endl;
        Out << sm::dec << f4 << sm::endl;
        vec<float, 4> f5(542.3f, 645.3f, 771.6f, 1024.2f);
        Out << f5 << sm::endl;
// CHECK-NEXT: 545
// CHECK-NEXT: 545, 645
// CHECK-NEXT: 545, 645, 771
// CHECK-NEXT: 0x84675, 0x285, 0x303, 0x400
// CHECK-NEXT: 542325, 645, 771, 1024
// CHECK-NEXT: 542.3, 645.3, 771.6, 1024.2

        // Swizzles
        Out << f4.xyzw() << sm::endl;
// CHECK-NEXT: 542325, 645, 771, 1024

        // SYCL types
        Out << id<3>(11, 12, 13) << sm::endl;
        Out << range<3>(11, 12, 13) << sm::endl;
        Out << cl::sycl::nd_range<3>(cl::sycl::range<3>(2, 4, 1),
                                     cl::sycl::range<3>(1, 2, 1))
            << sm::endl;
// CHECK-NEXT: {11, 12, 13}
// CHECK-NEXT: {11, 12, 13}
// CHECK-NEXT: nd_range(global_range: {2, 4, 1}, local_range: {1, 2, 1}, offset: {0, 0, 0})
      });
    });
    Queue.wait();

    // Stream in parallel_for
    Queue.submit([&](handler &CGH) {
      stream Out(1024, 80, CGH);
      CGH.parallel_for<class stream_string>(
          range<1>(10), [=](id<1> i) { Out << "Hello, World!\n"; });
    });
    Queue.wait();
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!

    // nd_item
    Queue.submit([&](handler &CGH) {
      stream Out(1024, 80, CGH);
      CGH.parallel_for<class stream_nd_range>(
          nd_range<3>(range<3>(4, 4, 4), range<3>(2, 2, 2)),
          [=](nd_item<3> item) {
            if (item.get_global_id(0) == 1 && item.get_global_id(1) == 2 &&
                item.get_global_id(2) == 3)
              Out << item << sm::endl;
          });
    });
    Queue.wait();
// CHECK-NEXT: nd_item(global_id: {1, 2, 3}, local_id: {1, 0, 1})

    Queue.submit([&](handler &CGH) {
      stream Out(1024, 80, CGH);
      CGH.parallel_for_work_group<class stream_h_item>(
          range<3>(1, 1, 1), range<3>(1, 1, 1), [=](group<3> Group) {
            Group.parallel_for_work_item(
                [&](h_item<3> Item) { Out << Item << sm::endl; });
          });
    });
// CHECK-NEXT: h_item(
// CHECK-NEXT:   global item(range: {1, 1, 1}, id: {0, 0, 0})
// CHECK-NEXT:   logical local item(range: {1, 1, 1}, id: {0, 0, 0})
// CHECK-NEXT:   physical local item(range: {1, 1, 1}, id: {0, 0, 0})
// CHECK-NEXT: )

    // Multiple streams in command group
    Queue.submit([&](handler &CGH) {
      stream Out1(1024, 80, CGH);
      stream Out2(500, 10, CGH);
      CGH.parallel_for<class multiple_streams>(range<1>(2), [=](id<1> i) {
        Out1 << "Hello, World!\n";
        Out2 << "Hello, World!\n";
      });
    });
    Queue.wait();
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!

    // The case when stream buffer is full. To check that there is no problem
    // with end of line symbol when printing out the stream buffer.
    Queue.submit([&](handler &CGH) {
      stream Out(10, 10, CGH);
      CGH.parallel_for<class full_stream_buffer>(
          range<1>(2), [=](id<1> i) { Out << "aaaaaaaaa\n"; });
    });
    Queue.wait();
  }
// CHECK-NEXT: aaaaaaaaa

  return 0;
}
