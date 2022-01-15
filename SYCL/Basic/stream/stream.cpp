// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// FIXME Disabled on host until sporadic failure is fixed
// RUNx: %HOST_RUN_PLACEHOLDER %t.out %HOST_CHECK_PLACEHOLDER
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_ON_LINUX_PLACEHOLDER %t.out %GPU_CHECK_ON_LINUX_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER

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

int main() {
  {
    default_selector Selector;
    queue Queue(Selector);
    context Context = Queue.get_context();

    // Check constructor and getters
    Queue.submit([&](handler &CGH) {
      stream Out(1024, 80, CGH,
                 property_list{property::buffer::context_bound{Context}});
      assert(Out.get_size() == 1024);
      assert(Out.get_max_statement_size() == 80);
      assert(Out.has_property<property::buffer::context_bound>());
      assert(!Out.has_property<property::queue::in_order>());
      assert(
          Out.get_property<property::buffer::context_bound>().get_context() ==
          Context);

      CGH.single_task<class DummyTask1>([=]() {});
    });

    // Check common reference semantics
    std::hash<stream> Hasher;

    Queue.submit([&](handler &CGH) {
      stream Out1(1024, 80, CGH);
      stream Out2(Out1);

      assert(Out1 == Out2);
      assert(Hasher(Out1) == Hasher(Out2));

      stream Out3(std::move(Out1));

      assert(Out2 == Out3);
      assert(Hasher(Out2) == Hasher(Out3));

      CGH.single_task<class DummyTask2>([=]() {});
    });

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
        Out << true << endl;
        Out << false << endl;
        // CHECK-NEXT: true
        // CHECK-NEXT: false

        // Integral types
        Out << static_cast<unsigned char>(123) << endl;
        Out << static_cast<signed char>(-123) << endl;
        Out << static_cast<short>(2344) << endl;
        Out << static_cast<signed short>(-2344) << endl;
        Out << 3454 << endl;
        Out << -3454 << endl;
        Out << 3454U << endl;
        Out << 3454L << endl;
        Out << 12345678901245UL << endl;
        Out << -12345678901245LL << endl;
        Out << 12345678901245ULL << endl;
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
        Out << 33.4f << endl;
        Out << 5.2 << endl;
        Out << -33.4f << endl;
        Out << -5.2 << endl;
        Out << 0.0003 << endl;
        Out << -1.0 / 0.0 << endl;
        Out << 1.0 / 0.0 << endl;
        Out << cl::sycl::sqrt(-1.0) << endl;
        Out << -1.0f / 0.0f << endl;
        Out << 1.0f / 0.0f << endl;
        Out << cl::sycl::sqrt(-1.0f) << endl;
        // CHECK-NEXT: 33.4
        // CHECK-NEXT: 5.2
        // CHECK-NEXT: -33.4
        // CHECK-NEXT: -5.2
        // CHECK-NEXT: 0.0003
        // CHECK-NEXT: -inf
        // CHECK-NEXT: inf
        // CHECK-NEXT: nan
        // CHECK-NEXT: -inf
        // CHECK-NEXT: inf
        // CHECK-NEXT: nan

        // Manipulators for integral types
        Out << dec << 0213 << endl;
        Out << dec << 0x213A << endl;
        Out << oct << 139 << endl;
        Out << hex << 8506 << endl;
        // CHECK-NEXT: 139
        // CHECK-NEXT: 8506
        // CHECK-NEXT: 213
        // CHECK-NEXT: 213a

        Out << oct << showbase << 8506 << ' ' << noshowbase << 8506 << endl;
        Out << hex << showbase << 8506 << ' ' << noshowbase << 8506 << endl;
        Out << dec << showbase << 8506 << ' ' << noshowbase << 8506 << endl;
        // CHECK-NEXT: 020472 20472
        // CHECK-NEXT: 0x213a 213a
        // CHECK-NEXT: 8506 8506

        Out << dec << showpos << 234 << ' ' << noshowpos << 234 << endl;
        Out << hex << showpos << 234 << ' ' << noshowpos << 234 << endl;
        Out << oct << showpos << 234 << ' ' << noshowpos << 234 << endl;
        // CHECK-NEXT: +234 234
        // CHECK-NEXT: ea ea
        // CHECK-NEXT: 352 352

        Out << hex << showpos << -1 << ' ' << noshowpos << -1 << endl;
        Out << oct << showpos << -1 << ' ' << noshowpos << -1 << endl;
        Out << dec << showpos << -1 << ' ' << noshowpos << -1 << endl;
        // CHECK-NEXT: ffffffff ffffffff
        // CHECK-NEXT: 37777777777 37777777777
        // CHECK-NEXT: -1 -1

        // Pointers
        int a = 5;
        int *Ptr = &a;
        Out << Ptr << endl;
        const int *const ConstPtr = &a;
        Out << ConstPtr << endl;
        auto multiPtr = private_ptr<int>(Ptr);
        Out << multiPtr << endl;
        // CHECK-NEXT: 0x{{[0-9a-fA-F]*$}}
        // CHECK-NEXT: 0x{{[0-9a-fA-F]*$}}
        // CHECK-NEXT: 0x{{[0-9a-fA-F]*$}}

        // Vectors
        vec<int, 1> f1(545);
        Out << f1 << endl;
        vec<int, 2> f2(545, 645);
        Out << f2 << endl;
        vec<int, 3> f3(545, 645, 771);
        Out << f3 << endl;
        vec<int, 4> f4(542325, 645, 771, 1024);
        Out << hex << showbase << f4 << endl;
        Out << dec << f4 << endl;
        vec<float, 4> f5(542.3f, 645.3f, 771.6f, 1024.2f);
        Out << f5 << endl;
        // CHECK-NEXT: 545
        // CHECK-NEXT: 545, 645
        // CHECK-NEXT: 545, 645, 771
        // CHECK-NEXT: 0x84675, 0x285, 0x303, 0x400
        // CHECK-NEXT: 542325, 645, 771, 1024
        // CHECK-NEXT: 542.3, 645.3, 771.6, 1024.2

        // Swizzles
        Out << f4.xyzw() << endl;
        // CHECK-NEXT: 542325, 645, 771, 1024

        // SYCL types
        Out << id<1>(23) << endl;
        Out << range<1>(32) << endl;
        Out << id<3>(11, 12, 13) << endl;
        Out << range<3>(11, 12, 13) << endl;
        Out << cl::sycl::nd_range<3>(cl::sycl::range<3>(2, 4, 1),
                                     cl::sycl::range<3>(1, 2, 1))
            << endl;
        // CHECK-NEXT: {23}
        // CHECK-NEXT: {32}
        // CHECK-NEXT: {11, 12, 13}
        // CHECK-NEXT: {11, 12, 13}
        // CHECK-NEXT: nd_range(global_range: {2, 4, 1}, local_range: {1, 2, 1},
        // offset: {0, 0, 0})
      });
    });
    Queue.wait();

    // Stream in parallel_for
    Queue.submit([&](handler &CGH) {
      stream Out(1024, 80, CGH);
      CGH.parallel_for<class stream_string>(
          range<1>(10), [=](id<1> i) { Out << "Hello, World!" << endl; });
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
              Out << item << endl;
          });
    });
    Queue.wait();
    // CHECK-NEXT: nd_item(global_id: {1, 2, 3}, local_id: {1, 0, 1})

    Queue.submit([&](handler &CGH) {
      stream Out(1024, 200, CGH);
      CGH.parallel_for_work_group<class stream_h_item>(
          range<3>(1, 1, 1), range<3>(1, 1, 1), [=](group<3> Group) {
            Group.parallel_for_work_item(
                [&](h_item<3> Item) { Out << Item << endl; });
          });
    });
    Queue.wait();
    // CHECK-NEXT: h_item(
    // CHECK-NEXT:   global item(range: {1, 1, 1}, id: {0, 0, 0})
    // CHECK-NEXT:   logical local item(range: {1, 1, 1}, id: {0, 0, 0})
    // CHECK-NEXT:   physical local item(range: {1, 1, 1}, id: {0, 0, 0})
    // CHECK-NEXT: )

    // Multiple streams in command group
    Queue.submit([&](handler &CGH) {
      stream Out1(1024, 80, CGH);
      stream Out2(500, 20, CGH);
      CGH.parallel_for<class multiple_streams>(range<1>(2), [=](id<1> i) {
        Out1 << "Hello, World!" << endl;
        Out2 << "Hello, World!" << endl;
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
          range<1>(2), [=](id<1> i) { Out << "aaaaaaaaa" << endl; });
    });
    Queue.wait();
    // CHECK-NEXT: aaaaaaaaa

    // Use a big statement size to verify the stream internal implementation can
    // create a big enough flush buffer in global memory to handle this case.
    range<1> global = 16;
    range<1> local = 16;
    Queue.submit([&](handler &cgh) {
      stream ostream(198, 8192, cgh);
      cgh.parallel_for<class test_stream>(
          nd_range<1>(global, local), [=](nd_item<1> it) {
            ostream << "global id " << it.get_global_id(0)
                    << stream_manipulator::endl;
          });
    });
    Queue.wait();
    // CHECK: global id {{[0-9]+}}
    // CHECK: global id {{[0-9]+}}
    // CHECK: global id {{[0-9]+}}
    // CHECK: global id {{[0-9]+}}
    // CHECK: global id {{[0-9]+}}
    // CHECK: global id {{[0-9]+}}
    // CHECK: global id {{[0-9]+}}
    // CHECK: global id {{[0-9]+}}
    // CHECK: global id {{[0-9]+}}
    // CHECK: global id {{[0-9]+}}
    // CHECK: global id {{[0-9]+}}
    // CHECK: global id {{[0-9]+}}
    // CHECK: global id {{[0-9]+}}
    // CHECK: global id {{[0-9]+}}
    // CHECK: global id {{[0-9]+}}
    // CHECK: global id {{[0-9]+}}
  }

  return 0;
}
