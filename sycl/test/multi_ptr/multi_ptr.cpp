// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-dead-args-optimization %s -o %t.out
// RUN: %RUN_ON_HOST %t.out
// RUN: %clangxx -DRESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR -fsycl -fsycl-targets=%sycl_triple -fsycl-dead-args-optimization %s -o %t1.out
// RUN: %RUN_ON_HOST %t1.out

//==--------------- multi_ptr.cpp - SYCL multi_ptr test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>
#include <type_traits>

using namespace cl::sycl;

/* This is the class used to name the kernel for the runtime.
 * This must be done when the kernel is expressed as a lambda. */
template <typename T> class testMultPtrKernel;
template <typename T> class testMultPtrArrowOperatorKernel;

template <typename T> struct point {
  point(const point &rhs) : x(rhs.x), y(rhs.y) {}
  point(T x, T y) : x(x), y(y) {}
  point(T v) : x(v), y(v) {}
  point() : x(0), y(0) {}
  bool operator==(const T &rhs) { return rhs == x && rhs == y; }
  bool operator==(const point<T> &rhs) { return rhs.x == x && rhs.y == y; }
  T x;
  T y;
};

template <typename T>
void innerFunc(id<1> wiID, global_ptr<T> ptr_1, global_ptr<T> ptr_2,
               local_ptr<T> local_ptr) {
  T t = ptr_1[wiID.get(0)];
  local_ptr[wiID.get(0)] = t;
  t = local_ptr[wiID.get(0)];
  ptr_2[wiID.get(0)] = t;
}

template <typename T> void testMultPtr() {
  T data_1[10];
  for (size_t i = 0; i < 10; ++i) {
    data_1[i] = 1;
  }
  T data_2[10];
  for (size_t i = 0; i < 10; ++i) {
    data_2[i] = 2;
  }

  {
    range<1> numOfItems{10};
    buffer<T, 1> bufferData_1(data_1, numOfItems);
    buffer<T, 1> bufferData_2(data_2, numOfItems);
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      accessor<T, 1, access::mode::read, access::target::global_buffer,
               access::placeholder::false_t>
          accessorData_1(bufferData_1, cgh);
      accessor<T, 1, access::mode::read_write, access::target::global_buffer,
               access::placeholder::false_t>
          accessorData_2(bufferData_2, cgh);
      accessor<T, 1, access::mode::read_write, access::target::local>
          localAccessor(numOfItems, cgh);

      cgh.parallel_for<class testMultPtrKernel<T>>(range<1>{10}, [=](id<1> wiID) {
        auto ptr_1 = make_ptr<T, access::address_space::global_space>(
          accessorData_1.get_pointer());
        auto ptr_2 = make_ptr<T, access::address_space::global_space>(
          accessorData_2.get_pointer());
        auto local_ptr = make_ptr<T, access::address_space::local_space>(
          localAccessor.get_pointer());

        // General conversions in multi_ptr class
        T *RawPtr = nullptr;
        global_ptr<T> ptr_4(RawPtr);
        ptr_4 = RawPtr;

        global_ptr<T> ptr_5(accessorData_1);

        global_ptr<void> ptr_6((void *)RawPtr);

        ptr_6 = (void *)RawPtr;

        // Explicit conversions for device_ptr/host_ptr to global_ptr
        device_ptr<void> ptr_7((void *)RawPtr);
        global_ptr<void> ptr_8 = global_ptr<void>(ptr_7);
        host_ptr<void> ptr_9((void *)RawPtr);
        global_ptr<void> ptr_10 = global_ptr<void>(ptr_9);
        // TODO: need propagation of a7b763b26 patch to acl tool before testing
        // these conversions - otherwise the test would fail on accelerator
        // device during reversed translation from SPIR-V to LLVM IR
        // device_ptr<T> ptr_11(accessorData_1);
        // global_ptr<T> ptr_12 = global_ptr<T>(ptr_11);

        innerFunc<T>(wiID.get(0), ptr_1, ptr_2, local_ptr);
      });
    });
  }
  for (size_t i = 0; i < 10; ++i) {
    assert(data_1[i] == 1 && "Expected data_1[i] == 1");
  }
  for (size_t i = 0; i < 10; ++i) {
    assert(data_2[i] == 1 && "Expected data_2[i] == 1");
  }
}

template <typename T>
void testMultPtrArrowOperator() {
  point<T> data_1[1] = {1};
  point<T> data_2[1] = {2};
  point<T> data_3[1] = {3};
  point<T> data_4[1] = {4};

  {
    range<1> numOfItems{1};
    buffer<point<T>, 1> bufferData_1(data_1, numOfItems);
    buffer<point<T>, 1> bufferData_2(data_2, numOfItems);
    buffer<point<T>, 1> bufferData_3(data_3, numOfItems);
    buffer<point<T>, 1> bufferData_4(data_4, numOfItems);
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      accessor<point<T>, 1, access::mode::read, access::target::global_buffer,
               access::placeholder::false_t>
          accessorData_1(bufferData_1, cgh);
      accessor<point<T>, 1, access::mode::read, access::target::constant_buffer,
               access::placeholder::false_t>
          accessorData_2(bufferData_2, cgh);
      accessor<point<T>, 1, access::mode::read_write, access::target::local,
               access::placeholder::false_t>
          accessorData_3(1, cgh);
      accessor<point<T>, 1, access::mode::read, access::target::global_buffer,
               access::placeholder::false_t>
          accessorData_4(bufferData_4, cgh);

      cgh.single_task<class testMultPtrArrowOperatorKernel<T>>([=]() {
        auto ptr_1 = make_ptr<point<T>, access::address_space::global_space>(
            accessorData_1.get_pointer());
        auto ptr_2 = make_ptr<point<T>, access::address_space::constant_space>(
            accessorData_2.get_pointer());
        auto ptr_3 = make_ptr<point<T>, access::address_space::local_space>(
            accessorData_3.get_pointer());
        auto ptr_4 = make_ptr<point<T>, access::address_space::global_device_space>(
            accessorData_4.get_pointer());

        auto x1 = ptr_1->x;
        auto x2 = ptr_2->x;
        auto x3 = ptr_3->x;
        auto x4 = ptr_4->x;

        static_assert(std::is_same<decltype(x1), T>::value,
                      "Expected decltype(ptr_1->x) == T");
        static_assert(std::is_same<decltype(x2), T>::value,
                      "Expected decltype(ptr_2->x) == T");
        static_assert(std::is_same<decltype(x3), T>::value,
                      "Expected decltype(ptr_3->x) == T");
        static_assert(std::is_same<decltype(x4), T>::value,
                      "Expected decltype(ptr_4->x) == T");
      });
    });
  }
}

int main() {
  testMultPtr<int>();
  testMultPtr<float>();
  testMultPtr<point<int>>();
  testMultPtr<point<float>>();

  testMultPtrArrowOperator<int>();
  testMultPtrArrowOperator<float>();

  return 0;
}
