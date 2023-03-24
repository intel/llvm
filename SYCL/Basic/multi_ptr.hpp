//==------------ multi_ptr.cpp - SYCL multi_ptr test header ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>
#include <type_traits>

using namespace sycl;

/* This is the class used to name the kernel for the runtime.
 * This must be done when the kernel is expressed as a lambda. */
template <typename T, access::decorated IsDecorated> class testMultPtrKernel;
template <typename T, access::decorated IsDecorated>
class testMultPtrArrowOperatorKernel;

template <typename T> struct point {
  point(const point &rhs) = default;
  point(T x, T y) : x(x), y(y) {}
  point(T v) : x(v), y(v) {}
  point() : x(0), y(0) {}
  bool operator==(const T &rhs) { return rhs == x && rhs == y; }
  bool operator==(const point<T> &rhs) { return rhs.x == x && rhs.y == y; }
  T x;
  T y;
};

template <typename T, access::decorated IsDecorated>
void innerFunc(id<1> wiID, global_ptr<T, IsDecorated> ptr_1,
               global_ptr<T, IsDecorated> ptr_2,
               global_ptr<T, IsDecorated> ptr_3,
               global_ptr<T, IsDecorated> ptr_4,
               global_ptr<T, IsDecorated> ptr_5,
               local_ptr<T, IsDecorated> local_ptr,
               private_ptr<T, IsDecorated> priv_ptr) {
  T t = ptr_1[wiID.get(0)];

  // Write to ptr_2 using local_ptr subscript.
  local_ptr[wiID.get(0)] = t;
  ptr_2[wiID.get(0)] = local_ptr[wiID.get(0)];

  // Reset local ptr
  local_ptr[wiID.get(0)] = 0;

  // Write to ptr_3 using dereferencing.
  *(local_ptr + wiID.get(0)) = t;
  *(ptr_3 + wiID.get(0)) = *(local_ptr + wiID.get(0));

  // Write to ptr_2 using priv_ptr subscript.
  priv_ptr[wiID.get(0)] = t;
  ptr_4[wiID.get(0)] = priv_ptr[wiID.get(0)];

  // Reset local ptr
  priv_ptr[wiID.get(0)] = 0;

  // Write to ptr_3 using dereferencing.
  *(priv_ptr + wiID.get(0)) = t;
  *(ptr_5 + wiID.get(0)) = *(priv_ptr + wiID.get(0));
}

template <typename T, access::decorated IsDecorated> void testMultPtr() {
  T data_1[10];
  T data_2[10];
  T data_3[10];
  T data_4[10];
  T data_5[10];
  for (size_t i = 0; i < 10; ++i) {
    data_1[i] = 1;
    data_2[i] = 2;
    data_3[i] = 3;
    data_4[i] = 4;
    data_5[i] = 5;
  }

  {
    range<1> numOfItems{10};
    buffer<T, 1> bufferData_1(data_1, numOfItems);
    buffer<T, 1> bufferData_2(data_2, numOfItems);
    buffer<T, 1> bufferData_3(data_3, numOfItems);
    buffer<T, 1> bufferData_4(data_4, numOfItems);
    buffer<T, 1> bufferData_5(data_5, numOfItems);
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      accessor<T, 1, access::mode::read, access::target::device,
               access::placeholder::false_t>
          accessorData_1(bufferData_1, cgh);
      accessor<T, 1, access::mode::read_write, access::target::device,
               access::placeholder::false_t>
          accessorData_2(bufferData_2, cgh);
      accessor<T, 1, access::mode::read_write, access::target::device,
               access::placeholder::false_t>
          accessorData_3(bufferData_3, cgh);
      accessor<T, 1, access::mode::read_write, access::target::device,
               access::placeholder::false_t>
          accessorData_4(bufferData_4, cgh);
      accessor<T, 1, access::mode::read_write, access::target::device,
               access::placeholder::false_t>
          accessorData_5(bufferData_5, cgh);
      local_accessor<T, 1> localAccessor(numOfItems, cgh);

      cgh.parallel_for<class testMultPtrKernel<
          T, IsDecorated>>(range<1>{10}, [=](id<1> wiID) {
        T private_data[10];
        for (size_t i = 0; i < 10; ++i)
          private_data[i] = 0;
        localAccessor[wiID] = 0;

        auto ptr_1 =
            multi_ptr<T, access::address_space::global_space, IsDecorated>(
                accessorData_1);
        auto ptr_2 =
            multi_ptr<T, access::address_space::global_space, IsDecorated>(
                accessorData_2);
        auto ptr_3 =
            multi_ptr<T, access::address_space::global_space, IsDecorated>(
                accessorData_3);
        auto ptr_4 =
            multi_ptr<T, access::address_space::global_space, IsDecorated>(
                accessorData_4);
        auto ptr_5 =
            multi_ptr<T, access::address_space::global_space, IsDecorated>(
                accessorData_5);
        auto local_ptr =
            multi_ptr<T, access::address_space::local_space, IsDecorated>(
                localAccessor);
        auto priv_ptr = address_space_cast<access::address_space::private_space,
                                           IsDecorated>(private_data);
        static_assert(
            std::is_same_v<private_ptr<T, IsDecorated>, decltype(priv_ptr)>,
            "Incorrect type for priv_ptr.");

        // Construct extension pointer from accessors.
        auto dev_ptr =
            multi_ptr<T, access::address_space::ext_intel_global_device_space,
                      IsDecorated>(accessorData_1);
        static_assert(std::is_same_v<ext::intel::device_ptr<T, IsDecorated>,
                                     decltype(dev_ptr)>,
                      "Incorrect type for dev_ptr.");

        // General conversions in multi_ptr class
        T *RawPtr = nullptr;
        global_ptr<T, IsDecorated> ptr_6 =
            address_space_cast<access::address_space::global_space,
                               IsDecorated>(RawPtr);

        global_ptr<T, IsDecorated> ptr_7(accessorData_1);

        global_ptr<void, IsDecorated> ptr_8 =
            address_space_cast<access::address_space::global_space,
                               IsDecorated>((void *)RawPtr);

        // Explicit conversions for device_ptr/host_ptr to global_ptr
        ext::intel::device_ptr<void, IsDecorated> ptr_9 = address_space_cast<
            access::address_space::ext_intel_global_device_space, IsDecorated>(
            (void *)RawPtr);
        global_ptr<void, IsDecorated> ptr_10 =
            global_ptr<void, IsDecorated>(ptr_9);
        ext::intel::host_ptr<void, IsDecorated> ptr_11 = address_space_cast<
            access::address_space::ext_intel_global_host_space, IsDecorated>(
            (void *)RawPtr);
        global_ptr<void, IsDecorated> ptr_12 =
            global_ptr<void, IsDecorated>(ptr_11);

        innerFunc<T, IsDecorated>(wiID.get(0), ptr_1, ptr_2, ptr_3, ptr_4,
                                  ptr_5, local_ptr, priv_ptr);
      });
    });
  }
  for (size_t i = 0; i < 10; ++i) {
    assert(data_1[i] == 1 && "Expected data_1[i] == 1");
    assert(data_2[i] == 1 && "Expected data_2[i] == 1");
    assert(data_3[i] == 1 && "Expected data_3[i] == 1");
    assert(data_4[i] == 1 && "Expected data_4[i] == 1");
    assert(data_5[i] == 1 && "Expected data_5[i] == 1");
  }
}

template <typename T, access::decorated IsDecorated>
void testMultPtrArrowOperator() {
  point<T> data_1[1] = {1};
  point<T> data_2[1] = {2};
  point<T> data_3[1] = {3};

  {
    range<1> numOfItems{1};
    buffer<point<T>, 1> bufferData_1(data_1, numOfItems);
    buffer<point<T>, 1> bufferData_2(data_2, numOfItems);
    buffer<point<T>, 1> bufferData_3(data_3, numOfItems);
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      accessor<point<T>, 1, access::mode::read, access::target::device,
               access::placeholder::false_t>
          accessorData_1(bufferData_1, cgh);
      local_accessor<point<T>, 1> accessorData_2(1, cgh);
      accessor<point<T>, 1, access::mode::read, access::target::device,
               access::placeholder::false_t>
          accessorData_3(bufferData_3, cgh);

      cgh.single_task<class testMultPtrArrowOperatorKernel<T, IsDecorated>>(
          [=]() {
            point<T> private_val = 0;

            auto ptr_1 =
                multi_ptr<point<T>, access::address_space::global_space,
                          IsDecorated>(accessorData_1);
            auto ptr_2 = multi_ptr<point<T>, access::address_space::local_space,
                                   IsDecorated>(accessorData_2);
            auto ptr_3 =
                multi_ptr<point<T>,
                          access::address_space::ext_intel_global_device_space,
                          IsDecorated>(accessorData_3);
            auto ptr_4 =
                address_space_cast<access::address_space::private_space,
                                   IsDecorated>(&private_val);
            static_assert(std::is_same_v<private_ptr<point<T>, IsDecorated>,
                                         decltype(ptr_4)>,
                          "Incorrect type for ptr_4.");

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

template <access::decorated IsDecorated> void runTestsForDecoration() {
  testMultPtr<int, IsDecorated>();
  testMultPtr<float, IsDecorated>();
  testMultPtr<point<int>, IsDecorated>();
  testMultPtr<point<float>, IsDecorated>();

  testMultPtrArrowOperator<int, IsDecorated>();
  testMultPtrArrowOperator<float, IsDecorated>();
}

int main() {
  runTestsForDecoration<access::decorated::yes>();
  runTestsForDecoration<access::decorated::no>();
  return 0;
}
