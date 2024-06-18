//==------- multi_ptr_legacy.hpp - SYCL multi_ptr legacy test header -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/usm_pointers.hpp>

#include <cassert>
#include <iostream>
#include <type_traits>

using namespace sycl;

/* This is the class used to name the kernel for the runtime.
 * This must be done when the kernel is expressed as a lambda. */
template <typename T> class testMultPtrKernel;
template <typename T> class testMultPtrArrowOperatorKernel;

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

template <typename T>
void innerFunc(id<1> wiID, global_ptr<const T> ptr_1, global_ptr<T> ptr_2,
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
      accessor<T, 1, access::mode::read, access::target::device,
               access::placeholder::false_t>
          accessorData_1(bufferData_1, cgh);
      accessor<T, 1, access::mode::read_write, access::target::device,
               access::placeholder::false_t>
          accessorData_2(bufferData_2, cgh);
      local_accessor<T, 1> localAccessor(numOfItems, cgh);

      cgh.parallel_for<class testMultPtrKernel<T>>(
          nd_range<1>{10, 10}, [=](nd_item<1> wiID) {
            auto ptr_1 = make_ptr<const T, access::address_space::global_space,
                                  access::decorated::legacy>(
                accessorData_1
                    .template get_multi_ptr<sycl::access::decorated::legacy>());
            auto ptr_2 = make_ptr<T, access::address_space::global_space,
                                  access::decorated::legacy>(
                accessorData_2
                    .template get_multi_ptr<sycl::access::decorated::legacy>());
            auto local_ptr = make_ptr<T, access::address_space::local_space,
                                      access::decorated::legacy>(
                localAccessor.get_pointer());

            auto local_ptr2 =
                multi_ptr<T, access::address_space::local_space,
                          access::decorated::legacy>(localAccessor);

            auto local_ptr3 =
                multi_ptr<void, access::address_space::local_space,
                          access::decorated::legacy>(localAccessor);

            auto local_ptr4 =
                multi_ptr<const void, access::address_space::local_space,
                          access::decorated::legacy>(localAccessor);

            auto local_ptr5 =
                multi_ptr<T, access::address_space::generic_space,
                          access::decorated::legacy>(localAccessor);

            auto local_ptr6 =
                multi_ptr<void, access::address_space::generic_space,
                          access::decorated::legacy>(localAccessor);

            auto local_ptr7 =
                multi_ptr<const void, access::address_space::generic_space,
                          access::decorated::legacy>(localAccessor);

            // Construct extension pointer from accessors.
            auto dev_ptr =
                multi_ptr<const T,
                          access::address_space::ext_intel_global_device_space>(
                    accessorData_1);
            static_assert(std::is_same_v<ext::intel::device_ptr<const T>,
                                         decltype(dev_ptr)>,
                          "Incorrect type for dev_ptr.");

            // General conversions in multi_ptr class
            T *RawPtr = nullptr;
            global_ptr<T> ptr_4(RawPtr);
            ptr_4 = RawPtr;

            global_ptr<const T> ptr_5(accessorData_1);

            global_ptr<void> ptr_6((void *)RawPtr);

            ptr_6 = (void *)RawPtr;

            // Explicit conversions for device_ptr/host_ptr to global_ptr
            ext::intel::device_ptr<void> ptr_7((void *)RawPtr);
            global_ptr<void> ptr_8 = global_ptr<void>(ptr_7);
            ext::intel::host_ptr<void> ptr_9((void *)RawPtr);
            global_ptr<void> ptr_10 = global_ptr<void>(ptr_9);
            // TODO: need propagation of a7b763b26 patch to acl tool before
            // testing these conversions - otherwise the test would fail on
            // accelerator device during reversed translation from SPIR-V to
            // LLVM IR device_ptr<T> ptr_11(accessorData_1); global_ptr<T>
            // ptr_12 = global_ptr<T>(ptr_11);

            innerFunc<T>(wiID.get_local_id().get(0), ptr_1, ptr_2, local_ptr);
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

template <typename T> void testMultPtrArrowOperator() {
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
      accessor<point<T>, 1, access::mode::read, access::target::device,
               access::placeholder::false_t>
          accessorData_1(bufferData_1, cgh);
      accessor<point<T>, 1, access::mode::read, access::target::constant_buffer,
               access::placeholder::false_t>
          accessorData_2(bufferData_2, cgh);
      local_accessor<point<T>, 1> accessorData_3(1, cgh);
      accessor<point<T>, 1, access::mode::read, access::target::device,
               access::placeholder::false_t>
          accessorData_4(bufferData_4, cgh);

      cgh.parallel_for<class testMultPtrArrowOperatorKernel<T>>(
          sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
            auto ptr_1 =
                make_ptr<const point<T>, access::address_space::global_space,
                         access::decorated::legacy>(
                    accessorData_1.template get_multi_ptr<
                        sycl::access::decorated::legacy>());
            auto ptr_2 =
                make_ptr<point<T>, access::address_space::constant_space,
                         access::decorated::legacy>(
                    accessorData_2.get_pointer());
            auto ptr_3 = make_ptr<point<T>, access::address_space::local_space,
                                  access::decorated::legacy>(
                accessorData_3.get_pointer());
            auto ptr_4 =
                make_ptr<const point<T>,
                         access::address_space::ext_intel_global_device_space,
                         access::decorated::legacy>(
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
