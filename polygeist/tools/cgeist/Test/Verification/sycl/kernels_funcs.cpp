// Copyright (C) Intel

//===--- kernels_funcs.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: sycl-clang.py %s -S | FileCheck %s
// XFAIL: *


#include <sycl/sycl.hpp>

// CHECK-NOT: module

// CHECK: gpu.module @device_functions
//
// CHECK-DAG: gpu.func @_ZTS8kernel_1
// CHECK-SAME: kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]}
// CHECK-DAG: gpu.func @_ZTSZZ6host_2vENKUlRN4sycl3_V17handlerEE_clES2_E8kernel_2
// CHECK-SAME: kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]}
// CHECK-DAG: func.func @_ZN12StoreWrapperIiLi1ELN4sycl3_V16access4modeE1026EEC1ENS1_8accessorIiLi1ELS3_1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS1_3ext6oneapi22accessor_property_listIJEEEEENS1_2idILi1EEERKi
// CHECK-SAME: attributes {llvm.linkage = #llvm.linkage<linkonce_odr>}
// COM: StoreWrapper constructor:
// CHECK-DAG: func.func @_ZN12StoreWrapperIiLi1ELN4sycl3_V16access4modeE1026EEclEv
// CHECK-SAME: attributes {llvm.linkage = #llvm.linkage<linkonce_odr>}

template <typename DataT,
          int Dimensions = 1,
          sycl::access::mode AccessMode = (std::is_const_v<DataT>
                                           ? sycl::access_mode::read
                                           : sycl::access_mode::read_write)>
class StoreWrapper {
public:
  StoreWrapper(sycl::accessor<DataT, Dimensions, AccessMode> acc,
               sycl::id<Dimensions> index,
               const DataT& el)
    : acc{acc}, index{index}, el{el} {}

  void operator()() {
    acc[index] = el;
  }

private:
  sycl::accessor<DataT, Dimensions, AccessMode> acc;
  sycl::id<Dimensions> index;
  DataT el;
};

class kernel_1 {
  sycl::accessor<sycl::cl_int, 1, sycl::access::mode::read_write> A;

public:
  kernel_1(sycl::accessor<sycl::cl_int, 1, sycl::access::mode::read_write> A)
    : A(A) {}

  void operator()(sycl::id<1> id) const {
    StoreWrapper W{A, id, 42};
    W();
  }
};

void host_1() {
  auto q = sycl::queue{};
  auto range = sycl::range<1>{1};

  {
    auto buf = sycl::buffer<int, 1>{nullptr, range};
    q.submit([&](sycl::handler &cgh) {
	       auto A = buf.get_access<sycl::access::mode::read_write>(cgh);
	       auto ker =  kernel_1{A};
	       cgh.parallel_for<kernel_1>(range, ker);
	     });
  }
}

void host_2() {
  auto q = sycl::queue{};
  auto range = sycl::range<1>{1};

  {
    auto buf = sycl::buffer<int, 1>{nullptr, range};
    q.submit([&](sycl::handler &cgh) {
	       auto A = buf.get_access<sycl::access::mode::read_write>(cgh);
	       cgh.parallel_for<class kernel_2>(range, [=](sycl::id<1> id) {
							 A[id] = 42;
						       });
	     });
  }
}

SYCL_EXTERNAL void function_1(sycl::item<2, true> item) {
  auto id = item.get_id(0);
}
