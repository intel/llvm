// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-vtable-layouts %s -o %t.out | FileCheck %s
// REQUIRES: linux

#include <sycl/sycl.hpp>

// clang-format off

// Changing vtable breaks ABI. If this test fails, please, refer to ABI Policy
// Guide for further instructions.

void foo(sycl::detail::PropertyWithDataBase *Prop) { delete Prop; }
// CHECK:    Vtable for 'sycl::detail::PropertyWithDataBase' (4 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | sycl::detail::PropertyWithDataBase RTTI
// CHECK-NEXT:       -- (sycl::detail::PropertyWithDataBase, 0) vtable address --
// CHECK-NEXT:   2 | sycl::detail::PropertyWithDataBase::~PropertyWithDataBase() [complete]
// CHECK-NEXT:   3 | sycl::detail::PropertyWithDataBase::~PropertyWithDataBase() [deleting]

void foo(sycl::detail::SYCLMemObjAllocator &Allocator) {
  (void)Allocator.allocate(0);
}
// CHECK:    Vtable for 'sycl::detail::SYCLMemObjAllocator' (9 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | sycl::detail::SYCLMemObjAllocator RTTI
// CHECK-NEXT:       -- (sycl::detail::SYCLMemObjAllocator, 0) vtable address --
// CHECK-NEXT:   2 | void *sycl::detail::SYCLMemObjAllocator::getAllocatorImpl() [pure]
// CHECK-NEXT:   3 | sycl::detail::SYCLMemObjAllocator::~SYCLMemObjAllocator() [complete]
// CHECK-NEXT:   4 | sycl::detail::SYCLMemObjAllocator::~SYCLMemObjAllocator() [deleting]
// CHECK-NEXT:   5 | void *sycl::detail::SYCLMemObjAllocator::allocate(std::size_t) [pure]
// CHECK-NEXT:   6 | void sycl::detail::SYCLMemObjAllocator::deallocate(void *, std::size_t) [pure]
// CHECK-NEXT:   7 | std::size_t sycl::detail::SYCLMemObjAllocator::getValueSize() const [pure]
// CHECK-NEXT:   8 | void sycl::detail::SYCLMemObjAllocator::setAlignment(std::size_t) [pure]

void foo(sycl::device_selector &DeviceSelector) {
  (void)DeviceSelector.select_device();
}
// CHECK:    Vtable for 'sycl::device_selector' (6 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | sycl::device_selector RTTI
// CHECK-NEXT:       -- (sycl::device_selector, 0) vtable address --
// CHECK-NEXT:   2 | sycl::device_selector::~device_selector() [complete]
// CHECK-NEXT:   3 | sycl::device_selector::~device_selector() [deleting]
// CHECK-NEXT:   4 | device sycl::device_selector::select_device() const
// CHECK-NEXT:   5 | int sycl::device_selector::operator()(const device &) const [pure]
