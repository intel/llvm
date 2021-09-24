// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-vtable-layouts %s | FileCheck %s
// REQUIRES: linux

#include <CL/sycl.hpp>

// clang-format off

// Changing vtable breaks ABI. If this test fails, please, refer to ABI Policy
// Guide for further instructions.

void foo(sycl::detail::HostKernelBase &HKB) {
  sycl::detail::NDRDescT Desc;
  sycl::detail::HostProfilingInfo HPInfo;
  HKB.call(Desc, &HPInfo);
}

// CHECK:      Vtable for '{{.*}}::detail::HostKernelBase' (6 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | {{.*}}::detail::HostKernelBase RTTI
// CHECK-NEXT:       -- ({{.*}}::detail::HostKernelBase, 0) vtable address --
// CHECK-NEXT:   2 | void {{.*}}::detail::HostKernelBase::call(const {{.*}}::detail::NDRDescT &, {{.*}}::detail::HostProfilingInfo *) [pure]
// CHECK-NEXT:   3 | char *{{.*}}::detail::HostKernelBase::getPtr() [pure]
// CHECK-NEXT:   4 | {{.*}}::detail::HostKernelBase::~HostKernelBase() [complete]
// CHECK-NEXT:   5 | {{.*}}::detail::HostKernelBase::~HostKernelBase() [deleting]

void foo(sycl::detail::SYCLMemObjI &MemObj) { (void)MemObj.getType(); }

// CHECK:    Vtable for '{{.*}}::detail::SYCLMemObjI' (11 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | {{.*}}::detail::SYCLMemObjI RTTI
// CHECK-NEXT:       -- ({{.*}}::detail::SYCLMemObjI, 0) vtable address --
// CHECK-NEXT:   2 | {{.*}}::detail::SYCLMemObjI::~SYCLMemObjI() [complete]
// CHECK-NEXT:   3 | {{.*}}::detail::SYCLMemObjI::~SYCLMemObjI() [deleting]
// CHECK-NEXT:   4 | {{.*}}::detail::SYCLMemObjI::MemObjType {{.*}}::detail::SYCLMemObjI::getType() const [pure]
// CHECK-NEXT:   5 | void *{{.*}}::detail::SYCLMemObjI::allocateMem({{.*}}::detail::ContextImplPtr, bool, void *, RT::PiEvent &) [pure]
// CHECK-NEXT:   6 | void *{{.*}}::detail::SYCLMemObjI::allocateHostMem() [pure]
// CHECK-NEXT:   7 | void {{.*}}::detail::SYCLMemObjI::releaseMem({{.*}}::detail::ContextImplPtr, void *) [pure]
// CHECK-NEXT:   8 | void {{.*}}::detail::SYCLMemObjI::releaseHostMem(void *) [pure]
// CHECK-NEXT:   9 | size_t {{.*}}::detail::SYCLMemObjI::getSize() const [pure]
// CHECK-NEXT:  10 | {{.*}}::detail::ContextImplPtr {{.*}}::detail::SYCLMemObjI::getInteropContext() const [pure]

void foo(sycl::detail::pi::DeviceBinaryImage &Img) { Img.print(); }
// CHECK:    Vtable for '{{.*}}::detail::pi::DeviceBinaryImage' (6 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | {{.*}}::detail::pi::DeviceBinaryImage RTTI
// CHECK-NEXT:       -- ({{.*}}::detail::pi::DeviceBinaryImage, 0) vtable address --
// CHECK-NEXT:   2 | void {{.*}}::detail::pi::DeviceBinaryImage::print() const
// CHECK-NEXT:   3 | void {{.*}}::detail::pi::DeviceBinaryImage::dump(std::ostream &) const
// CHECK-NEXT:   4 | {{.*}}::detail::pi::DeviceBinaryImage::~DeviceBinaryImage() [complete]
// CHECK-NEXT:   5 | {{.*}}::detail::pi::DeviceBinaryImage::~DeviceBinaryImage() [deleting]

void foo(sycl::detail::CG *CG) { delete CG; }
// CHECK:    Vtable for '{{.*}}::detail::CG' (4 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | {{.*}}::detail::CG RTTI
// CHECK-NEXT:       -- ({{.*}}::detail::CG, 0) vtable address --
// CHECK-NEXT:   2 | {{.*}}::detail::CG::~CG() [complete]
// CHECK-NEXT:   3 | {{.*}}::detail::CG::~CG() [deleting]

void foo(sycl::detail::PropertyWithDataBase *Prop) { delete Prop; }
// CHECK:    Vtable for '{{.*}}::detail::PropertyWithDataBase' (4 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | {{.*}}::detail::PropertyWithDataBase RTTI
// CHECK-NEXT:       -- ({{.*}}::detail::PropertyWithDataBase, 0) vtable address --
// CHECK-NEXT:   2 | {{.*}}::detail::PropertyWithDataBase::~PropertyWithDataBase() [complete]
// CHECK-NEXT:   3 | {{.*}}::detail::PropertyWithDataBase::~PropertyWithDataBase() [deleting]

void foo(sycl::detail::SYCLMemObjAllocator &Allocator) {
  (void)Allocator.allocate(0);
}
// CHECK:    Vtable for '{{.*}}::detail::SYCLMemObjAllocator' (9 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | {{.*}}::detail::SYCLMemObjAllocator RTTI
// CHECK-NEXT:       -- ({{.*}}::detail::SYCLMemObjAllocator, 0) vtable address --
// CHECK-NEXT:   2 | void *{{.*}}::detail::SYCLMemObjAllocator::getAllocatorImpl() [pure]
// CHECK-NEXT:   3 | {{.*}}::detail::SYCLMemObjAllocator::~SYCLMemObjAllocator() [complete]
// CHECK-NEXT:   4 | {{.*}}::detail::SYCLMemObjAllocator::~SYCLMemObjAllocator() [deleting]
// CHECK-NEXT:   5 | void *{{.*}}::detail::SYCLMemObjAllocator::allocate(std::size_t) [pure]
// CHECK-NEXT:   6 | void {{.*}}::detail::SYCLMemObjAllocator::deallocate(void *, std::size_t) [pure]
// CHECK-NEXT:   7 | std::size_t {{.*}}::detail::SYCLMemObjAllocator::getValueSize() const [pure]
// CHECK-NEXT:   8 | void {{.*}}::detail::SYCLMemObjAllocator::setAlignment(std::size_t) [pure]

void foo(sycl::device_selector &DeviceSelector) {
  (void)DeviceSelector.select_device();
}
// CHECK:    Vtable for '{{.*}}::device_selector' (6 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | {{.*}}::device_selector RTTI
// CHECK-NEXT:       -- ({{.*}}::device_selector, 0) vtable address --
// CHECK-NEXT:   2 | {{.*}}::device_selector::~device_selector() [complete]
// CHECK-NEXT:   3 | {{.*}}::device_selector::~device_selector() [deleting]
// CHECK-NEXT:   4 | {{.*}}::device {{.*}}::device_selector::select_device() const
// CHECK-NEXT:   5 | int {{.*}}::device_selector::operator()(const {{.*}}::device &) const [pure]
