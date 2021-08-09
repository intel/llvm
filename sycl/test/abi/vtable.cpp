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

// CHECK:      Vtable for 'sycl::detail::HostKernelBase' (6 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | sycl::detail::HostKernelBase RTTI
// CHECK-NEXT:       -- (sycl::detail::HostKernelBase, 0) vtable address --
// CHECK-NEXT:   2 | void sycl::detail::HostKernelBase::call(const sycl::detail::NDRDescT &, sycl::detail::HostProfilingInfo *) [pure]
// CHECK-NEXT:   3 | char *sycl::detail::HostKernelBase::getPtr() [pure]
// CHECK-NEXT:   4 | sycl::detail::HostKernelBase::~HostKernelBase() [complete]
// CHECK-NEXT:   5 | sycl::detail::HostKernelBase::~HostKernelBase() [deleting]

void foo(sycl::detail::SYCLMemObjI &MemObj) { (void)MemObj.getType(); }

// CHECK:    Vtable for 'sycl::detail::SYCLMemObjI' (11 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | sycl::detail::SYCLMemObjI RTTI
// CHECK-NEXT:       -- (sycl::detail::SYCLMemObjI, 0) vtable address --
// CHECK-NEXT:   2 | sycl::detail::SYCLMemObjI::~SYCLMemObjI() [complete]
// CHECK-NEXT:   3 | sycl::detail::SYCLMemObjI::~SYCLMemObjI() [deleting]
// CHECK-NEXT:   4 | sycl::detail::SYCLMemObjI::MemObjType sycl::detail::SYCLMemObjI::getType() const [pure]
// CHECK-NEXT:   5 | void *sycl::detail::SYCLMemObjI::allocateMem(sycl::detail::ContextImplPtr, bool, void *, RT::PiEvent &) [pure]
// CHECK-NEXT:   6 | void *sycl::detail::SYCLMemObjI::allocateHostMem() [pure]
// CHECK-NEXT:   7 | void sycl::detail::SYCLMemObjI::releaseMem(sycl::detail::ContextImplPtr, void *) [pure]
// CHECK-NEXT:   8 | void sycl::detail::SYCLMemObjI::releaseHostMem(void *) [pure]
// CHECK-NEXT:   9 | size_t sycl::detail::SYCLMemObjI::getSize() const [pure]
// CHECK-NEXT:  10 | sycl::detail::ContextImplPtr sycl::detail::SYCLMemObjI::getInteropContext() const [pure]

void foo(sycl::detail::pi::DeviceBinaryImage &Img) { Img.print(); }
// CHECK:    Vtable for 'sycl::detail::pi::DeviceBinaryImage' (6 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | sycl::detail::pi::DeviceBinaryImage RTTI
// CHECK-NEXT:       -- (sycl::detail::pi::DeviceBinaryImage, 0) vtable address --
// CHECK-NEXT:   2 | void sycl::detail::pi::DeviceBinaryImage::print() const
// CHECK-NEXT:   3 | void sycl::detail::pi::DeviceBinaryImage::dump(std::ostream &) const
// CHECK-NEXT:   4 | sycl::detail::pi::DeviceBinaryImage::~DeviceBinaryImage() [complete]
// CHECK-NEXT:   5 | sycl::detail::pi::DeviceBinaryImage::~DeviceBinaryImage() [deleting]

void foo(sycl::detail::CG *CG) { delete CG; }
// CHECK:    Vtable for 'sycl::detail::CG' (4 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | sycl::detail::CG RTTI
// CHECK-NEXT:       -- (sycl::detail::CG, 0) vtable address --
// CHECK-NEXT:   2 | sycl::detail::CG::~CG() [complete]
// CHECK-NEXT:   3 | sycl::detail::CG::~CG() [deleting]

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
// CHECK-NEXT:   4 | sycl::device sycl::device_selector::select_device() const
// CHECK-NEXT:   5 | int sycl::device_selector::operator()(const sycl::device &) const [pure]
