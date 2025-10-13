// RUN: env CPLUS_INCLUDE_PATH=%test_include_path \
// RUN: %clangxx %s -fsyntax-only -Xclang -fdump-record-layouts \
// RUN:          -Xclang -fsycl-is-host \
// RUN:   | FileCheck --check-prefixes=CHECK,HOST %s
// RUN: env CPLUS_INCLUDE_PATH=%test_include_path \
// RUN: %clangxx %s -fsyntax-only -Xclang -fdump-record-layouts \
// RUN:          -fsycl-device-only \
// RUN:   | FileCheck --check-prefixes=CHECK,DEVICE %s
//
// UNSUPPORTED: windows

// The record layouts below correspond to use cases in the GDB pretty printers:
//   * sycl/gdb/libsycl.so-gdb.py
// Changes here should be reflected there, and vice versa.

#include <sycl/accessor.hpp>
#include <sycl/buffer.hpp>
#include <sycl/device.hpp>
#include <sycl/id.hpp>
#include <sycl/item.hpp>
#include <sycl/queue.hpp>
#include <sycl/range.hpp>

#include "detail/accessor_impl.hpp"
#include "detail/buffer_impl.hpp"
#include "detail/device_impl.hpp"
#include "detail/helpers.hpp"
#include "detail/queue_impl.hpp"
#include "detail/sycl_mem_obj_t.hpp"

sycl::accessor<int> a;
sycl::buffer<int> b(1);
sycl::device d;
sycl::local_accessor<int> l;
sycl::queue q;
sycl::range<1> r(3);

sycl::item<1, true> item_w_offset =
    sycl::detail::Builder::createItem<1, true>({4}, {2}, {1});
sycl::item<2, false> item_wo_offset =
    sycl::detail::Builder::createItem<2, false>({4, 4}, {2, 2});

// CHECK:         0 | class sycl::range<>
// CHECK:         0 |     size_t[1] common_array

// CHECK:         0 | class sycl::device
// CHECK:         0 |   class std::shared_ptr<class sycl::detail::device_impl> impl
// CHECK:         0 |       element_type * _M_ptr

// CHECK:         0 | class sycl::queue
// CHECK:         0 |   class std::shared_ptr<class sycl::detail::queue_impl> impl
// CHECK:         0 |       element_type * _M_ptr

// HOST:          0 | class sycl::detail::AccessorImplHost
// HOST:          0 |   struct sycl::detail::AccHostDataT MAccData
// HOST:          0 |     class sycl::id<3> MOffset
// HOST:         24 |     class sycl::range<3> MAccessRange
// HOST:         48 |     class sycl::range<3> MMemoryRange
// HOST:        120 |   detail::SYCLMemObjI * MSYCLMemObj

// HOST:          0 | class sycl::detail::SYCLMemObjT
// HOST:        112 |   size_t MSizeInBytes
// HOST:        120 |   void * MUserPtr

// CHECK:         0 | class sycl::detail::buffer_impl
// CHECK:       120 |     void * MUserPtr

// CHECK:         0 | class sycl::detail::platform_impl
// CHECK:        32 |   backend MBackend

// CHECK:         0 | class sycl::detail::device_impl
// CHECK:        24 |   platform_impl & MPlatform

// DEVICE:        0 | class sycl::detail::AccessorImplDevice<1>
// DEVICE:        0 |   class sycl::id<> Offset
// DEVICE:        8 |   class sycl::range<> AccessRange
// DEVICE:       16 |   class sycl::range<> MemRange

// CHECK:         0 | class sycl::detail::queue_impl
// CHECK:        56 |   device_impl & MDevice

// CHECK:         0 | class sycl::accessor<int>
// HOST:          0 |     {{.*}} sycl::detail::AccessorImplHost{{.*}} impl
// HOST:         16 |   detail::AccHostDataT * MAccData
// DEVICE:        0 |   class sycl::detail::AccessorImplDevice<1> impl
// DEVICE:       24 |   union sycl::accessor<int>::(anonymous {{.*}})
// DEVICE-NEXT:  24 |     ConcreteASPtrType MData

// CHECK:         0 | class sycl::buffer<int>
// CHECK:         0 |     class std::shared_ptr<class sycl::detail::buffer_impl> impl
// CHECK:         0 |         element_type * _M_ptr
// CHECK:        16 |   class sycl::range<> Range
// CHECK:        16 |       size_t[1] common_array

// CHECK:         0 | class sycl::local_accessor<int>
// HOST-NOT:        |     ConcreteASPtrType MData
// HOST:            | [sizeof={{.*}}
// DEVICE:        0 |     class sycl::detail::LocalAccessorBaseDevice<1> impl
// DEVICE:        8 |       class sycl::range<> MemRange
// DEVICE:        8 |           size_t[1] common_array
// DEVICE:       24 |     ConcreteASPtrType MData

// CHECK:         0 | class sycl::item<1, true>
// CHECK:         0 |   struct sycl::detail::ItemBase<1, true> MImpl
// CHECK:         0 |     class sycl::range<> MExtent
// CHECK:         0 |       class sycl::detail::array<> (base)
// CHECK:         0 |         size_t[1] common_array
// CHECK:         8 |     class sycl::id<> MIndex
// CHECK:         8 |       class sycl::detail::array<> (base)
// CHECK:         8 |         size_t[1] common_array
// CHECK:        16 |     class sycl::id<> MOffset
// CHECK:        16 |       class sycl::detail::array<> (base)
// CHECK:        16 |         size_t[1] common_array

// CHECK:         0 | class sycl::item<2, false>
// CHECK:         0 |   struct sycl::detail::ItemBase<2, false> MImpl
// CHECK:         0 |     class sycl::range<2> MExtent
// CHECK:         0 |       class sycl::detail::array<2> (base)
// CHECK:         0 |         size_t[2] common_array
// CHECK:        16 |     class sycl::id<2> MIndex
// CHECK:        16 |       class sycl::detail::array<2> (base)
// CHECK:        16 |         size_t[2] common_array
// CHECK-NOT:    32 |     class sycl::id<2> MOffset
// CHECK-NOT:    32 |       class sycl::detail::array<2> (base)
// CHECK-NOT:    32 |         size_t[2] common_array
