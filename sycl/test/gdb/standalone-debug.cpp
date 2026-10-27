// RUN: %clangxx %fsycl-host-only -g -S -emit-llvm -o - %s | FileCheck %s
//
// UNSUPPORTED: windows

// The gdb pretty printers (sycl/gdb/libsycl.so-gdb.py) read data members of
// these classes, so a TU that merely references them must still carry their
// full definition. Constructor homing would otherwise leave only a forward
// declaration, because the constructors that get emitted in a user's TU
// delegate into libsycl, which ships without debug info. The classes opt out
// via __SYCL_STANDALONE_DEBUG.

#include <sycl/device.hpp>
#include <sycl/queue.hpp>

void use(sycl::device &, sycl::queue &) {}

// Only a definition is `distinct` and has an `elements:` list; a forward
// declaration has neither.
// CHECK-DAG: distinct !DICompositeType(tag: DW_TAG_class_type, name: "device",{{.*}} elements: ![[#]], identifier: "_ZTSN4sycl3_V16deviceE")
// CHECK-DAG: distinct !DICompositeType(tag: DW_TAG_class_type, name: "queue",{{.*}} elements: ![[#]], identifier: "_ZTSN4sycl3_V15queueE")
