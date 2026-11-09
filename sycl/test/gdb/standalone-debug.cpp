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
#include <sycl/exception.hpp>
#include <sycl/handler.hpp>
#include <sycl/queue.hpp>

void use(sycl::device &, sycl::exception &, sycl::handler &, sycl::queue &) {}

// Only a definition is `distinct` and has an `elements:` list; a forward
// declaration has neither. (std::exception shares the name "exception" but is
// a forward declaration here, so it cannot match.)
// CHECK-DAG: distinct !DICompositeType(tag: DW_TAG_class_type, name: "device",{{.*}} elements:
// CHECK-DAG: distinct !DICompositeType(tag: DW_TAG_class_type, name: "exception",{{.*}} elements:
// CHECK-DAG: distinct !DICompositeType(tag: DW_TAG_class_type, name: "handler",{{.*}} elements:
// CHECK-DAG: distinct !DICompositeType(tag: DW_TAG_class_type, name: "queue",{{.*}} elements:
