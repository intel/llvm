// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -internal-isystem %S/Inputs -triple spir64-unknown-unknown -fsycl-int-footer=%t.footer.h -fsycl-int-header=%t.header.h %s -emit-llvm -o %t.ll
// RUN: FileCheck -input-file=%t.footer.h %s --check-prefix=CHECK-FOOTER
// RUN: FileCheck -input-file=%t.header.h %s --check-prefix=CHECK-HEADER

// This test checks that integration header and footer are emitted correctly
// for device_global variables even without kernels.

#include "sycl.hpp"

using namespace sycl::ext::oneapi;

// CHECK-HEADER: namespace sycl {
// CHECK-HEADER-NEXT: inline namespace _V1 {
// CHECK-HEADER-NEXT: namespace detail {
// CHECK-HEADER: namespace {
// CHECK-HEADER-NEXT: class __sycl_device_global_registration {
// CHECK-HEADER-NEXT: public:
// CHECK-HEADER-NEXT:   __sycl_device_global_registration() noexcept;
// CHECK-HEADER-NEXT: };
// CHECK-HEADER-NEXT: __sycl_device_global_registration __sycl_device_global_registrar;
// CHECK-HEADER-NEXT: } // namespace
// CHECK-HEADER: } // namespace detail
// CHECK-HEADER: } // namespace _V1
// CHECK-HEADER: } // namespace sycl

// CHECK-FOOTER: #include <sycl/detail/defines_elementary.hpp>

// CHECK-FOOTER: #include <sycl/detail/device_global_map.hpp>
// CHECK-FOOTER: namespace sycl::detail {
// CHECK-FOOTER: namespace {
// CHECK-FOOTER-NEXT: __sycl_device_global_registration::__sycl_device_global_registration() noexcept {

device_global<int> Basic;
// CHECK-FOOTER-NEXT: device_global_map::add((void *)&::Basic, "_Z5Basic");

// CHECK-FOOTER-NEXT: }
// CHECK-FOOTER-NEXT: }
// CHECK-FOOTER: }
