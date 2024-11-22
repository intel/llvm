// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -fsycl-int-footer=%t.footer.h -emit-llvm %s -o -
// RUN: FileCheck -input-file=%t.footer.h %s --check-prefix=CHECK-FOOTER

#include "sycl.hpp"

namespace sycl {
template <typename T> struct X {};
template <> struct X<int> {};
namespace detail {
struct Y {};
} // namespace detail
template <> struct X<detail::Y> {};
} // namespace sycl

using namespace sycl;
template <typename T, typename = X<detail::Y>> struct Arg1 { T val; };

using namespace sycl::ext::oneapi;
template <typename properties_t>
device_global<properties_t> dev_global;

SYCL_EXTERNAL auto foo() {
  (void)dev_global<Arg1<int>>;
}

// CHECK-FOOTER: __sycl_device_global_registration::__sycl_device_global_registration() noexcept {
// CHECK-FOOTER-NEXT: device_global_map::add((void *)&::dev_global<Arg1<int, sycl::X<sycl::detail::Y>>>, "_Z10dev_globalI4Arg1IiN4sycl1XINS1_6detail1YEEEEE");
// CHECK-FOOTER-NEXT: }
// CHECK-FOOTER-NEXT: } // namespace (unnamed)
