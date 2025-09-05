// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -internal-isystem %S/Inputs -triple spir64-unknown-unknown -fsycl-int-footer=%t.footer.h -fsycl-int-header=%t.header.h -fsycl-unique-prefix=THE_PREFIX %s -emit-llvm -o %t.ll
// RUN: FileCheck -input-file=%t.footer.h %s --check-prefix=CHECK-FOOTER
// RUN: FileCheck -input-file=%t.header.h %s --check-prefix=CHECK-HEADER

// This test checks that integration header and footer are emitted correctly
// for device_global variables.

#include "sycl.hpp"

using namespace sycl::ext::oneapi;

int main() {
  sycl::kernel_single_task<class first_kernel>([]() {});
}

// CHECK-HEADER: namespace sycl {
// CHECK-HEADER-NEXT: inline namespace _V1 {
// CHECK-HEADER-NEXT: namespace detail {
// CHECK-HEADER: #ifdef __clang__
// CHECK-HEADER-NEXT: #pragma clang diagnostic push
// CHECK-HEADER-NEXT: #pragma clang diagnostic ignored "-Wreserved-identifier"
// CHECK-HEADER-NEXT: #endif // defined(__clang__)
// CHECK-HEADER: namespace {
// CHECK-HEADER-NEXT: class __sycl_device_global_registration {
// CHECK-HEADER-NEXT: public:
// CHECK-HEADER-NEXT:   __sycl_device_global_registration() noexcept;
// CHECK-HEADER-NEXT: };
// CHECK-HEADER-NEXT: __sycl_device_global_registration __sycl_device_global_registrar;
// CHECK-HEADER-NEXT: } // namespace
// CHECK-HEADER: #ifdef __clang__
// CHECK-HEADER-NEXT: #pragma clang diagnostic pop
// CHECK-HEADER-NEXT: #endif // defined(__clang__)
// CHECK-HEADER: } // namespace detail
// CHECK-HEADER: } // namespace _V1
// CHECK-HEADER: } // namespace sycl

// CHECK-FOOTER: #include <sycl/detail/defines_elementary.hpp>

// Shims go before the registration.
// CHECK-FOOTER: namespace Foo {
// CHECK-FOOTER-NEXT: namespace {
// CHECK-FOOTER-NEXT: namespace __sycl_detail {
// CHECK-FOOTER-NEXT: static constexpr decltype(AnonNS) &__shim_[[SHIM0:[0-9]+]]() {
// CHECK-FOOTER-NEXT:   return AnonNS;
// CHECK-FOOTER-NEXT: }
// CHECK-FOOTER-NEXT: } // namespace __sycl_detail
// CHECK-FOOTER-NEXT: } // namespace
// CHECK-FOOTER-NEXT: } // namespace Foo
// CHECK-FOOTER-NEXT: namespace {
// CHECK-FOOTER-NEXT: namespace __sycl_detail {
// CHECK-FOOTER-NEXT: static constexpr decltype(HasVarTemplate::VarTempl<int>) &__shim_[[SHIM1:[0-9]+]]() {
// CHECK-FOOTER-NEXT:   return HasVarTemplate::VarTempl<int>;
// CHECK-FOOTER-NEXT: }
// CHECK-FOOTER-NEXT: } // namespace __sycl_detail
// CHECK-FOOTER-NEXT: } // namespace

// CHECK-FOOTER: #include <sycl/detail/device_global_map.hpp>
// CHECK-FOOTER: namespace sycl::detail {
// CHECK-FOOTER: #ifdef __clang__
// CHECK-FOOTER-NEXT: #pragma clang diagnostic push
// CHECK-FOOTER-NEXT: #pragma clang diagnostic ignored "-Wold-style-cast"
// CHECK-FOOTER-NEXT: #endif // defined(__clang__)
// CHECK-FOOTER: namespace {
// CHECK-FOOTER-NEXT: __sycl_device_global_registration::__sycl_device_global_registration() noexcept {

extern device_global<int> Basic;
device_global<int> Basic;
// CHECK-FOOTER-NEXT: device_global_map::add((void *)&::Basic, "_Z5Basic");
// CHECK-FOOTER-NOT: Basic

extern device_global<int> ExternDevGlobal;
// CHECK-FOOTER-NOT: ExternDevGlobal

struct Wrapper {
  static device_global<int> WrapperDevGlobal;
};
// CHECK-FOOTER-NEXT: device_global_map::add((void *)&::Wrapper::WrapperDevGlobal, "_ZN7Wrapper16WrapperDevGlobalE");

template <typename T>
struct WrapperTemplate {
  static device_global<T> WrapperDevGlobal;
};
template class WrapperTemplate<int>;
// CHECK-FOOTER-NEXT: device_global_map::add((void *)&::WrapperTemplate<int>::WrapperDevGlobal, "_ZN15WrapperTemplateIiE16WrapperDevGlobalE");

namespace Foo {
device_global<int> NS;
// CHECK-FOOTER-NEXT: device_global_map::add((void *)&::Foo::NS, "_ZN3Foo2NSE");

inline namespace Bar {
device_global<float> InlineNS;
// CHECK-FOOTER-NEXT: device_global_map::add((void *)&::Foo::InlineNS, "_ZN3Foo3Bar8InlineNSE");

struct Wrapper {
  static device_global<int> WrapperDevGlobal;
};
// CHECK-FOOTER-NEXT: device_global_map::add((void *)&::Foo::Wrapper::WrapperDevGlobal, "_ZN3Foo3Bar7Wrapper16WrapperDevGlobalE");

template <typename T>
struct WrapperTemplate {
  static device_global<T> WrapperDevGlobal;
};
template class WrapperTemplate<float>;
// CHECK-FOOTER-NEXT: device_global_map::add((void *)&::Foo::WrapperTemplate<float>::WrapperDevGlobal, "_ZN3Foo3Bar15WrapperTemplateIfE16WrapperDevGlobalE");
} // namespace Bar

namespace {
device_global<int> AnonNS;
}
// CHECK-FOOTER-NEXT: device_global_map::add((void *)&::Foo::__sycl_detail::__shim_[[SHIM0]](), "THE_PREFIX____ZN3Foo12_GLOBAL__N_16AnonNSE");

} // namespace Foo

// Validate that variable templates work correctly.
namespace {
struct HasVarTemplate {
  constexpr HasVarTemplate() {}
  template <typename T>
  static const device_global<T> VarTempl;
};

} // namespace
const auto x = HasVarTemplate::VarTempl<int>.get();
// CHECK-FOOTER-NEXT: device_global_map::add((void *)&::__sycl_detail::__shim_[[SHIM1]](), "THE_PREFIX____ZN12_GLOBAL__N_114HasVarTemplate8VarTemplIiEE");
// CHECK-FOOTER: #ifdef __clang__
// CHECK-FOOTER-NEXT: #pragma clang diagnostic pop
// CHECK-FOOTER-NEXT: #endif // defined(__clang__)
