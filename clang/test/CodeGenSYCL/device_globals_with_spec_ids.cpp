// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -internal-isystem %S/Inputs -triple spir64-unknown-unknown -fsycl-int-footer=%t.footer.h -fsycl-int-header=%t.header.h %s -emit-llvm -o %t.ll
// RUN: FileCheck -input-file=%t.footer.h %s

// Try and compile all this stuff.
// RUN: %clang_cc1 -fsycl-is-host -x c++ -std=c++17 -internal-isystem %S/Inputs -fsyntax-only -include %t.header.h -include %s %t.footer.h

// This test checks that integration footer is emitted correctly if both
// spec constants and device globals are used.

#include "sycl.hpp"

using namespace cl;
int main() {
  cl::sycl::kernel_single_task<class first_kernel>([]() {});
}

// CHECK: #include <CL/sycl/detail/defines_elementary.hpp>
constexpr sycl::specialization_id a{2};
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::a>() {
// CHECK-NEXT: return "____ZL1a";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
sycl::ext::oneapi::device_global<int> b;

struct Wrapper {
  static constexpr sycl::specialization_id a{18};
  static sycl::ext::oneapi::device_global<float> b;
};
// CHECK: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::Wrapper::a>() {
// CHECK-NEXT:   return "_ZN7Wrapper1aE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)

template <typename T>
struct TemplateWrapper {
  static constexpr sycl::specialization_id<T> a{18};
  static sycl::ext::oneapi::device_global<T> b;
};

template class TemplateWrapper<float>;
// CHECK: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::TemplateWrapper<float>::a>() {
// CHECK-NEXT:   return "_ZN15TemplateWrapperIfE1aE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)

namespace {
constexpr sycl::specialization_id a{2};
sycl::ext::oneapi::device_global<int> b;
} // namespace

// CHECK: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(a) &__shim_[[SHIM0:[0-9]+]]() {
// CHECK-NEXT:   return a;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace 

// CHECK: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::__sycl_detail::__shim_[[SHIM0]]()>() {
// CHECK-NEXT:   return "____ZN12_GLOBAL__N_11aE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)

// CHECK: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(b) &__shim_[[SHIM1:[0-9]+]]() {
// CHECK-NEXT:   return b;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace 

namespace outer {
namespace {
namespace inner {
namespace {
constexpr sycl::specialization_id a{2};
// CHECK: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace inner {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(a) &__shim_[[SHIM2:[0-9]+]]() {
// CHECK-NEXT:   return a;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace 
// CHECK-NEXT: } // namespace inner
// CHECK-NEXT: } // namespace 
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(inner::__sycl_detail::__shim_[[SHIM2]]()) &__shim_[[SHIM3:[0-9]+]]() {
// CHECK-NEXT:   return inner::__sycl_detail::__shim_[[SHIM2]]();
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace 
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::outer::__sycl_detail::__shim_[[SHIM3]]()>() {
// CHECK-NEXT:   return "____ZN5outer12_GLOBAL__N_15inner12_GLOBAL__N_11aE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
sycl::ext::oneapi::device_global<int> b;
// CHECK: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace inner {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(b) &__shim_[[SHIM4:[0-9]+]]() {
// CHECK-NEXT:   return b;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace 
// CHECK-NEXT: } // namespace inner
// CHECK-NEXT: } // namespace 
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(inner::__sycl_detail::__shim_[[SHIM4]]()) &__shim_[[SHIM5:[0-9]+]]() {
// CHECK-NEXT:   return inner::__sycl_detail::__shim_[[SHIM4]]();
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace 
// CHECK-NEXT: } // namespace outer
struct Wrapper {
  static constexpr sycl::specialization_id a{18};
  static sycl::ext::oneapi::device_global<int> b;
  static sycl::ext::oneapi::device_global<float> c;
};
// CHECK: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace inner {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(Wrapper::a) &__shim_[[SHIM6:[0-9]+]]() {
// CHECK-NEXT:   return Wrapper::a;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace 
// CHECK-NEXT: } // namespace inner
// CHECK-NEXT: } // namespace 
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(inner::__sycl_detail::__shim_[[SHIM6]]()) &__shim_[[SHIM7:[0-9]+]]() {
// CHECK-NEXT:   return inner::__sycl_detail::__shim_[[SHIM6]]();
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace 
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::outer::__sycl_detail::__shim_[[SHIM7]]()>() {
// CHECK-NEXT:   return "____ZN5outer12_GLOBAL__N_15inner12_GLOBAL__N_17Wrapper1aE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)

// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace inner {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(Wrapper::b) &__shim_[[SHIM8:[0-9]+]]() {
// CHECK-NEXT:   return Wrapper::b;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace 
// CHECK-NEXT: } // namespace inner
// CHECK-NEXT: } // namespace 
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(inner::__sycl_detail::__shim_[[SHIM8]]()) &__shim_[[SHIM9:[0-9]+]]() {
// CHECK-NEXT:   return inner::__sycl_detail::__shim_[[SHIM8]]();
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace 
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace inner {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(Wrapper::c) &__shim_[[SHIM10:[0-9]+]]() {
// CHECK-NEXT:   return Wrapper::c;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace 
// CHECK-NEXT: } // namespace inner
// CHECK-NEXT: } // namespace 
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(inner::__sycl_detail::__shim_[[SHIM10]]()) &__shim_[[SHIM11:[0-9]+]]() {
// CHECK-NEXT:   return inner::__sycl_detail::__shim_[[SHIM10]]();
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace 
// CHECK-NEXT: } // namespace outer

// FIXME: Shims don't work with templated wrapper classes for some reason
// template <typename T>
// struct TemplateWrapper {
//   static constexpr sycl::specialization_id<T> a{18};
//   static sycl::ext::oneapi::device_global<T> b;
// };
// 
// template class TemplateWrapper<float>;

}
}
}
}

// CHECK: #include <CL/sycl/detail/spec_const_integration.hpp>
// CHECK-NEXT: #include <CL/sycl/detail/device_global_map.hpp>
// CHECK-NEXT: namespace sycl::detail {
// CHECK-NEXT: namespace {
// CHECK-NEXT: __sycl_device_global_registration::__sycl_device_global_registration() noexcept {
// CHECK-NEXT: device_global_map::add((void *)&::b, "_Z1b");
// CHECK-NEXT: device_global_map::add((void *)&::Wrapper::b, "_ZN7Wrapper1bE");
// CHECK-NEXT: device_global_map::add((void *)&::TemplateWrapper<float>::b, "_ZN15TemplateWrapperIfE1bE");
// CHECK-NEXT: device_global_map::add((void *)&::__sycl_detail::__shim_[[SHIM1]](), "____ZN12_GLOBAL__N_11bE");
// CHECK-NEXT: device_global_map::add((void *)&::outer::__sycl_detail::__shim_[[SHIM5]](), "____ZN5outer12_GLOBAL__N_15inner12_GLOBAL__N_11bE");
// CHECK-NEXT: device_global_map::add((void *)&::outer::__sycl_detail::__shim_[[SHIM9]](), "____ZN5outer12_GLOBAL__N_15inner12_GLOBAL__N_17Wrapper1bE");
// CHECK-NEXT: device_global_map::add((void *)&::outer::__sycl_detail::__shim_[[SHIM11]](), "____ZN5outer12_GLOBAL__N_15inner12_GLOBAL__N_17Wrapper1cE");
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace (unnamed)
// CHECK-NEXT: } // namespace sycl::detail
