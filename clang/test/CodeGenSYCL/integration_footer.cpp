// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsycl-int-footer=%t.h %s -emit-llvm -o %t.ll
// RUN: FileCheck -input-file=%t.h %s

#include "Inputs/sycl.hpp"

int main() {
  cl::sycl::kernel_single_task<class first_kernel>([]() {});
}

// CHECK: #include <CL/sycl/detail/defines_elementary.hpp>

using namespace cl::sycl;

cl::sycl::specialization_id<int> GlobalSpecID;
// CHECK: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::GlobalSpecID>() {
// CHECK-NEXT: return "_Z12GlobalSpecID";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl

struct Wrapper {
  static specialization_id<int> WrapperSpecID;
  // CHECK: namespace sycl {
  // CHECK-NEXT: namespace detail {
  // CHECK-NEXT: template<>
  // CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::Wrapper::WrapperSpecID>() {
  // CHECK-NEXT: return "_ZN7Wrapper13WrapperSpecIDE";
  // CHECK-NEXT: }
  // CHECK-NEXT: } // namespace detail
  // CHECK-NEXT: } // namespace sycl
};

template <typename T>
struct WrapperTemplate {
  static specialization_id<T> WrapperSpecID;
};
template class WrapperTemplate<int>;
// CHECK: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::WrapperTemplate<int>::WrapperSpecID>() {
// CHECK-NEXT: return "_ZN15WrapperTemplateIiE13WrapperSpecIDE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
template class WrapperTemplate<double>;
// CHECK: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::WrapperTemplate<double>::WrapperSpecID>() {
// CHECK-NEXT: return "_ZN15WrapperTemplateIdE13WrapperSpecIDE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl

namespace Foo {
specialization_id<int> NSSpecID;
// CHECK: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::Foo::NSSpecID>() {
// CHECK-NEXT: return "_ZN3Foo8NSSpecIDE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
inline namespace Bar {
specialization_id<int> InlineNSSpecID;
// CHECK: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::Foo::InlineNSSpecID>() {
// CHECK-NEXT: return "_ZN3Foo3Bar14InlineNSSpecIDE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
specialization_id<int> NSSpecID;
// CHECK: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::Foo::Bar::NSSpecID>() {
// CHECK-NEXT: return "_ZN3Foo3Bar8NSSpecIDE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl

struct Wrapper {
  static specialization_id<int> WrapperSpecID;
  // CHECK: namespace sycl {
  // CHECK-NEXT: namespace detail {
  // CHECK-NEXT: template<>
  // CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::Foo::Wrapper::WrapperSpecID>() {
  // CHECK-NEXT: return "_ZN3Foo3Bar7Wrapper13WrapperSpecIDE";
  // CHECK-NEXT: }
  // CHECK-NEXT: } // namespace detail
  // CHECK-NEXT: } // namespace sycl
};

template <typename T>
struct WrapperTemplate {
  static specialization_id<T> WrapperSpecID;
};
template class WrapperTemplate<int>;
// CHECK: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::Foo::WrapperTemplate<int>::WrapperSpecID>() {
// CHECK-NEXT: return "_ZN3Foo3Bar15WrapperTemplateIiE13WrapperSpecIDE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
template class WrapperTemplate<double>;
// CHECK: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::Foo::WrapperTemplate<double>::WrapperSpecID>() {
// CHECK-NEXT: return "_ZN3Foo3Bar15WrapperTemplateIdE13WrapperSpecIDE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
} // namespace Bar
namespace {
specialization_id<int> AnonNSSpecID;

// CHECK: namespace Foo {
// CHECK: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(AnonNSSpecID) &__shim_[[SHIM0:[0-9]+]]() {
// CHECK-NEXT: return AnonNSSpecID;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: } // namespace Foo

// CHECK: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::Foo::__sycl_detail::__shim_[[SHIM0]]()>() {
// CHECK-NEXT: return "____ZN3Foo12_GLOBAL__N_112AnonNSSpecIDE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
} // namespace

} // namespace Foo

// make sure we don't emit a deduced type that isn't a spec constant.
enum SomeEnum { SE_A };
enum AnotherEnum : unsigned int { AE_A };

template<SomeEnum E> struct GetThing{};
template<> struct GetThing<SE_A>{
  static constexpr auto thing = AE_A;
};

struct container {
  static constexpr auto Thing = GetThing<SE_A>::thing;
};
// CHECK-NOT: ::GetThing
// CHECK-NOT: ::container::Thing

// Validate that variable templates work correctly.  Previously they printed
// without their template arguments.
namespace {
struct HasVarTemplate {
  constexpr HasVarTemplate(){}
  template<typename T, int case_num>
  static constexpr specialization_id<T> VarTempl{case_num};
};
}

auto x = HasVarTemplate::VarTempl<int, 2>.getDefaultValue();
// CHECK: namespace {
// CHECK-NEXT: namespace __sycl_detail
// CHECK-NEXT: static constexpr decltype(HasVarTemplate::VarTempl<int, 2>) &__shim_[[SHIM1:[0-9]+]]() {
// CHECK-NEXT: return HasVarTemplate::VarTempl<int, 2>;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::__sycl_detail::__shim_[[SHIM1]]()>() {
// CHECK-NEXT: return "____ZN12_GLOBAL__N_114HasVarTemplate8VarTemplIiLi2EEE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)

template <typename T> struct GlobalWrapper {
  template<int Value> static constexpr specialization_id<T> sc{Value};
};

auto &y = GlobalWrapper<int>::template sc<20>;

// Should not generate the uninstantiated template.
// CHECK-NOT: inline const char *get_spec_constant_symbolic_ID_impl<::GlobalWrapper<int>::sc>()
// CHECK: inline const char *get_spec_constant_symbolic_ID_impl<::GlobalWrapper<int>::sc<20>>()

// CHECK: #include <CL/sycl/detail/spec_const_integration.hpp>
