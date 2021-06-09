// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -fsycl-int-footer=%t.h %s -emit-llvm -o %t.ll
// RUN: FileCheck -input-file=%t.h %s

#include "Inputs/sycl.hpp"

int main() {
  cl::sycl::kernel_single_task<class first_kernel>([]() {});
}

using namespace cl::sycl;

cl::sycl::specialization_id<int> GlobalSpecID;
// CHECK: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::GlobalSpecID>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl

struct Wrapper {
  static specialization_id<int> WrapperSpecID;
  // CHECK: namespace sycl {
  // CHECK-NEXT: namespace detail {
  // CHECK-NEXT: template<>
  // CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::Wrapper::WrapperSpecID>() {
  // CHECK-NEXT: return "";
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
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::WrapperTemplate<int>::WrapperSpecID>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
template class WrapperTemplate<double>;
// CHECK: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::WrapperTemplate<double>::WrapperSpecID>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl

namespace Foo {
specialization_id<int> NSSpecID;
// CHECK: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::Foo::NSSpecID>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
inline namespace Bar {
specialization_id<int> InlineNSSpecID;
// CHECK: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::Foo::InlineNSSpecID>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
specialization_id<int> NSSpecID;
// CHECK: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::Foo::Bar::NSSpecID>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl

struct Wrapper {
  static specialization_id<int> WrapperSpecID;
  // CHECK: namespace sycl {
  // CHECK-NEXT: namespace detail {
  // CHECK-NEXT: template<>
  // CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::Foo::Wrapper::WrapperSpecID>() {
  // CHECK-NEXT: return "";
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
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::Foo::WrapperTemplate<int>::WrapperSpecID>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
template class WrapperTemplate<double>;
// CHECK: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::Foo::WrapperTemplate<double>::WrapperSpecID>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
} // namespace Bar
namespace {
specialization_id<int> AnonNSSpecID;

// CHECK: namespace Foo {
// CHECK: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(AnonNSSpecID) &__spec_id_shim_[[SHIM0:[0-9]+]]() {
// CHECK-NEXT: return AnonNSSpecID;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: } // namespace Foo

// CHECK: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::Foo::__sycl_detail::__spec_id_shim_[[SHIM0]]()>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
} // namespace

} // namespace Foo

// CHECK: #include <CL/sycl/detail/spec_const_integration.hpp>
