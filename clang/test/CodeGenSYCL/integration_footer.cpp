// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -fsycl-int-footer=%t.h %s -emit-llvm -o %t.ll
// RUN: FileCheck -input-file=%t.h %s

#include "Inputs/sycl.hpp"

int main() {
  cl::sycl::kernel_single_task<class first_kernel>([]() {});
}

using namespace cl::sycl;

cl::sycl::specialization_id<int> GlobalSpecID;
// CHECK: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<GlobalSpecID>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }

struct Wrapper {
  static specialization_id<int> WrapperSpecID;
  // CHECK: template<>
  // CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<Wrapper::WrapperSpecID>() {
  // CHECK-NEXT: return "";
  // CHECK-NEXT: }
};

template <typename T>
struct WrapperTemplate {
  static specialization_id<T> WrapperSpecID;
};
template class WrapperTemplate<int>;
// CHECK: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<WrapperTemplate<int>::WrapperSpecID>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
template class WrapperTemplate<double>;
// CHECK: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<WrapperTemplate<double>::WrapperSpecID>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }

namespace Foo {
specialization_id<int> NSSpecID;
// CHECK: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<Foo::NSSpecID>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
inline namespace Bar {
specialization_id<int> InlineNSSpecID;
// CHECK: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<Foo::InlineNSSpecID>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
specialization_id<int> NSSpecID;
// CHECK: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<Foo::Bar::NSSpecID>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }

struct Wrapper {
  static specialization_id<int> WrapperSpecID;
  // CHECK: template<>
  // CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<Foo::Wrapper::WrapperSpecID>() {
  // CHECK-NEXT: return "";
  // CHECK-NEXT: }
};

template <typename T>
struct WrapperTemplate {
  static specialization_id<T> WrapperSpecID;
};
template class WrapperTemplate<int>;
// CHECK: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<Foo::WrapperTemplate<int>::WrapperSpecID>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
template class WrapperTemplate<double>;
// CHECK: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<Foo::WrapperTemplate<double>::WrapperSpecID>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
} // namespace Bar
namespace {
specialization_id<int> AnonNSSpecID;
// CHECK: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<Foo::AnonNSSpecID>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
} // namespace

} // namespace Foo

// CHECK: #include <CL/sycl/detail/spec_const_integration.hpp>
