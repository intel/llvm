// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -triple spir64-unknown-unknown -fsycl-int-footer=%t.h %s -emit-llvm -o %t.ll
// RUN: FileCheck -input-file=%t.h %s
// A test that validates the more complex cases of the specialization-constant
// integration footer details, basically any situation we can come up with that
// has an anonymous namespace.

#include "Inputs/sycl.hpp"
int main() {
  cl::sycl::kernel_single_task<class first_kernel>([]() {});
}

// CHECK: #include <CL/sycl/detail/defines_elementary.hpp>

using namespace cl;

struct S1 {
  static constexpr sycl::specialization_id a{1};
};
// CHECK: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::S1::a>() {
// CHECK-NEXT: return "_ZN2S11aE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)

constexpr sycl::specialization_id b{202};
// CHECK: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::b>() {
// CHECK-NEXT: return "____ZL1b";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
inline constexpr sycl::specialization_id c{3};
// CHECK: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::c>() {
// CHECK-NEXT: return "_Z1c";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
static constexpr sycl::specialization_id d{205};
// CHECK: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::d>() {
// CHECK-NEXT: return "____ZL1d";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)

namespace inner {
constexpr sycl::specialization_id same_name{5};
// CHECK: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::inner::same_name>() {
// CHECK-NEXT: return "____ZN5innerL9same_nameE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
}
constexpr sycl::specialization_id same_name{6};
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::same_name>() {
// CHECK-NEXT: return "____ZL9same_name";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
namespace {
constexpr sycl::specialization_id same_name{207};
// CHECK: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(same_name) &__shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::__sycl_detail::__shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "____ZN12_GLOBAL__N_19same_nameE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
}
namespace {
namespace inner {
constexpr sycl::specialization_id same_name{208};
// CHECK: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(inner::same_name) &__shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return inner::same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::__sycl_detail::__shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "____ZN12_GLOBAL__N_15inner9same_nameE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
}
} // namespace

namespace outer::inline inner {
namespace {
constexpr sycl::specialization_id same_name{209};
// CHECK: namespace outer {
// CHECK-NEXT: namespace inner {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(same_name) &__shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: } // inline namespace inner
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::outer::inner::__sycl_detail::__shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "____ZN5outer5inner12_GLOBAL__N_19same_nameE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
}
}

// CHECK: #include <CL/sycl/detail/spec_const_integration.hpp>
