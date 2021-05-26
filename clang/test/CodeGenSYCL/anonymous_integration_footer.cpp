// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -triple spir64-unknown-unknown-sycldevice -fsycl-int-footer=%t.h %s -emit-llvm -o %t.ll
// RUN: FileCheck -input-file=%t.h %s
// A test that validates the more complex cases of the specialization-constant
// integration footer details, basically any situation we can come up with that
// has an anonymous namespace.

#include "Inputs/sycl.hpp"
int main() {
  cl::sycl::kernel_single_task<class first_kernel>([]() {});
}

using namespace cl;
// Example ways in which the application can declare a "specialization_id"
// variable.
struct S1 {
  static constexpr sycl::specialization_id a{1};
  // CHECK: __SYCL_INLINE_NAMESPACE(cl) {
  // CHECK-NEXT: namespace sycl {
  // CHECK-NEXT: namespace detail {
  // CHECK-NEXT: template<>
  // CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::S1::a>() {
  // CHECK-NEXT: return "";
  // CHECK-NEXT: }
  // CHECK-NEXT: } // namespace detail
  // CHECK-NEXT: } // namespace sycl
  // CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
};
constexpr sycl::specialization_id b{2};
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::b>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
inline constexpr sycl::specialization_id c{3};
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::c>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
static constexpr sycl::specialization_id d{4};
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::d>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)

namespace {
struct S2 {
  static constexpr sycl::specialization_id a{18};
  // CHECK-NEXT: namespace {
  // CHECK-NEXT: namespace __sycl_detail {
  // CHECK-NEXT: static constexpr decltype(S2::a) &__spec_id_shim_[[SHIM_ID:[0-9]+]]() {
  // CHECK-NEXT: return S2::a;
  // CHECK-NEXT: }
  // CHECK-NEXT: } // namespace __sycl_detail
  // CHECK-NEXT: } // namespace
  // CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
  // CHECK-NEXT: namespace sycl {
  // CHECK-NEXT: namespace detail {
  // CHECK-NEXT: template<>
  // CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::__sycl_detail::__spec_id_shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "";
  // CHECK-NEXT: }
  // CHECK-NEXT: } // namespace detail
  // CHECK-NEXT: } // namespace sycl
  // CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
};
} // namespace

template <int Val>
struct S3 {
  static constexpr sycl::specialization_id a{Val};
};
template class S3<1>;
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::S3<1>::a>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
template class S3<2>;
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::S3<2>::a>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)

namespace inner {
constexpr sycl::specialization_id same_name{5};
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::inner::same_name>() {
// CHECK-NEXT: return "";
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
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::same_name>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
namespace {
constexpr sycl::specialization_id same_name{7};
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(same_name) &__spec_id_shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::__sycl_detail::__spec_id_shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
}
namespace {
namespace inner {
constexpr sycl::specialization_id same_name{8};
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(inner::same_name) &__spec_id_shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return inner::same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::__sycl_detail::__spec_id_shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
}
} // namespace
namespace inner {
namespace {
constexpr sycl::specialization_id same_name{9};
// CHECK-NEXT: namespace inner {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(same_name) &__spec_id_shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: } // namespace inner
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::inner::__sycl_detail::__spec_id_shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
}
} // namespace inner

namespace outer {
constexpr sycl::specialization_id same_name{10};
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::outer::same_name>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
namespace {
constexpr sycl::specialization_id same_name{11};
// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(same_name) &__spec_id_shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::outer::__sycl_detail::__spec_id_shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)

namespace inner {
constexpr sycl::specialization_id same_name{12};
// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(inner::same_name) &__spec_id_shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return inner::same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::outer::__sycl_detail::__spec_id_shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)

namespace {
// This has multiple anonymous namespaces in its declaration context, we need to
// make sure we emit a shim for each.
constexpr sycl::specialization_id same_name{13};
// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace inner {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(same_name) &__spec_id_shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: } // namespace inner
// CHECK-NEXT: } // namespace
// CHECK-NEXT: } // namespace outer

// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(inner::__sycl_detail::__spec_id_shim_[[SHIM_ID]]()) &__spec_id_shim_[[SHIM_ID_2:[0-9]+]]() {
// CHECK-NEXT: return inner::__sycl_detail::__spec_id_shim_[[SHIM_ID]]();
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::outer::__sycl_detail::__spec_id_shim_[[SHIM_ID_2]]()>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
}
} // namespace inner
} // namespace
} // namespace outer

namespace {
namespace outer {
constexpr sycl::specialization_id same_name{14};
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(outer::same_name) &__spec_id_shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return outer::same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::__sycl_detail::__spec_id_shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
namespace {
constexpr sycl::specialization_id same_name{15};
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(same_name) &__spec_id_shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: } // namespace
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(outer::__sycl_detail::__spec_id_shim_[[SHIM_ID]]()) &__spec_id_shim_[[SHIM_ID2:[0-9]+]]() {
// CHECK-NEXT: return outer::__sycl_detail::__spec_id_shim_[[SHIM_ID]]();
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::__sycl_detail::__spec_id_shim_[[SHIM_ID2]]()>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
namespace inner {
constexpr sycl::specialization_id same_name{16};
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(inner::same_name) &__spec_id_shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return inner::same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: } // namespace
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(outer::__sycl_detail::__spec_id_shim_[[SHIM_ID]]()) &__spec_id_shim_[[SHIM_ID2:[0-9]+]]() {
// CHECK-NEXT: return outer::__sycl_detail::__spec_id_shim_[[SHIM_ID]]();
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::__sycl_detail::__spec_id_shim_[[SHIM_ID2]]()>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
}
} // namespace
} // namespace outer
} // namespace

namespace outer {
namespace inner {
constexpr sycl::specialization_id same_name{17};
// CHECK: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID<::outer::inner::same_name>() {
// CHECK-NEXT: return "";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace sycl
// CHECK-NEXT: } // __SYCL_INLINE_NAMESPACE(cl)
}
} // namespace outer

// CHECK: #include <CL/sycl/detail/spec_const_integration.hpp>
