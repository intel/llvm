// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -triple spir64-unknown-unknown -fsycl-int-footer=%t.h %s -emit-llvm -o %t.ll
// RUN: FileCheck -input-file=%t.h %s
// A test that validates the more complex cases of the specialization-constant
// integration footer details, basically any situation we can come up with that
// has an anonymous namespace.

#include "Inputs/sycl.hpp"
int main() {
  sycl::kernel_single_task<class first_kernel>([]() {});
}

// CHECK: #include <sycl/detail/defines_elementary.hpp>

// Example ways in which the application can declare a "specialization_id"
// variable.
struct S1 {
  static constexpr sycl::specialization_id a{1};
  // CHECK: namespace sycl {
  // CHECK-NEXT: inline namespace _V1 {
  // CHECK-NEXT: namespace detail {
  // CHECK-NEXT: template<>
  // CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::S1::a>() {
  // CHECK-NEXT: return "_ZN2S11aE";
  // CHECK-NEXT: }
  // CHECK-NEXT: } // namespace detail
  // CHECK-NEXT: } // namespace _V1
  // CHECK-NEXT: } // namespace sycl
};
constexpr sycl::specialization_id b{2};
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::b>() {
// CHECK-NEXT: return "____ZL1b";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl
inline constexpr sycl::specialization_id c{3};
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::c>() {
// CHECK-NEXT: return "_Z1c";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl
static constexpr sycl::specialization_id d{4};
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::d>() {
// CHECK-NEXT: return "____ZL1d";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl

namespace {
struct S2 {
  static constexpr sycl::specialization_id a{18};
  // CHECK-NEXT: namespace {
  // CHECK-NEXT: namespace __sycl_detail {
  // CHECK-NEXT: static constexpr decltype(S2::a) &__shim_[[SHIM_ID:[0-9]+]]() {
  // CHECK-NEXT: return S2::a;
  // CHECK-NEXT: }
  // CHECK-NEXT: } // namespace __sycl_detail
  // CHECK-NEXT: } // namespace
  // CHECK-NEXT: namespace sycl {
  // CHECK-NEXT: inline namespace _V1 {
  // CHECK-NEXT: namespace detail {
  // CHECK-NEXT: template<>
  // CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::__sycl_detail::__shim_[[SHIM_ID]]()>() {
  // CHECK-NEXT: return "____ZN12_GLOBAL__N_12S21aE";
  // CHECK-NEXT: }
  // CHECK-NEXT: } // namespace detail
  // CHECK-NEXT: } // namespace _V1
  // CHECK-NEXT: } // namespace sycl
};
} // namespace

template <int Val>
struct S3 {
  static constexpr sycl::specialization_id a{Val};
};
template class S3<1>;
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::S3<1>::a>() {
// CHECK-NEXT: return "_ZN2S3ILi1EE1aE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl
template class S3<2>;
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::S3<2>::a>() {
// CHECK-NEXT: return "_ZN2S3ILi2EE1aE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl

namespace {
  namespace inner {
    template <int Val>
    struct S3 {
      static constexpr sycl::specialization_id a{Val};
    };
  }
}
template class inner::S3<10>;
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(inner::S3<10>::a) &__shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return inner::S3<10>::a;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::__sycl_detail::__shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "____ZN12_GLOBAL__N_15inner2S3ILi10EE1aE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl

namespace {
  namespace inner {
    template <int Val1, int Val2>
    struct S4 {
      static constexpr sycl::specialization_id a{Val1};
    };
  }
}
template class inner::S4<10, 20>;
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(inner::S4<10, 20>::a) &__shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return inner::S4<10, 20>::a;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::__sycl_detail::__shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "____ZN12_GLOBAL__N_15inner2S4ILi10ELi20EE1aE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl

namespace inner {
constexpr sycl::specialization_id same_name{5};
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::inner::same_name>() {
// CHECK-NEXT: return "____ZN5innerL9same_nameE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl
}
constexpr sycl::specialization_id same_name{6};
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::same_name>() {
// CHECK-NEXT: return "____ZL9same_name";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl
namespace {
constexpr sycl::specialization_id same_name{7};
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(same_name) &__shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::__sycl_detail::__shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "____ZN12_GLOBAL__N_19same_nameE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl
}
namespace {
namespace inner {
constexpr sycl::specialization_id same_name{8};
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(inner::same_name) &__shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return inner::same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::__sycl_detail::__shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "____ZN12_GLOBAL__N_15inner9same_nameE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl
}
} // namespace
namespace inner {
namespace {
constexpr sycl::specialization_id same_name{9};
// CHECK-NEXT: namespace inner {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(same_name) &__shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: } // namespace inner
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::inner::__sycl_detail::__shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "____ZN5inner12_GLOBAL__N_19same_nameE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl
}
} // namespace inner

namespace outer {
constexpr sycl::specialization_id same_name{10};
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::outer::same_name>() {
// CHECK-NEXT: return "____ZN5outerL9same_nameE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl
namespace {
constexpr sycl::specialization_id same_name{11};
// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(same_name) &__shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::outer::__sycl_detail::__shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "____ZN5outer12_GLOBAL__N_19same_nameE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl

namespace inner {
constexpr sycl::specialization_id same_name{12};
// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(inner::same_name) &__shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return inner::same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::outer::__sycl_detail::__shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "____ZN5outer12_GLOBAL__N_15inner9same_nameE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl

namespace {
// This has multiple anonymous namespaces in its declaration context, we need to
// make sure we emit a shim for each.
constexpr sycl::specialization_id same_name{13};
// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace inner {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(same_name) &__shim_[[SHIM_ID:[0-9]+]]() {
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
// CHECK-NEXT: static constexpr decltype(inner::__sycl_detail::__shim_[[SHIM_ID]]()) &__shim_[[SHIM_ID_2:[0-9]+]]() {
// CHECK-NEXT: return inner::__sycl_detail::__shim_[[SHIM_ID]]();
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::outer::__sycl_detail::__shim_[[SHIM_ID_2]]()>() {
// CHECK-NEXT: return "____ZN5outer12_GLOBAL__N_15inner12_GLOBAL__N_19same_nameE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl
}
} // namespace inner
} // namespace
} // namespace outer

namespace {
namespace outer {
constexpr sycl::specialization_id same_name{14};
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(outer::same_name) &__shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return outer::same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::__sycl_detail::__shim_[[SHIM_ID]]()>() {
// CHECK-NEXT: return "____ZN12_GLOBAL__N_15outer9same_nameE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl
namespace {
constexpr sycl::specialization_id same_name{15};
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(same_name) &__shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: } // namespace
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(outer::__sycl_detail::__shim_[[SHIM_ID]]()) &__shim_[[SHIM_ID2:[0-9]+]]() {
// CHECK-NEXT: return outer::__sycl_detail::__shim_[[SHIM_ID]]();
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::__sycl_detail::__shim_[[SHIM_ID2]]()>() {
// CHECK-NEXT: return "____ZN12_GLOBAL__N_15outer12_GLOBAL__N_19same_nameE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl
namespace inner {
constexpr sycl::specialization_id same_name{16};
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace outer {
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(inner::same_name) &__shim_[[SHIM_ID:[0-9]+]]() {
// CHECK-NEXT: return inner::same_name;
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: } // namespace outer
// CHECK-NEXT: } // namespace
// CHECK-NEXT: namespace {
// CHECK-NEXT: namespace __sycl_detail {
// CHECK-NEXT: static constexpr decltype(outer::__sycl_detail::__shim_[[SHIM_ID]]()) &__shim_[[SHIM_ID2:[0-9]+]]() {
// CHECK-NEXT: return outer::__sycl_detail::__shim_[[SHIM_ID]]();
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace __sycl_detail
// CHECK-NEXT: } // namespace
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::__sycl_detail::__shim_[[SHIM_ID2]]()>() {
// CHECK-NEXT: return "____ZN12_GLOBAL__N_15outer12_GLOBAL__N_15inner9same_nameE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl
}
} // namespace
} // namespace outer
} // namespace

namespace outer {
namespace inner {
constexpr sycl::specialization_id same_name{17};
// CHECK: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {
// CHECK-NEXT: template<>
// CHECK-NEXT: inline const char *get_spec_constant_symbolic_ID_impl<::outer::inner::same_name>() {
// CHECK-NEXT: return "____ZN5outer5innerL9same_nameE";
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace detail
// CHECK-NEXT: } // namespace _V1
// CHECK-NEXT: } // namespace sycl
}
} // namespace outer

// CHECK: #include <sycl/detail/spec_const_integration.hpp>
