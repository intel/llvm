// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -fno-sycl-unnamed-lambda -fsycl-int-header=%t.h %s -sycl-std=2020 -fsyntax-only
// RUN: FileCheck -input-file=%t.h %s --check-prefixes=CHECK,NUL
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -fsycl-int-header=%t.h %s -sycl-std=2020 -fsyntax-only
// RUN: FileCheck -input-file=%t.h %s --check-prefixes=CHECK,UL

#include "Inputs/sycl.hpp"

enum unscoped_enum : int {
  val_1,
  val_2
};

enum class no_namespace_int : int {
  val_1,
  val_2
};

enum class no_namespace_short : short {
  val_1,
  val_2
};

namespace internal {
enum class namespace_short : short {
  val_1,
  val_2
};
}

namespace {
enum class enum_in_anonNS : short {
  val_1,
  val_2
};
}

enum class no_type_set {
  val_1,
  val_2
};

template <no_namespace_int EnumType>
class dummy_functor_1 {
public:
  void operator()() const {}
};

template <no_namespace_short EnumType>
class dummy_functor_2 {
public:
  void operator()() const {}
};

template <internal::namespace_short EnumType>
class dummy_functor_3 {
public:
  void operator()() const {}
};

template <enum_in_anonNS EnumType>
class dummy_functor_4 {
public:
  void operator()() const {}
};

template <no_type_set EnumType>
class dummy_functor_5 {
public:
  void operator()() const {}
};

template <unscoped_enum EnumType>
class dummy_functor_6 {
public:
  void operator()() const {}
};

template <typename EnumType>
class dummy_functor_7 {
public:
  void operator()() const {}
};

namespace type_argument_template_enum {
enum class E : int {
  A,
  B,
  C
};
}

template <typename T>
class T1 {};
template <type_argument_template_enum::E EnumValue>
class T2 {};
template <typename EnumType>
class T3 {};

enum class EnumTypeOut : int { A,
                               B,
};
enum class EnumValueIn : int { A,
                               B,
};
template <EnumValueIn EnumValue, typename EnumTypeIn>
class Baz;
template <typename EnumTypeOut, template <EnumValueIn EnumValue, typename EnumTypeIn> class T>
class dummy_functor_8 {
public:
  void operator()() const {}
};

int main() {

  dummy_functor_1<no_namespace_int::val_1> f1;
  dummy_functor_2<no_namespace_short::val_2> f2;
  dummy_functor_3<internal::namespace_short::val_2> f3;
  dummy_functor_4<enum_in_anonNS::val_2> f4;
  dummy_functor_5<no_type_set::val_1> f5;
  dummy_functor_6<unscoped_enum::val_1> f6;
  dummy_functor_7<no_namespace_int> f7;
  dummy_functor_7<internal::namespace_short> f8;
  dummy_functor_8<EnumTypeOut, Baz> f9;

  cl::sycl::queue q;

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task(f1);
  });

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task(f2);
  });

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task(f3);
  });

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task(f4);
  });

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task(f5);
  });

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task(f6);
  });

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task(f7);
  });

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task(f8);
  });

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task<T1<T2<type_argument_template_enum::E::A>>>([=]() {});
  });

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task<T1<T3<type_argument_template_enum::E>>>([=]() {});
  });

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task(f9);
  });

  return 0;
}

// CHECK: Forward declarations of templated kernel function types:
// NUL: enum class no_namespace_int : int;
// NUL: template <no_namespace_int EnumType> class dummy_functor_1;
// NUL: enum class no_namespace_short : short;
// NUL: template <no_namespace_short EnumType> class dummy_functor_2;
// NUL: namespace internal {
// NUL-NEXT: enum class namespace_short : short;
// NUL-NEXT: }
// NUL: template <internal::namespace_short EnumType> class dummy_functor_3;
// NUL: namespace  {
// NUL-NEXT: enum class enum_in_anonNS : short;
// NUL-NEXT: }
// NUL: template <enum_in_anonNS EnumType> class dummy_functor_4;
// NUL: enum class no_type_set : int;
// NUL: template <no_type_set EnumType> class dummy_functor_5;
// NUL: enum unscoped_enum : int;
// NUL: template <unscoped_enum EnumType> class dummy_functor_6;
// NUL: template <typename EnumType> class dummy_functor_7;
// CHECK: namespace type_argument_template_enum {
// CHECK-NEXT: enum class E : int;
// CHECK-NEXT: }
// CHECK: template <type_argument_template_enum::E EnumValue> class T2;
// CHECK: template <typename T> class T1;
// NUL: enum class EnumTypeOut : int;
// NUL: enum class EnumValueIn : int;
// NUL: template <EnumValueIn EnumValue, typename EnumTypeIn> class Baz;
// NUL: template <typename EnumTypeOut, template <EnumValueIn EnumValue, typename EnumTypeIn> class T> class dummy_functor_8;

// CHECK: Specializations of KernelInfo for kernel function types:
// NUL: template <> struct KernelInfo<::dummy_functor_1<static_cast<::no_namespace_int>(0)>>
// UL: template <> struct KernelInfoData<'_', 'Z', 'T', 'S', '1', '5', 'd', 'u', 'm', 'm', 'y', '_', 'f', 'u', 'n', 'c', 't', 'o', 'r', '_', '1', 'I', 'L', '1', '6', 'n', 'o', '_', 'n', 'a', 'm', 'e', 's', 'p', 'a', 'c', 'e', '_', 'i', 'n', 't', '0', 'E', 'E'>
// NUL: template <> struct KernelInfo<::dummy_functor_2<static_cast<::no_namespace_short>(1)>>
// UL: template <> struct KernelInfoData<'_', 'Z', 'T', 'S', '1', '5', 'd', 'u', 'm', 'm', 'y', '_', 'f', 'u', 'n', 'c', 't', 'o', 'r', '_', '2', 'I', 'L', '1', '8', 'n', 'o', '_', 'n', 'a', 'm', 'e', 's', 'p', 'a', 'c', 'e', '_', 's', 'h', 'o', 'r', 't', '1', 'E', 'E'>
// NUL: template <> struct KernelInfo<::dummy_functor_3<static_cast<::internal::namespace_short>(1)>>
// UL: template <> struct KernelInfoData<'_', 'Z', 'T', 'S', '1', '5', 'd', 'u', 'm', 'm', 'y', '_', 'f', 'u', 'n', 'c', 't', 'o', 'r', '_', '3', 'I', 'L', 'N', '8', 'i', 'n', 't', 'e', 'r', 'n', 'a', 'l', '1', '5', 'n', 'a', 'm', 'e', 's', 'p', 'a', 'c', 'e', '_', 's', 'h', 'o', 'r', 't', 'E', '1', 'E', 'E'>
// NUL: template <> struct KernelInfo<::dummy_functor_4<static_cast<::enum_in_anonNS>(1)>>
// UL: template <> struct KernelInfoData<'_', 'Z', 'T', 'S', '1', '5', 'd', 'u', 'm', 'm', 'y', '_', 'f', 'u', 'n', 'c', 't', 'o', 'r', '_', '4', 'I', 'L', 'N', '1', '2', '_', 'G', 'L', 'O', 'B', 'A', 'L', '_', '_', 'N', '_', '1', '1', '4', 'e', 'n', 'u', 'm', '_', 'i', 'n', '_', 'a', 'n', 'o', 'n', 'N', 'S', 'E', '1', 'E', 'E'>
// NUL: template <> struct KernelInfo<::dummy_functor_5<static_cast<::no_type_set>(0)>>
// UL: template <> struct KernelInfoData<'_', 'Z', 'T', 'S', '1', '5', 'd', 'u', 'm', 'm', 'y', '_', 'f', 'u', 'n', 'c', 't', 'o', 'r', '_', '5', 'I', 'L', '1', '1', 'n', 'o', '_', 't', 'y', 'p', 'e', '_', 's', 'e', 't', '0', 'E', 'E'>
// NUL: template <> struct KernelInfo<::dummy_functor_6<static_cast<::unscoped_enum>(0)>>
// UL: template <> struct KernelInfoData<'_', 'Z', 'T', 'S', '1', '5', 'd', 'u', 'm', 'm', 'y', '_', 'f', 'u', 'n', 'c', 't', 'o', 'r', '_', '6', 'I', 'L', '1', '3', 'u', 'n', 's', 'c', 'o', 'p', 'e', 'd', '_', 'e', 'n', 'u', 'm', '0', 'E', 'E'>
// NUL: template <> struct KernelInfo<::dummy_functor_7<::no_namespace_int>>
// UL: template <> struct KernelInfoData<'_', 'Z', 'T', 'S', '1', '5', 'd', 'u', 'm', 'm', 'y', '_', 'f', 'u', 'n', 'c', 't', 'o', 'r', '_', '7', 'I', '1', '6', 'n', 'o', '_', 'n', 'a', 'm', 'e', 's', 'p', 'a', 'c', 'e', '_', 'i', 'n', 't', 'E'>
// NUL: template <> struct KernelInfo<::dummy_functor_7<::internal::namespace_short>>
// UL: template <> struct KernelInfoData<'_', 'Z', 'T', 'S', '1', '5', 'd', 'u', 'm', 'm', 'y', '_', 'f', 'u', 'n', 'c', 't', 'o', 'r', '_', '7', 'I', 'N', '8', 'i', 'n', 't', 'e', 'r', 'n', 'a', 'l', '1', '5', 'n', 'a', 'm', 'e', 's', 'p', 'a', 'c', 'e', '_', 's', 'h', 'o', 'r', 't', 'E', 'E'>
// CHECK: template <> struct KernelInfo<::T1<::T2<static_cast<::type_argument_template_enum::E>(0)>>>
// CHECK: template <> struct KernelInfo<::T1<::T3<::type_argument_template_enum::E>>>
// NUL: template <> struct KernelInfo<::dummy_functor_8<::EnumTypeOut, Baz>>
// UL: template <> struct KernelInfoData<'_', 'Z', 'T', 'S', '1', '5', 'd', 'u', 'm', 'm', 'y', '_', 'f', 'u', 'n', 'c', 't', 'o', 'r', '_', '8', 'I', '1', '1', 'E', 'n', 'u', 'm', 'T', 'y', 'p', 'e', 'O', 'u', 't', '3', 'B', 'a', 'z', 'E'>
