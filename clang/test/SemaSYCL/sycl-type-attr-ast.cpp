// RUN: %clang_cc1 -fsycl-is-device -ast-dump %s | FileCheck %s

// Tests for AST of sycl_type() attribute

class [[__sycl_detail__::sycl_type(accessor)]] accessor {};
// CHECK: CXXRecordDecl {{.*}} class accessor definition
// CHECK: SYCLTypeAttr {{.*}} accessor

class accessor;
// CHECK: CXXRecordDecl {{.*}} prev {{.*}} class accessor
// CHECK: SYCLTypeAttr {{.*}} Inherited accessor

enum class [[__sycl_detail__::sycl_type(aspect)]] aspect {};
// CHECK: EnumDecl {{.*}} class aspect 'int'
// CHECK: SYCLTypeAttr {{.*}} aspect

template <typename T>
class [[__sycl_detail__::sycl_type(local_accessor)]] local_accessor {};
// CHECK: ClassTemplateDecl {{.*}} local_accessor
// CHECK: CXXRecordDecl {{.*}} class local_accessor definition
// CHECK: SYCLTypeAttr {{.*}} local_accessor

template <>
class [[__sycl_detail__::sycl_type(local_accessor)]] local_accessor <int> {};
// CHECK: ClassTemplateSpecializationDecl {{.*}} class local_accessor definition
// CHECK: SYCLTypeAttr {{.*}} local_accessor

class [[__sycl_detail__::sycl_type(multi_ptr)]] multi_ptr {};
// CHECK: CXXRecordDecl {{.*}} class multi_ptr definition
// CHECK: SYCLTypeAttr {{.*}} multi_ptr
