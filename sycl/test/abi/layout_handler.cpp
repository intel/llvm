// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -ast-dump %s | FileCheck %s
// REQUIRES: linux

#include <CL/sycl/handler.hpp>

// The order of field declarations and their types are important.

// CHECK: CXXRecordDecl {{.*}} class handler definition
// CHECK: FieldDecl {{.*}} MQueue 'shared_ptr_class<detail::queue_impl>':'std::shared_ptr<cl::sycl::detail::queue_impl>'
// CHECK-NEXT: FieldDecl {{.*}} MArgsStorage 'vector_class<vector_class<char>>':'std::vector<std::vector<char, std::allocator<char>>, std::allocator<std::vector<char, std::allocator<char>>>>'
// CHECK-NEXT: FieldDecl {{.*}} MAccStorage 'vector_class<detail::AccessorImplPtr>':'std::vector<std::shared_ptr<cl::sycl::detail::AccessorImplHost>, std::allocator<std::shared_ptr<cl::sycl::detail::AccessorImplHost>>>'
// CHECK-NEXT: FieldDecl {{.*}} MLocalAccStorage 'vector_class<detail::LocalAccessorImplPtr>':'std::vector<std::shared_ptr<cl::sycl::detail::LocalAccessorImplHost>, std::allocator<std::shared_ptr<cl::sycl::detail::LocalAccessorImplHost>>>'
// CHECK-NEXT: FieldDecl {{.*}} MStreamStorage 'vector_class<shared_ptr_class<detail::stream_impl>>':'std::vector<std::shared_ptr<cl::sycl::detail::stream_impl>, std::allocator<std::shared_ptr<cl::sycl::detail::stream_impl>>>'
// CHECK-NEXT: FieldDecl {{.*}} MSharedPtrStorage 'vector_class<shared_ptr_class<const void>>':'std::vector<std::shared_ptr<const void>, std::allocator<std::shared_ptr<const void>>>'
// CHECK-NEXT: FieldDecl {{.*}} MArgs 'vector_class<detail::ArgDesc>':'std::vector<cl::sycl::detail::ArgDesc, std::allocator<cl::sycl::detail::ArgDesc>>'
// CHECK-NEXT: FieldDecl {{.*}} MAssociatedAccesors 'vector_class<detail::ArgDesc>':'std::vector<cl::sycl::detail::ArgDesc, std::allocator<cl::sycl::detail::ArgDesc>>'
// CHECK-NEXT: FieldDecl {{.*}} MRequirements 'vector_class<detail::Requirement *>':'std::vector<cl::sycl::detail::AccessorImplHost *, std::allocator<cl::sycl::detail::AccessorImplHost *>>'
// CHECK-NEXT: FieldDecl {{.*}} MNDRDesc 'detail::NDRDescT':'cl::sycl::detail::NDRDescT'
// CHECK-NEXT: FieldDecl {{.*}} MKernelName 'cl::sycl::string_class':'std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>'
// CHECK-NEXT: FieldDecl {{.*}} MKernel 'shared_ptr_class<detail::kernel_impl>':'std::shared_ptr<cl::sycl::detail::kernel_impl>'
// CHECK-NEXT: FieldDecl {{.*}} MCGType 'detail::CG::CGTYPE':'cl::sycl::detail::CG::CGTYPE'
// CHECK-NEXT: DeclRefExpr {{.*}} 'cl::sycl::detail::CG::CGTYPE' EnumConstant {{.*}} 'NONE' 'cl::sycl::detail::CG::CGTYPE'
// CHECK-NEXT: FieldDecl {{.*}} MSrcPtr 'void *'
// CHECK-NEXT: ImplicitCastExpr {{.*}} <NullToPointer>
// CHECK-NEXT: CXXNullPtrLiteralExpr {{.*}} 'nullptr_t'
// CHECK-NEXT: FieldDecl {{.*}} MDstPtr 'void *'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void *' <NullToPointer>
// CHECK-NEXT: CXXNullPtrLiteralExpr {{.*}} 'nullptr_t'
// CHECK-NEXT: FieldDecl {{.*}} MLength 'size_t':'unsigned long'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'size_t':'unsigned long' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: FieldDecl {{.*}} MPattern 'vector_class<char>':'std::vector<char, std::allocator<char>>'
// CHECK-NEXT: FieldDecl {{.*}} MHostKernel 'unique_ptr_class<detail::HostKernelBase>':'std::unique_ptr<cl::sycl::detail::HostKernelBase, std::default_delete<cl::sycl::detail::HostKernelBase>>'
// CHECK-NEXT: FieldDecl {{.*}} MHostTask 'unique_ptr_class<detail::HostTask>':'std::unique_ptr<cl::sycl::detail::HostTask, std::default_delete<cl::sycl::detail::HostTask>>'
// CHECK-NEXT: FieldDecl {{.*}} MOSModuleHandle 'detail::OSModuleHandle':'long'
// CHECK-NEXT: FieldDecl {{.*}} MInteropTask 'std::unique_ptr<detail::InteropTask>':'std::unique_ptr<cl::sycl::detail::InteropTask, std::default_delete<cl::sycl::detail::InteropTask>>'
// CHECK-NEXT: FieldDecl {{.*}} MEvents 'vector_class<detail::EventImplPtr>':'std::vector<std::shared_ptr<cl::sycl::detail::event_impl>, std::allocator<std::shared_ptr<cl::sycl::detail::event_impl>>>'
// CHECK-NEXT: FieldDecl {{.*}} MIsHost 'bool'
// CHECK-NEXT: CXXBoolLiteralExpr {{.*}} 'bool' false
