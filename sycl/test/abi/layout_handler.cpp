// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -ast-dump %s | FileCheck %s

#include <CL/sycl.hpp>

// The order of field declarations and their types are important.

// CHECK: CXXRecordDecl {{.*}} class handler definition
// CHECK: AccessSpecDecl {{.*}} private
// CHECK-NEXT: FieldDecl {{.*}} MQueue 'shared_ptr_class<detail::queue_impl>':'std::shared_ptr<cl::sycl::detail::queue_impl>'
// CHECK-NEXT: FieldDecl {{.*}} referenced MArgsStorage 'vector_class<vector_class<char> >':'std::vector<std::vector<char, std::allocator<char> >, std::allocator<std::vector<char, std::allocator<char> > > >'
// CHECK-NEXT: FieldDecl {{.*}} referenced MAccStorage 'vector_class<detail::AccessorImplPtr>':'std::vector<std::shared_ptr<cl::sycl::detail::AccessorImplHost>, std::allocator<std::shared_ptr<cl::sycl::detail::AccessorImplHost> > >'
// CHECK-NEXT: FieldDecl {{.*}} referenced MLocalAccStorage 'vector_class<detail::LocalAccessorImplPtr>':'std::vector<std::shared_ptr<cl::sycl::detail::LocalAccessorImplHost>, std::allocator<std::shared_ptr<cl::sycl::detail::LocalAccessorImplHost> > >'
// CHECK-NEXT: FieldDecl {{.*}} referenced MStreamStorage 'vector_class<shared_ptr_class<detail::stream_impl> >':'std::vector<std::shared_ptr<cl::sycl::detail::stream_impl>, std::allocator<std::shared_ptr<cl::sycl::detail::stream_impl> > >'
// CHECK-NEXT: FieldDecl {{.*}} referenced MSharedPtrStorage 'vector_class<shared_ptr_class<const void> >':'std::vector<std::shared_ptr<const void>, std::allocator<std::shared_ptr<const void> > >'
// CHECK-NEXT: FieldDecl {{.*}} referenced MArgs 'vector_class<detail::ArgDesc>':'std::vector<cl::sycl::detail::ArgDesc, std::allocator<cl::sycl::detail::ArgDesc> >'
// CHEKC-NEXT: FieldDecl {{.*}} referenced MAssociatedAccesors 'vector_class<detail::ArgDesc>':'std::vector<cl::sycl::detail::ArgDesc, std::allocator<cl::sycl::detail::ArgDesc> >'
// CHECK-NEXT: FieldDecl {{.*}} referenced MRequirements 'vector_class<detail::Requirement *>':'std::vector<cl::sycl::detail::AccessorImplHost *, std::allocator<cl::sycl::detail::AccessorImplHost *> >'
// CHECK-NEXT: FieldDecl {{.*}} referenced MNDRDesc 'detail::NDRDescT':'cl::sycl::detail::NDRDescT'
// CHECK-NEXT: FieldDecl {{.*}} referenced MKernelName 'cl::sycl::string_class':'std::__cxx11::basic_string<char>'
// CHECK-NEXT: FieldDecl {{.*}} referenced MKernel 'shared_ptr_class<detail::kernel_impl>':'std::shared_ptr<cl::sycl::detail::kernel_impl>'
// CHECK-NEXT: FieldDecl {{.*}} referenced MCGType 'detail::CG::CGTYPE':'cl::sycl::detail::CG::CGTYPE'
// CHECK-NEXT: DeclRefExpr {{.*}} 'cl::sycl::detail::CG::CGTYPE' EnumConstant {{.*}} 'NONE' 'cl::sycl::detail::CG::CGTYPE'
// CHECK-NEXT: FieldDecl {{.*}} referenced MSrcPtr 'void *'
// CHECK-NEXT: ImplicitCastExpr {{.*}} <NullToPointer>
// CHECK-NEXT: CXXNullPtrLiteralExpr {{.*}} 'nullptr_t'
// CHECK-NEXT: FieldDecl {{.*}} referenced MDstPtr 'void *'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void *' <NullToPointer>
// CHECK-NEXT: CXXNullPtrLiteralExpr {{.*}} 'nullptr_t'
// CHECK-NEXT: FieldDecl {{.*}} col:10 referenced MLength 'size_t':'unsigned long'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'size_t':'unsigned long' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: FieldDecl {{.*}} referenced MPattern 'vector_class<char>':'std::vector<char, std::allocator<char> >'
// CHECK-NEXT: FieldDecl {{.*}} referenced MHostKernel 'unique_ptr_class<detail::HostKernelBase>':'std::unique_ptr<cl::sycl::detail::HostKernelBase, std::default_delete<cl::sycl::detail::HostKernelBase> >'
// CHECK-NEXT: FieldDecl {{.*}} referenced MOSModuleHandle 'detail::OSModuleHandle':'long'
// CHECK-NEXT: FieldDecl {{.*}} referenced MInteropTask 'std::unique_ptr<detail::InteropTask>':'std::unique_ptr<cl::sycl::detail::InteropTask, std::default_delete<cl::sycl::detail::InteropTask> >'
// CHECK-NEXT: FieldDecl {{.*}} referenced MEvents 'vector_class<detail::EventImplPtr>':'std::vector<std::shared_ptr<cl::sycl::detail::event_impl>, std::allocator<std::shared_ptr<cl::sycl::detail::event_impl> > >'
// CHECK-NEXT: FieldDecl {{.*}} referenced MIsHost 'bool'
// CHECK-NEXT: CXXBoolLiteralExpr {{.*}} 'bool' false
