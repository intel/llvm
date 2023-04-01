// RUN: %clangxx -fsyntax-only -fno-color-diagnostics -std=c++17 -I %sycl_include/sycl -I %sycl_include -Xclang -ast-dump %s | FileCheck %s
// RUN: %clangxx -c -fno-color-diagnostics -std=c++17 -I %sycl_include/sycl -I %sycl_include -Xclang -emit-llvm -g %s -o - | FileCheck %s --check-prefixes CHECK-DEBUG-INFO
// UNSUPPORTED: windows
#include <sycl/sycl.hpp>

void foo(sycl::buffer<int, 1> &BufA) {
  auto HostAcc = BufA.get_access<sycl::access_mode::read>();

  sycl::accessor<char, 1, sycl::access::mode::read_write, sycl::target::local>
      *LocalAcc;
}

// Host accessors should have the following methods which are used by gdb
// pretty-printers
//
// CHECK: CXXRecordDecl {{.*}} class accessor definition
// CHECK: CXXMethodDecl {{.*}}getOffset
// CHECK: CXXMethodDecl {{.*}}getAccessRange
// CHECK: CXXMethodDecl {{.*}}getMemoryRange
// CHECK: CXXMethodDecl {{.*}}getPtr

// CHECK: CXXRecordDecl {{.*}} class local_accessor_base definition
// CHECK: CXXMethodDecl {{.*}}getSize
// CHECK: CXXMethodDecl {{.*}}getPtr

// CHECK-DEBUG-INFO: !DICompositeType(tag: DW_TAG_class_type, name: "accessor<int, 1, (sycl::_V1::access::mode)1024, (sycl::_V1::access::target)2018, (sycl::_V1::access::placeholder)0, sycl::_V1::ext::oneapi::accessor_property_list<> >", {{.*}}, templateParams: ![[TEMPL_METADATA:[0-9]+]]
// CHECK-DEBUG-INFO-DAG: ![[TEMPL_METADATA]] = !{![[DATA_T:[0-9]+]], ![[Dims:[0-9]+]], ![[AccMode:[0-9]+]], ![[AccTarget:[0-9]+]], ![[IsPlh:[0-9]+]], ![[PropListT:[0-9]+]]}
// CHECK-DEBUG-INFO-DAG: ![[DATA_T]] = !DITemplateTypeParameter(name: "DataT"
// CHECK-DEBUG-INFO-DAG: ![[Dims]] = !DITemplateValueParameter(name: "Dimensions"
// CHECK-DEBUG-INFO-DAG: ![[AccMode]] = !DITemplateValueParameter(name: "AccessMode"
// CHECK-DEBUG-INFO-DAG: ![[AccTarget]] = !DITemplateValueParameter(name: "AccessTarget"
// CHECK-DEBUG-INFO-DAG: ![[IsPlh]] = !DITemplateValueParameter(name: "IsPlaceholder"
// CHECK-DEBUG-INFO-DAG: ![[PropListT]] = !DITemplateTypeParameter(name: "PropertyListT"
// CHECK-NOT: !DICompositeType(tag: DW_TAG_class_type, name: "accessor<i
