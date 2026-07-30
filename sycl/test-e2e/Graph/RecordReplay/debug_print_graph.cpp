// RUN: %{build} -o %t.out
// RUN: %if linux && (level_zero || cuda) %{ %{run} %t.out ; FileCheck %s --implicit-check-not=LEAK --input-file graph.dot %} %else %{ %{run} %t.out %}
// Windows output format differs from linux format.
// The filecheck-based output checking is suited to linux standards.
// On Windows, we only test that printing takes place correctly and does not
// trigger errors or throw exceptions.
//
// CHECK: digraph dot {
// CHECK-NEXT: "0x[[#%x,NODE1:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE1]]\nTYPE = CGExecKernel \nNAME = typeinfo name for run_kernels<unsigned short>(sycl::_V1::queue, unsigned long, sycl::_V1::buffer<unsigned short, 1, sycl::_V1::detail::aligned_allocator<std::remove_const<unsigned short>::type>, void>, sycl::_V1::buffer<unsigned short, 1, sycl::_V1::detail::aligned_allocator<std::remove_const<unsigned short>::type>, void>, sycl::_V1::buffer<unsigned short, 1, sycl::_V1::detail::aligned_allocator<std::remove_const<unsigned short>::type>, void>)::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda(sycl::_V1::item<1, true>)#1}\n"];
// CHECK-NEXT: "0x[[#%x,NODE2:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE2]]\nTYPE = CGExecKernel \nNAME = typeinfo name for run_kernels<unsigned short>(sycl::_V1::queue, unsigned long, sycl::_V1::buffer<unsigned short, 1, sycl::_V1::detail::aligned_allocator<std::remove_const<unsigned short>::type>, void>, sycl::_V1::buffer<unsigned short, 1, sycl::_V1::detail::aligned_allocator<std::remove_const<unsigned short>::type>, void>, sycl::_V1::buffer<unsigned short, 1, sycl::_V1::detail::aligned_allocator<std::remove_const<unsigned short>::type>, void>)::{lambda(sycl::_V1::handler&)#2}::operator()(sycl::_V1::handler&) const::{lambda(sycl::_V1::item<1, true>)#1}\n"];
// CHECK-NEXT: "0x[[#NODE1]]" -> "0x[[#NODE2]]"
// CHECK-NEXT: "0x[[#%x,NODE3:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE3]]\nTYPE = CGExecKernel \nNAME = typeinfo name for run_kernels<unsigned short>(sycl::_V1::queue, unsigned long, sycl::_V1::buffer<unsigned short, 1, sycl::_V1::detail::aligned_allocator<std::remove_const<unsigned short>::type>, void>, sycl::_V1::buffer<unsigned short, 1, sycl::_V1::detail::aligned_allocator<std::remove_const<unsigned short>::type>, void>, sycl::_V1::buffer<unsigned short, 1, sycl::_V1::detail::aligned_allocator<std::remove_const<unsigned short>::type>, void>)::{lambda(sycl::_V1::handler&)#4}::operator()(sycl::_V1::handler&) const::{lambda(sycl::_V1::item<1, true>)#1}\n"];
// CHECK-DAG: "0x[[#NODE2]]" -> "0x[[#NODE3]]"
// CHECK-DAG: "0x[[#%x,NODE7:]]" -> "0x[[#NODE3]]"
// CHECK-NEXT: "0x[[#%x,NODE4:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE4]]\nTYPE = CGCopy Device-to-Device \n"];
// CHECK-DAG: "0x[[#NODE3]]" -> "0x[[#NODE4]]"
// CHECK-DAG: "0x[[#NODE1]]" -> "0x[[#NODE4]]"
// CHECK-NEXT: "0x[[#%x,NODE5:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE5]]\nTYPE = CGCopy Device-to-Host \n"];
// CHECK-DAG: "0x[[#NODE3]]" -> "0x[[#NODE5]]"
// CHECK-NEXT: "0x[[#%x,NODE6:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE6]]\nTYPE = None \n"];
// CHECK-DAG: "0x[[#NODE5]]" -> "0x[[#NODE6]]"
// CHECK-NEXT: "0x[[#NODE7]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE7]]\nTYPE = CGExecKernel \nNAME = typeinfo name for run_kernels<unsigned short>(sycl::_V1::queue, unsigned long, sycl::_V1::buffer<unsigned short, 1, sycl::_V1::detail::aligned_allocator<std::remove_const<unsigned short>::type>, void>, sycl::_V1::buffer<unsigned short, 1, sycl::_V1::detail::aligned_allocator<std::remove_const<unsigned short>::type>, void>, sycl::_V1::buffer<unsigned short, 1, sycl::_V1::detail::aligned_allocator<std::remove_const<unsigned short>::type>, void>)::{lambda(sycl::_V1::handler&)#3}::operator()(sycl::_V1::handler&) const::{lambda(sycl::_V1::item<1, true>)#1}\n"];
// CHECK-DAG: "0x[[#NODE1]]" -> "0x[[#NODE7]]"

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/debug_print_graph.cpp"
