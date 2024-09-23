// RUN: %{build} -o %t.out
// RUN: %if linux && (level_zero || cuda) %{ %{run} %t.out ; FileCheck %s --implicit-check-not=LEAK --input-file graph.dot %} %else %{ %{run} %t.out %}
// Windows output format differs from linux format.
// The filecheck-based output checking is suited to linux standards.
// On Windows, we only test that printing takes place correctly and does not
// trigger errors or throw exceptions.
//
// CHECK: digraph dot {
// CHECK-NEXT: "0x[[#%x,NODE1:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE1]]\nTYPE = CGExecKernel \nNAME = _ZTSZZ11run_kernelsItEN4sycl3_V15eventENS1_5queueEmNS1_6bufferIT_Li1ENS1_6detail17aligned_allocatorINSt12remove_constIS5_E4typeEEEvEESC_SC_ENKUlRNS1_7handlerEE_clESE_EUlNS1_4itemILi1ELb1EEEE_\n"];
// CHECK-NEXT: "0x[[#%x,NODE2:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE2]]\nTYPE = CGExecKernel \nNAME = _ZTSZZ11run_kernelsItEN4sycl3_V15eventENS1_5queueEmNS1_6bufferIT_Li1ENS1_6detail17aligned_allocatorINSt12remove_constIS5_E4typeEEEvEESC_SC_ENKUlRNS1_7handlerEE0_clESE_EUlNS1_4itemILi1ELb1EEEE_\n"];
// CHECK-NEXT: "0x[[#NODE1]]" -> "0x[[#NODE2]]"
// CHECK-NEXT: "0x[[#%x,NODE3:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE3]]\nTYPE = CGExecKernel \nNAME = _ZTSZZ11run_kernelsItEN4sycl3_V15eventENS1_5queueEmNS1_6bufferIT_Li1ENS1_6detail17aligned_allocatorINSt12remove_constIS5_E4typeEEEvEESC_SC_ENKUlRNS1_7handlerEE2_clESE_EUlNS1_4itemILi1ELb1EEEE_\n"];
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
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE7]]\nTYPE = CGExecKernel \nNAME = _ZTSZZ11run_kernelsItEN4sycl3_V15eventENS1_5queueEmNS1_6bufferIT_Li1ENS1_6detail17aligned_allocatorINSt12remove_constIS5_E4typeEEEvEESC_SC_ENKUlRNS1_7handlerEE1_clESE_EUlNS1_4itemILi1ELb1EEEE_\n"];
// CHECK-DAG: "0x[[#NODE1]]" -> "0x[[#NODE7]]"

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/debug_print_graph.cpp"
