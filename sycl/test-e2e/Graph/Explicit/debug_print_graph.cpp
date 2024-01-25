// RUN: %{build} -o %t.out
// RUN: %if linux && (level_zero || cuda) %{ %{run} %t.out ; FileCheck %s --input-file graph.dot %} %else %{ %{run} %t.out %}
// Windows output format differs from linux format.
// The filecheck-based output checking is suited to linux standards.
// On Windows, we only test that printing takes place correctly and does not
// trigger errors or throw exceptions.
//
// CHECK: digraph dot {
// CHECK-NEXT: "0x[[#%x,NODE1:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE1]]\nTYPE = CGExecKernel \nNAME = _ZTSZZ11add_kernelsItESt6vectorIN4sycl3_V13ext6oneapi12experimental4nodeESaIS6_EENS5_13command_graphILNS5_11graph_stateE0EEEmNS2_6bufferIT_Li1ENS2_6detail17aligned_allocatorINSt12remove_constISD_E4typeEEEvEESK_SK_ENKUlRNS2_7handlerEE_clESM_EUlNS2_4itemILi1ELb1EEEE_\n"];
// CHECK-NEXT: "0x[[#%x,NODE2:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE2]]\nTYPE = CGExecKernel \nNAME = _ZTSZZ11add_kernelsItESt6vectorIN4sycl3_V13ext6oneapi12experimental4nodeESaIS6_EENS5_13command_graphILNS5_11graph_stateE0EEEmNS2_6bufferIT_Li1ENS2_6detail17aligned_allocatorINSt12remove_constISD_E4typeEEEvEESK_SK_ENKUlRNS2_7handlerEE0_clESM_EUlNS2_4itemILi1ELb1EEEE_\n"];
// CHECK-NEXT: "0x[[#NODE1]]" -> "0x[[#NODE2]]"
// CHECK-NEXT: "0x[[#%x,NODE3:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE3]]\nTYPE = CGExecKernel \nNAME = _ZTSZZ11add_kernelsItESt6vectorIN4sycl3_V13ext6oneapi12experimental4nodeESaIS6_EENS5_13command_graphILNS5_11graph_stateE0EEEmNS2_6bufferIT_Li1ENS2_6detail17aligned_allocatorINSt12remove_constISD_E4typeEEEvEESK_SK_ENKUlRNS2_7handlerEE1_clESM_EUlNS2_4itemILi1ELb1EEEE_\n"];
// CHECK-NEXT: "0x[[#NODE2]]" -> "0x[[#NODE3]]"
// CHECK-NEXT: "0x[[#%x,NODE4:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE4]]\nTYPE = CGExecKernel \nNAME = _ZTSZZ11add_kernelsItESt6vectorIN4sycl3_V13ext6oneapi12experimental4nodeESaIS6_EENS5_13command_graphILNS5_11graph_stateE0EEEmNS2_6bufferIT_Li1ENS2_6detail17aligned_allocatorINSt12remove_constISD_E4typeEEEvEESK_SK_ENKUlRNS2_7handlerEE2_clESM_EUlNS2_4itemILi1ELb1EEEE_\n"];
// CHECK-DAG: "0x[[#NODE3]]" -> "0x[[#NODE4]]"
// CHECK-DAG: "0x[[#NODE2]]" -> "0x[[#NODE4]]"
// CHECK-NEXT: "0x[[#%x,NODE5:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE5]]\nTYPE = CGCopy Device-to-Device \n"];
// CHECK-DAG: "0x[[#NODE3]]" -> "0x[[#NODE5]]"
// CHECK-DAG: "0x[[#NODE4]]" -> "0x[[#NODE5]]
// CHECK-NEXT: "0x[[#%x,NODE6:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE6]]\nTYPE = CGCopy Device-to-Host \n"];
// CHECK-DAG: "0x[[#NODE4]]" -> "0x[[#NODE6]]"
// CHECK-NEXT: "0x[[#%x,NODE7:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE7]]\nTYPE = None \n"];
// CHECK-DAG: "0x[[#NODE6]]" -> "0x[[#NODE7]]"

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/debug_print_graph.cpp"
