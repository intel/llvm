// RUN: %{build} -o %t.out
// RUN: %if linux && (level_zero || cuda) %{ %{run} %t.out ; FileCheck %s --input-file graph_verbose.dot %} %else %{ %{run} %t.out %}
// Windows output format differs from linux format.
// The filecheck-based output checking is suited to linux standards.
// On Windows, we only test that printing takes place correctly and does not
// trigger errors or throw exceptions.
//
//
// CHECK: digraph dot {
// CHECK-NEXT: "0x[[#%x,NODE1:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE1]]\nTYPE = CGExecKernel \nNAME = _ZTSZZ11run_kernelsItEN4sycl3_V15eventENS1_5queueEmNS1_6bufferIT_Li1ENS1_6detail17aligned_allocatorINSt12remove_constIS5_E4typeEEEvEESC_SC_ENKUlRNS1_7handlerEE_clESE_EUlNS1_4itemILi1ELb1EEEE_\n
// CHECK-SAME: ARGS = \n0) Type: Accessor Ptr: 0x[[#%x,ADDR1:]]\n1) Type: STD_Layout Ptr: 0x[[#%x,ADDR2:]]\n2) Type: STD_Layout Ptr: 0x[[#%x,ADDR3:]]\n
// CHECK-SAME: 3) Type: STD_Layout Ptr: 0x[[#ADDR1]]\n"];
// CHECK-NEXT: "0x[[#%x,NODE2:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE2]]\nTYPE = CGExecKernel \nNAME = _ZTSZZ11run_kernelsItEN4sycl3_V15eventENS1_5queueEmNS1_6bufferIT_Li1ENS1_6detail17aligned_allocatorINSt12remove_constIS5_E4typeEEEvEESC_SC_ENKUlRNS1_7handlerEE0_clESE_EUlNS1_4itemILi1ELb1EEEE_\n
// CHECK-SAME: ARGS = \n0) Type: Accessor Ptr: 0x[[#%x,ADDR4:]]\n1) Type: STD_Layout Ptr: 0x[[#%x,ADDR5:]]\n2) Type: STD_Layout Ptr: 0x[[#%x,ADDR6:]]\n
// CHECK-SAME: 3) Type: STD_Layout Ptr: 0x[[#ADDR4]]\n4) Type: Accessor Ptr: 0x[[#%x,ADDR7:]]\n5) Type: STD_Layout Ptr: 0x[[#%x,ADDR8:]]\n6) Type: STD_Layout Ptr: 0x[[#%x,ADDR9:]]\n7) Type: STD_Layout Ptr: 0x[[#%x,ADDR10:]]\n"];
// CHECK-NEXT: "0x[[#NODE1]]" -> "0x[[#NODE2]]"
// CHECK-NEXT: "0x[[#%x,NODE3:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE3]]\nTYPE = CGExecKernel \nNAME = _ZTSZZ11run_kernelsItEN4sycl3_V15eventENS1_5queueEmNS1_6bufferIT_Li1ENS1_6detail17aligned_allocatorINSt12remove_constIS5_E4typeEEEvEESC_SC_ENKUlRNS1_7handlerEE1_clESE_EUlNS1_4itemILi1ELb1EEEE_\n
// CHECK-SAME: ARGS = \n0) Type: Accessor Ptr: 0x[[#%x,ADDR11:]]\n1) Type: STD_Layout Ptr: 0x[[#%x,ADDR12:]]\n2) Type: STD_Layout Ptr: 0x[[#%x,ADDR13:]]\n
// CHECK-SAME: 3) Type: STD_Layout Ptr: 0x[[#ADDR11]]\n4) Type: Accessor Ptr: 0x[[#%x,ADDR14:]]\n5) Type: STD_Layout Ptr: 0x[[#%x,ADDR15:]]\n6) Type: STD_Layout Ptr: 0x[[#%x,ADDR16:]]\n7) Type: STD_Layout Ptr: 0x[[#%x,ADDR17:]]\n"];
// CHECK-NEXT: "0x[[#NODE2]]" -> "0x[[#NODE3]]"
// CHECK-NEXT: "0x[[#%x,NODE4:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE4]]\nTYPE = CGExecKernel \nNAME = _ZTSZZ11run_kernelsItEN4sycl3_V15eventENS1_5queueEmNS1_6bufferIT_Li1ENS1_6detail17aligned_allocatorINSt12remove_constIS5_E4typeEEEvEESC_SC_ENKUlRNS1_7handlerEE2_clESE_EUlNS1_4itemILi1ELb1EEEE_\n
// CHECK-SAME: ARGS = \n0) Type: Accessor Ptr: 0x[[#%x,ADDR18:]]\n1) Type: STD_Layout Ptr: 0x[[#%x,ADDR19:]]\n2) Type: STD_Layout Ptr: 0x[[#%x,ADDR20:]]\n
// CHECK-SAME: 3) Type: STD_Layout Ptr: 0x[[#ADDR18]]\n4) Type: Accessor Ptr: 0x[[#%x,ADDR21:]]\n5) Type: STD_Layout Ptr: 0x[[#%x,ADDR22:]]\n6) Type: STD_Layout Ptr: 0x[[#%x,ADDR23:]]\n7) Type: STD_Layout Ptr: 0x[[#%x,ADDR24:]]\n"];
// CHECK-DAG: "0x[[#NODE3]]" -> "0x[[#NODE4]]"
// CHECK-DAG: "0x[[#NODE2]]" -> "0x[[#NODE4]]"
// CHECK-NEXT: "0x[[#%x,NODE5:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE5]]\nTYPE = CGCopy Device-to-Device \nSrc: 0x[[#%x,ADDR25:]] Dst: 0x[[#%x,ADDR26:]]\n"];
// CHECK-DAG: "0x[[#NODE3]]" -> "0x[[#NODE5]]"
// CHECK-DAG: "0x[[#NODE4]]" -> "0x[[#NODE5]]
// CHECK-NEXT: "0x[[#%x,NODE6:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE6]]\nTYPE = CGCopy Device-to-Host \nSrc: 0x[[#%x,ADDR27:]] Dst: 0x[[#%x,ADDR28:]]\n"];
// CHECK-DAG: "0x[[#NODE4]]" -> "0x[[#NODE6]]"
// CHECK-NEXT: "0x[[#%x,NODE7:]]"
// CHECK-SAME: [style=bold, label="ID = 0x[[#NODE7]]\nTYPE = None \n"];
// CHECK-DAG: "0x[[#NODE6]]" -> "0x[[#NODE7]]"

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/debug_print_graph_verbose.cpp"
