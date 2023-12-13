// Check whether debug information for system header functions vector::insert and vector::pop_back are generated
// for a test program that includes <vector>.
//
// This testcase tests four option settings:
// By default                                                                              debug info for vector::insert and vector::pop_back is only generated if they are referenced
// When -fno-system-debug is used                                                          debug info for vector::insert and vector::pop_back is NOT generated
// When -fstandalone-debug is used more debug info is generated and                        debug info for vector::insert and vector::pop_back is generated
// When -fno-eliminate-unused-debug-types is used even more debug info is generated and    debug info for vector::insert and vector::pop_back is generated

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// only vector::insert is in the user source
//
// RUN: %clang -emit-llvm -S -g %s -o  %t.default                                                           -DINSERT
// RUN: %clang -emit-llvm -S -g %s -o  %t.no_system_debug                 -fno-system-debug                 -DINSERT
// RUN: %clang -emit-llvm -S -g %s -o  %t.standalone_debug                -fstandalone-debug                -DINSERT
// RUN: %clang -emit-llvm -S -g %s -o  %t.no_eliminate_unused_debug_types -fno-eliminate-unused-debug-types -DINSERT

// RUN: grep DISubprogram %t.default                         | FileCheck %if system-windows %{ --check-prefix=CHECK-WIN-INSERT-ONLY-DEBUG %} %else %{ --check-prefix=CHECK-INSERT-ONLY-DEBUG %} %s
// RUN: grep DISubprogram %t.no_system_debug                 | FileCheck %if system-windows %{ --check-prefix=CHECK-WIN-NO-DEBUG  %}         %else %{ --check-prefix=CHECK-NO-DEBUG  %}         %s
// RUN: grep DISubprogram %t.standalone_debug                | FileCheck %if system-windows %{ --check-prefix=CHECK-WIN-ALL-DEBUG %}         %else %{ --check-prefix=CHECK-ALL-DEBUG %}         %s
// RUN: grep DISubprogram %t.no_eliminate_unused_debug_types | FileCheck %if system-windows %{ --check-prefix=CHECK-WIN-ALL-DEBUG %}         %else %{ --check-prefix=CHECK-ALL-DEBUG %}         %s

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// vector::pop_back and vector::insert are both in user source
//
// RUN: %clang -emit-llvm -S -g %s -o  %t.default                                                           -DINSERT -DPOP_BACK
// RUN: %clang -emit-llvm -S -g %s -o  %t.no_system_debug                 -fno-system-debug                 -DINSERT -DPOP_BACK
// RUN: %clang -emit-llvm -S -g %s -o  %t.standalone_debug                -fstandalone-debug                -DINSERT -DPOP_BACK
// RUN: %clang -emit-llvm -S -g %s -o  %t.no_eliminate_unused_debug_types -fno-eliminate-unused-debug-types -DINSERT -DPOP_BACK

// RUN: grep DISubprogram %t.default                         | FileCheck %if system-windows %{ --check-prefix=CHECK-WIN-ALL-DEBUG %}         %else %{ --check-prefix=CHECK-ALL-DEBUG %}         %s
// RUN: grep DISubprogram %t.no_system_debug                 | FileCheck %if system-windows %{ --check-prefix=CHECK-WIN-NO-DEBUG  %}         %else %{ --check-prefix=CHECK-NO-DEBUG  %}         %s
// RUN: grep DISubprogram %t.standalone_debug                | FileCheck %if system-windows %{ --check-prefix=CHECK-WIN-ALL-DEBUG %}         %else %{ --check-prefix=CHECK-ALL-DEBUG %}         %s
// RUN: grep DISubprogram %t.no_eliminate_unused_debug_types | FileCheck %if system-windows %{ --check-prefix=CHECK-WIN-ALL-DEBUG %}         %else %{ --check-prefix=CHECK-ALL-DEBUG %}         %s

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// class vector<int> is explicitly instantiated.
// vector::pop_back and vector::insert are NOT in user source
//
// RUN: %clang -emit-llvm -S -g %s -o  %t.default                                                           -DEXPLICIT_INSTANTIATION
// RUN: %clang -emit-llvm -S -g %s -o  %t.no_system_debug                 -fno-system-debug                 -DEXPLICIT_INSTANTIATION
// RUN: %clang -emit-llvm -S -g %s -o  %t.standalone_debug                -fstandalone-debug                -DEXPLICIT_INSTANTIATION
// RUN: %clang -emit-llvm -S -g %s -o  %t.no_eliminate_unused_debug_types -fno-eliminate-unused-debug-types -DEXPLICIT_INSTANTIATION

// RUN: grep DISubprogram %t.default                         | FileCheck %if system-windows %{ --check-prefix=CHECK-WIN-ALL-DEBUG %}         %else %{ --check-prefix=CHECK-ALL-DEBUG %}         %s
// RUN: grep DISubprogram %t.no_system_debug                 | FileCheck %if system-windows %{ --check-prefix=CHECK-WIN-NO-DEBUG  %}         %else %{ --check-prefix=CHECK-NO-DEBUG  %}         %s
// RUN: grep DISubprogram %t.standalone_debug                | FileCheck %if system-windows %{ --check-prefix=CHECK-WIN-ALL-DEBUG %}         %else %{ --check-prefix=CHECK-ALL-DEBUG %}         %s
// RUN: grep DISubprogram %t.no_eliminate_unused_debug_types | FileCheck %if system-windows %{ --check-prefix=CHECK-WIN-ALL-DEBUG %}         %else %{ --check-prefix=CHECK-ALL-DEBUG %}         %s

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// class vector<int> is explicitly instantiated.
// vector::pop_back and vector::insert are BOTH in user source
//
// RUN: %clang -emit-llvm -S -g %s -o  %t.default                                                           -DEXPLICIT_INSTANTIATION -DINSERT -DPOP_BACK
// RUN: %clang -emit-llvm -S -g %s -o  %t.no_system_debug                 -fno-system-debug                 -DEXPLICIT_INSTANTIATION -DINSERT -DPOP_BACK
// RUN: %clang -emit-llvm -S -g %s -o  %t.standalone_debug                -fstandalone-debug                -DEXPLICIT_INSTANTIATION -DINSERT -DPOP_BACK
// RUN: %clang -emit-llvm -S -g %s -o  %t.no_eliminate_unused_debug_types -fno-eliminate-unused-debug-types -DEXPLICIT_INSTANTIATION -DINSERT -DPOP_BACK

// RUN: grep DISubprogram %t.default                         | FileCheck %if system-windows %{ --check-prefix=CHECK-WIN-ALL-DEBUG %}         %else %{ --check-prefix=CHECK-ALL-DEBUG %}         %s
// RUN: grep DISubprogram %t.no_system_debug                 | FileCheck %if system-windows %{ --check-prefix=CHECK-WIN-NO-DEBUG  %}         %else %{ --check-prefix=CHECK-NO-DEBUG  %}         %s
// RUN: grep DISubprogram %t.standalone_debug                | FileCheck %if system-windows %{ --check-prefix=CHECK-WIN-ALL-DEBUG %}         %else %{ --check-prefix=CHECK-ALL-DEBUG %}         %s
// RUN: grep DISubprogram %t.no_eliminate_unused_debug_types | FileCheck %if system-windows %{ --check-prefix=CHECK-WIN-ALL-DEBUG %}         %else %{ --check-prefix=CHECK-ALL-DEBUG %}         %s

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// vector::pop_back and vector::insert must both not be found
// CHECK-NO-DEBUG-NOT:                                   _ZNSt6vectorIiSaIiEE8pop_backEv
// CHECK-NO-DEBUG-NOT:                                   _ZNSt6vectorIiSaIiEE6insertEN9__gnu_cxx17__normal_iteratorIPKiS1_EEOi

// only vector::insert must be found
// CHECK-INSERT-ONLY-DEBUG-NOT:                          _ZNSt6vectorIiSaIiEE8pop_backEv
// CHECK-INSERT-ONLY-DEBUG:                              _ZNSt6vectorIiSaIiEE6insertEN9__gnu_cxx17__normal_iteratorIPKiS1_EEOi

// vector::pop_back and vector::insert must both be found
// CHECK-ALL-DEBUG:                                      _ZNSt6vectorIiSaIiEE8pop_backEv
// CHECK-ALL-DEBUG:                                      _ZNSt6vectorIiSaIiEE6insertEN9__gnu_cxx17__normal_iteratorIPKiS1_EEOi

// vector::pop_back and vector::insert must both not be found
// CHECK-WIN-NO-DEBUG-NOT:                               ?pop_back@?$vector@HV?$allocator@H@std@@@std@@QEAAXXZ
// CHECK-WIN-NO-DEBUG-NOT:                               ?insert@?$vector@HV?$allocator@H@std@@@std@@QEAA?AV?$_Vector_iterator@V?$_Vector_val@U?$_Simple_types@H@std@@@std@@@2@V?$_Vector_const_iterator@V?$_Vector_val@U?$_Simple_types@H@std@@@std@@@2@$$QEAH@Z

// only vector::insert must be found
// CHECK-WIN-INSERT-ONLY-DEBUG-NOT:                      ?pop_back@?$vector@HV?$allocator@H@std@@@std@@QEAAXXZ
// CHECK-WIN-INSERT-ONLY-DEBUG:                          ?insert@?$vector@HV?$allocator@H@std@@@std@@QEAA?AV?$_Vector_iterator@V?$_Vector_val@U?$_Simple_types@H@std@@@std@@@2@V?$_Vector_const_iterator@V?$_Vector_val@U?$_Simple_types@H@std@@@std@@@2@$$QEAH@Z

// vector::pop_back and vector::insert must both be found
// CHECK-WIN-ALL-DEBUG:                                  ?pop_back@?$vector@HV?$allocator@H@std@@@std@@QEAAXXZ
// CHECK-WIN-ALL-DEBUG:                                  ?insert@?$vector@HV?$allocator@H@std@@@std@@QEAA?AV?$_Vector_iterator@V?$_Vector_val@U?$_Simple_types@H@std@@@std@@@2@V?$_Vector_const_iterator@V?$_Vector_val@U?$_Simple_types@H@std@@@std@@@2@$$QEAH@Z

#include <vector>

#ifdef EXPLICIT_INSTANTIATION
template class std::vector<int>;
#endif

int main() {
#if defined(INSERT) || defined(POP_BACK)
  std::vector<int> a;
#endif
  
#ifdef INSERT
  a.insert(a.begin(),2);
#endif
#ifdef POP_BACK  
  a.pop_back();
#endif
}
