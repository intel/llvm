// Check whether debug information for system header functions vector::insert and vector::pop_back are generated.

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// only vector::insert is in the user source
// RUN: %clang --target=x86_64-unknown-linux -emit-llvm -S -g %s -o  %t.default
// RUN: %clang --target=x86_64-unknown-linux -emit-llvm -S -g %s -o  %t.no_system_debug                 -fno-system-debug
// RUN: %clang --target=x86_64-unknown-linux -emit-llvm -S -g %s -o  %t.standalone_debug                -fstandalone-debug
// RUN: %clang --target=x86_64-unknown-linux -emit-llvm -S -g %s -o  %t.no_eliminate_unused_debug_types -fno-eliminate-unused-debug-types

// RUN: grep DISubprogram %t.default                         | FileCheck --check-prefix=CHECK-DEFAULT                         %s
// RUN: grep DISubprogram %t.no_system_debug                 | FileCheck --check-prefix=CHECK-NO-SYSTEM-DEBUG                 %s                
// RUN: grep DISubprogram %t.standalone_debug                | FileCheck --check-prefix=CHECK-STANDALONE-DEBUG                %s
// RUN: grep DISubprogram %t.no_eliminate_unused_debug_types | FileCheck --check-prefix=CHECK-NO-ELIMINATE-UNUSED-DEBUG-TYPES %s

// default generates debug info only for referenced declarations (i.e. vector::insert)
// CHECK-DEFAULT-NOT:                                  _ZNSt6vectorIiSaIiEE8pop_backEv
// CHECK-DEFAULT:                                      _ZNSt6vectorIiSaIiEE6insertEN9__gnu_cxx17__normal_iteratorIPKiS1_EEOi

// -fno-system-debug disables generation for both
// CHECK-NO-SYSTEM_DEBUG-NOT:                          _ZNSt6vectorIiSaIiEE8pop_backEv
// CHECK-NO-SYSTEM-DEBUG-NOT:                          _ZNSt6vectorIiSaIiEE6insertEN9__gnu_cxx17__normal_iteratorIPKiS1_EEOi

// -fstandalone-debug generates more debug info
// (i.e. declarations for system headers generate debug info)
// CHECK-STANDALONE-DEBUG:                             _ZNSt6vectorIiSaIiEE8pop_backEv
// CHECK-STANDALONE-DEBUG:                             _ZNSt6vectorIiSaIiEE6insertEN9__gnu_cxx17__normal_iteratorIPKiS1_EEOi

// -fno-eliminate-unused-debug-types generates even more debug info
// (i.e. declarations for system headers generate debug info)
// CHECK-NO-ELIMINATE-UNUSED-DEBUG-TYPES:              _ZNSt6vectorIiSaIiEE8pop_backEv
// CHECK-NO-ELIMINATE-UNUSED-DEBUG-TYPES:              _ZNSt6vectorIiSaIiEE6insertEN9__gnu_cxx17__normal_iteratorIPKiS1_EEOi

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// vector::pop_back and vector::insert are both in user source
// RUN: %clang -emit-llvm -S -g %s -o  %t.default                                                           -DPOP_BACK
// RUN: %clang -emit-llvm -S -g %s -o  %t.no_system_debug                 -fno-system-debug                 -DPOP_BACK
// RUN: %clang -emit-llvm -S -g %s -o  %t.standalone_debug                -fstandalone-debug                -DPOP_BACK
// RUN: %clang -emit-llvm -S -g %s -o  %t.no_eliminate_unused_debug_types -fno-eliminate-unused-debug-types -DPOP_BACK

// RUN: grep DISubprogram %t.default                         | FileCheck --check-prefix=CHECK-POP-BACK-DEFAULT                         %s
// RUN: grep DISubprogram %t.no_system_debug                 | FileCheck --check-prefix=CHECK-POP-BACK-NO-SYSTEM-DEBUG                 %s                
// RUN: grep DISubprogram %t.standalone_debug                | FileCheck --check-prefix=CHECK-POP-BACK-STANDALONE-DEBUG                %s
// RUN: grep DISubprogram %t.no_eliminate_unused_debug_types | FileCheck --check-prefix=CHECK-POP-BACK-NO-ELIMINATE-UNUSED-DEBUG-TYPES %s

// default generates debug info for referenced declarations (i.e. vector::insert and vector::pop_back)
// CHECK-POP-BACK-DEFAULT:                                      _ZNSt6vectorIiSaIiEE8pop_backEv
// CHECK-POP-BACK-DEFAULT:                                      _ZNSt6vectorIiSaIiEE6insertEN9__gnu_cxx17__normal_iteratorIPKiS1_EEOi

// -fno-system-debug disables generation for both
// CHECK-POP-BACK-NO-SYSTEM_DEBUG-NOT:                          _ZNSt6vectorIiSaIiEE8pop_backEv
// CHECK-POP-BACK-NO-SYSTEM-DEBUG-NOT:                          _ZNSt6vectorIiSaIiEE6insertEN9__gnu_cxx17__normal_iteratorIPKiS1_EEOi

// -fstandalone-debug generates more debug info
// (i.e. declarations for system headers generate debug info)
// CHECK-POP-BACK-STANDALONE-DEBUG:                             _ZNSt6vectorIiSaIiEE8pop_backEv
// CHECK-POP-BACK-STANDALONE-DEBUG:                             _ZNSt6vectorIiSaIiEE6insertEN9__gnu_cxx17__normal_iteratorIPKiS1_EEOi

// -fno-eliminate-unused-debug-types generates even more debug info
// (i.e. declarations for system headers generate debug info)
// CHECK-POP-BACK-NO-ELIMINATE-UNUSED-DEBUG-TYPES:              _ZNSt6vectorIiSaIiEE8pop_backEv
// CHECK-POP-BACK-NO-ELIMINATE-UNUSED-DEBUG-TYPES:              _ZNSt6vectorIiSaIiEE6insertEN9__gnu_cxx17__normal_iteratorIPKiS1_EEOi

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>

int main() {
  std::vector<int> a;

  a.insert(a.begin(),2);
#ifdef POP_BACK
  a.pop_back();
#endif
}
