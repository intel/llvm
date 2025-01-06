// RUN: grep -r "UNSUPPORTED:.*!" %S/../../test-e2e \
// RUN: --include=*.cpp --no-group-separator > %t.unsupported
// RUN: cat %t.unsupported | wc -l | FileCheck %s --check-prefix UNSUPPORTED-WITH-NEGATIONS
// RUN: cat %t.unsupported | sed 's/\.cpp.*/.cpp/' | sort | FileCheck %s --check-prefix UNSUPPORTED-CHECK
//
// RUN: not grep -r "REQUIRES:.*!" %S/../../test-e2e
//
// UNSUPPORTED-WITH-NEGATIONS: 3 
//
// UNSUPPORTED-CHECK: Basic/accessor/host_task_accessor_deduction.cpp
// UNSUPPORTED-CHECK-NEXT: ESIMD/named_barriers/loop.cpp
// UNSUPPORTED-CHECK-NEXT: GroupAlgorithm/reduce_sycl2020.cpp
