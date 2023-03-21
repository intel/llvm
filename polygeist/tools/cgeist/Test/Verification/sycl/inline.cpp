// RUN: clang++ %s -fsycl -fsycl-device-only -O0 -w -fsycl-targets=spir64-unknown-unknown-syclmlir -mllvm -print-changed 2>&1 | FileCheck  %s

// COM: The alwaysinline function should be inlined even at O0

// CHECK: *** IR Dump At Start ***
// CHECK: define spir_func i32 @callee()
// CHECK:   ret i32 10
// CHECK: }

// CHECK: define spir_func i32 @caller()
// CHECK:   ret i32 10
// CHECK: }

// CHECK: *** IR Dump After AlwaysInlinerPass on [module] omitted because no change ***
// CHECK: *** IR Dump After CoroConditionalWrapper on [module] omitted because no change ***
// CHECK: *** IR Dump After AnnotationRemarksPass on callee omitted because no change ***
// CHECK: *** IR Dump After AnnotationRemarksPass on caller omitted because no change ***
// CHECK: *** IR Pass ModuleToFunctionPassAdaptor on [module] ignored ***

SYCL_EXTERNAL extern "C" int __attribute__((always_inline)) callee() { return 10; }
SYCL_EXTERNAL extern "C" int caller() { return callee(); }
