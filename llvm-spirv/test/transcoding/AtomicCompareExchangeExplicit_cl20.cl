// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
// RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r --spirv-target-env=CL2.0 %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

// RUN: llvm-spirv %t.bc -spirv-text -o %t.txt --spirv-ext=+SPV_KHR_untyped_pointers
// RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_KHR_untyped_pointers
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r --spirv-target-env=CL2.0 %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

#define DEFINE_KERNEL(TYPE)                                                            \
__kernel void testAtomicCompareExchangeExplicit_cl20_##TYPE(                           \
    volatile global atomic_##TYPE* object,                                             \
    global TYPE* expected,                                                             \
    TYPE desired)                                                                      \
{                                                                                      \
  /* Values of memory order and memory scope arguments correspond to SPIR-2.0 spec. */ \
  atomic_compare_exchange_strong_explicit(object, expected, desired,                   \
                                          memory_order_release, /* 3 */                \
                                          memory_order_relaxed  /* 0 */                \
                                         ); /* by default, assume device scope = 2 */  \
  atomic_compare_exchange_strong_explicit(object, expected, desired,                   \
                                          memory_order_acq_rel,   /* 4 */              \
                                          memory_order_relaxed,   /* 0 */              \
                                          memory_scope_work_group /* 1 */              \
                                         );                                            \
  atomic_compare_exchange_weak_explicit(object, expected, desired,                     \
                                        memory_order_release, /* 3 */                  \
                                        memory_order_relaxed  /* 0 */                  \
                                         ); /* by default, assume device scope = 2 */  \
  atomic_compare_exchange_weak_explicit(object, expected, desired,                     \
                                        memory_order_acq_rel,   /* 4 */                \
                                        memory_order_relaxed,   /* 0 */                \
                                        memory_scope_work_group /* 1 */                \
                                       );                                              \
}

DEFINE_KERNEL(int)
DEFINE_KERNEL(float)
DEFINE_KERNEL(double)

//CHECK-SPIRV: TypeInt [[int32:[0-9]+]] 32 0
//CHECK-SPIRV: TypeInt [[int64:[0-9]+]] 64 0
//; Constants below correspond to the SPIR-V spec
//CHECK-SPIRV-DAG: Constant [[int32]] [[DeviceScope:[0-9]+]] 1
//CHECK-SPIRV-DAG: Constant [[int32]] [[WorkgroupScope:[0-9]+]] 2
//CHECK-SPIRV-DAG: Constant [[int32]] [[ReleaseMemSem:[0-9]+]] 4
//CHECK-SPIRV-DAG: Constant [[int32]] [[RelaxedMemSem:[0-9]+]] 0
//CHECK-SPIRV-DAG: Constant [[int32]] [[AcqRelMemSem:[0-9]+]] 8

//CHECK-SPIRV: AtomicCompareExchange [[int32]] {{[0-9]+}} {{[0-9]+}} [[DeviceScope]] [[ReleaseMemSem]] [[RelaxedMemSem]]
//CHECK-SPIRV: AtomicCompareExchange [[int32]] {{[0-9]+}} {{[0-9]+}} [[WorkgroupScope]] [[AcqRelMemSem]] [[RelaxedMemSem]]
//CHECK-SPIRV: AtomicCompareExchange [[int32]] {{[0-9]+}} {{[0-9]+}} [[DeviceScope]] [[ReleaseMemSem]] [[RelaxedMemSem]]
//CHECK-SPIRV: AtomicCompareExchange [[int32]] {{[0-9]+}} {{[0-9]+}} [[WorkgroupScope]] [[AcqRelMemSem]] [[RelaxedMemSem]]

//CHECK-SPIRV: AtomicCompareExchange [[int32]] {{[0-9]+}} {{[0-9]+}} [[DeviceScope]] [[ReleaseMemSem]] [[RelaxedMemSem]]
//CHECK-SPIRV: AtomicCompareExchange [[int32]] {{[0-9]+}} {{[0-9]+}} [[WorkgroupScope]] [[AcqRelMemSem]] [[RelaxedMemSem]]
//CHECK-SPIRV: AtomicCompareExchange [[int32]] {{[0-9]+}} {{[0-9]+}} [[DeviceScope]] [[ReleaseMemSem]] [[RelaxedMemSem]]
//CHECK-SPIRV: AtomicCompareExchange [[int32]] {{[0-9]+}} {{[0-9]+}} [[WorkgroupScope]] [[AcqRelMemSem]] [[RelaxedMemSem]]

//CHECK-SPIRV: AtomicCompareExchange [[int64]] {{[0-9]+}} {{[0-9]+}} [[DeviceScope]] [[ReleaseMemSem]] [[RelaxedMemSem]]
//CHECK-SPIRV: AtomicCompareExchange [[int64]] {{[0-9]+}} {{[0-9]+}} [[WorkgroupScope]] [[AcqRelMemSem]] [[RelaxedMemSem]]
//CHECK-SPIRV: AtomicCompareExchange [[int64]] {{[0-9]+}} {{[0-9]+}} [[DeviceScope]] [[ReleaseMemSem]] [[RelaxedMemSem]]
//CHECK-SPIRV: AtomicCompareExchange [[int64]] {{[0-9]+}} {{[0-9]+}} [[WorkgroupScope]] [[AcqRelMemSem]] [[RelaxedMemSem]]

//CHECK-LLVM-LABEL: define spir_kernel void @testAtomicCompareExchangeExplicit_cl20_int(
//CHECK-LLVM: call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(ptr addrspace(4) %0, ptr addrspace(4) %expected{{.*}}, i32 %desired, i32 3, i32 0, i32 2)
//CHECK-LLVM: call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(ptr addrspace(4) %0, ptr addrspace(4) %expected{{.*}}, i32 %desired, i32 4, i32 0, i32 1)
//CHECK-LLVM: call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(ptr addrspace(4) %0, ptr addrspace(4) %expected{{.*}}, i32 %desired, i32 3, i32 0, i32 2)
//CHECK-LLVM: call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(ptr addrspace(4) %0, ptr addrspace(4) %expected{{.*}}, i32 %desired, i32 4, i32 0, i32 1)

//CHECK-LLVM-LABEL: define spir_kernel void @testAtomicCompareExchangeExplicit_cl20_float(
//CHECK-LLVM: [[OBJECT:%[0-9]+]] = addrspacecast ptr addrspace(1) %object to ptr addrspace(4)
//CHECK-LLVM: [[EXPECTED:%[0-9]+]] = addrspacecast ptr addrspace(1) %expected to ptr addrspace(4)
//CHECK-LLVM: [[CAST1:%[0-9]+]] = bitcast float %desired to i32
//CHECK-LLVM: %exp = load i32, ptr addrspace(4) [[EXPECTED]], align 4
//CHECK-LLVM: store i32 %exp, ptr [[EXPECTED_ALLOCA:%expected[0-9]+]], align 4
//CHECK-LLVM: [[EXPECTED_AS1:%expected.*]] = addrspacecast ptr [[EXPECTED_ALLOCA]] to ptr addrspace(4)
//CHECK-LLVM: call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(ptr addrspace(4) [[OBJECT]], ptr addrspace(4) [[EXPECTED_AS1]], i32 [[CAST1]], i32 3, i32 0, i32 2)
//CHECK-LLVM: [[CAST2:%[0-9]+]] = bitcast float %desired to i32
//CHECK-LLVM: [[LOAD2:%exp[0-9]+]] = load i32, ptr addrspace(4) [[EXPECTED]], align 4
//CHECK-LLVM: store i32 [[LOAD2]], ptr [[EXPECTED_ALLOCA2:%expected[0-9]+]], align 4
//CHECK-LLVM: [[EXPECTED_AS2:%expected.*]] = addrspacecast ptr [[EXPECTED_ALLOCA2]] to ptr addrspace(4)
//CHECK-LLVM: call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(ptr addrspace(4) [[OBJECT]], ptr addrspace(4) [[EXPECTED_AS2]], i32 [[CAST2]], i32 4, i32 0, i32 1)
//CHECK-LLVM: [[CAST3:%[0-9]+]] = bitcast float %desired to i32
//CHECK-LLVM: [[LOAD3:%exp[0-9]+]] = load i32, ptr addrspace(4) [[EXPECTED]], align 4
//CHECK-LLVM: store i32 [[LOAD3]], ptr [[EXPECTED_ALLOCA3:%expected[0-9]+]], align 4
//CHECK-LLVM: [[EXPECTED_AS3:%expected.*]] = addrspacecast ptr [[EXPECTED_ALLOCA3]] to ptr addrspace(4)
//CHECK-LLVM: call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(ptr addrspace(4) [[OBJECT]], ptr addrspace(4) [[EXPECTED_AS3]], i32 [[CAST3]], i32 3, i32 0, i32 2)
//CHECK-LLVM: [[CAST4:%[0-9]+]] = bitcast float %desired to i32
//CHECK-LLVM: [[LOAD4:%exp[0-9]+]] = load i32, ptr addrspace(4) [[EXPECTED]], align 4
//CHECK-LLVM: store i32 [[LOAD4]], ptr [[EXPECTED_ALLOCA4:%expected[0-9]+]], align 4
//CHECK-LLVM: [[EXPECTED_AS4:%expected.*]] = addrspacecast ptr [[EXPECTED_ALLOCA4]] to ptr addrspace(4)
//CHECK-LLVM: call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(ptr addrspace(4) [[OBJECT]], ptr addrspace(4) [[EXPECTED_AS4]], i32 [[CAST4]], i32 4, i32 0, i32 1)


//CHECK-LLVM-LABEL: define spir_kernel void @testAtomicCompareExchangeExplicit_cl20_double(
//CHECK-LLVM: [[OBJECT:%[0-9]+]] = addrspacecast ptr addrspace(1) %object to ptr addrspace(4)
//CHECK-LLVM: [[EXPECTED:%[0-9]+]] = addrspacecast ptr addrspace(1) %expected to ptr addrspace(4)
//CHECK-LLVM: [[CAST1:%[0-9]+]] = bitcast double %desired to i64
//CHECK-LLVM: %exp = load i64, ptr addrspace(4) [[EXPECTED]], align 8
//CHECK-LLVM: store i64 %exp, ptr [[EXPECTED_ALLOCA:%expected[0-9]+]], align 8
//CHECK-LLVM: [[EXPECTED_AS1:%expected.*]] = addrspacecast ptr [[EXPECTED_ALLOCA]] to ptr addrspace(4)
//CHECK-LLVM: call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiclPU3AS4ll12memory_orderS4_12memory_scope(ptr addrspace(4) [[OBJECT]], ptr addrspace(4) [[EXPECTED_AS1]], i64 [[CAST1]], i32 3, i32 0, i32 2)
//CHECK-LLVM: [[CAST2:%[0-9]+]] = bitcast double %desired to i64
//CHECK-LLVM: [[LOAD2:%exp[0-9]+]] = load i64, ptr addrspace(4) [[EXPECTED]], align 8
//CHECK-LLVM: store i64 [[LOAD2]], ptr [[EXPECTED_ALLOCA2:%expected[0-9]+]], align 8
//CHECK-LLVM: [[EXPECTED_AS2:%expected.*]] = addrspacecast ptr [[EXPECTED_ALLOCA2]] to ptr addrspace(4)
//CHECK-LLVM: call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiclPU3AS4ll12memory_orderS4_12memory_scope(ptr addrspace(4) [[OBJECT]], ptr addrspace(4) [[EXPECTED_AS2]], i64 [[CAST2]], i32 4, i32 0, i32 1)
//CHECK-LLVM: [[CAST3:%[0-9]+]] = bitcast double %desired to i64
//CHECK-LLVM: [[LOAD3:%exp[0-9]+]] = load i64, ptr addrspace(4) [[EXPECTED]], align 8
//CHECK-LLVM: store i64 [[LOAD3]], ptr [[EXPECTED_ALLOCA3:%expected[0-9]+]], align 8
//CHECK-LLVM: [[EXPECTED_AS3:%expected.*]] = addrspacecast ptr [[EXPECTED_ALLOCA3]] to ptr addrspace(4)
//CHECK-LLVM: call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiclPU3AS4ll12memory_orderS4_12memory_scope(ptr addrspace(4) [[OBJECT]], ptr addrspace(4) [[EXPECTED_AS3]], i64 [[CAST3]], i32 3, i32 0, i32 2)
//CHECK-LLVM: [[CAST4:%[0-9]+]] = bitcast double %desired to i64
//CHECK-LLVM: [[LOAD4:%exp[0-9]+]] = load i64, ptr addrspace(4) [[EXPECTED]], align 8
//CHECK-LLVM: store i64 [[LOAD4]], ptr [[EXPECTED_ALLOCA4:%expected[0-9]+]], align 8
//CHECK-LLVM: [[EXPECTED_AS4:%expected.*]] = addrspacecast ptr [[EXPECTED_ALLOCA4]] to ptr addrspace(4)
//CHECK-LLVM: call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiclPU3AS4ll12memory_orderS4_12memory_scope(ptr addrspace(4) [[OBJECT]], ptr addrspace(4) [[EXPECTED_AS4]], i64 [[CAST4]], i32 4, i32 0, i32 1)
