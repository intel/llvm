// RUN: %clang_cc1 -triple spir-unknown-unknown -O0 -cl-std=CL2.0 -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o %t.spv.txt
// RUN: FileCheck < %t.spv.txt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv --spirv-target-env CL2.0 -o %t.rev.bc
// RUN: llvm-dis %t.rev.bc
// RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM
// RUN: llvm-spirv -r %t.spv --spirv-target-env SPV-IR -o %t.rev.bc
// RUN: llvm-dis %t.rev.bc
// RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-SPV-IR

// CHECK-SPIRV: EntryPoint {{[0-9]+}} [[BlockKer1:[0-9]+]] "__device_side_enqueue_block_invoke_kernel"
// CHECK-SPIRV: EntryPoint {{[0-9]+}} [[BlockKer2:[0-9]+]] "__device_side_enqueue_block_invoke_2_kernel"
// CHECK-SPIRV: EntryPoint {{[0-9]+}} [[BlockKer3:[0-9]+]] "__device_side_enqueue_block_invoke_3_kernel"
// CHECK-SPIRV: EntryPoint {{[0-9]+}} [[BlockKer4:[0-9]+]] "__device_side_enqueue_block_invoke_4_kernel"
// CHECK-SPIRV: EntryPoint {{[0-9]+}} [[BlockKer5:[0-9]+]] "__device_side_enqueue_block_invoke_5_kernel"
// CHECK-SPIRV: Name [[BlockGlb1:[0-9]+]] "__block_literal_global"
// CHECK-SPIRV: Name [[BlockGlb2:[0-9]+]] "__block_literal_global.1"

// CHECK-SPIRV: TypeInt [[Int32Ty:[0-9]+]] 32
// CHECK-SPIRV: TypeInt [[Int8Ty:[0-9]+]] 8
// CHECK-SPIRV: Constant [[Int32Ty]] [[ConstInt0:[0-9]+]] 0
// CHECK-SPIRV: Constant [[Int32Ty]] [[ConstInt17:[0-9]+]] 21
// CHECK-SPIRV: Constant [[Int32Ty]] [[ConstInt2:[0-9]+]] 2
// CHECK-SPIRV: Constant [[Int32Ty]] [[ConstInt8:[0-9]+]] 8
// CHECK-SPIRV: Constant [[Int32Ty]] [[ConstInt20:[0-9]+]] 24

// CHECK-SPIRV: TypePointer [[Int8PtrGenTy:[0-9]+]] 8 [[Int8Ty]]
// CHECK-SPIRV: TypeVoid [[VoidTy:[0-9]+]]
// CHECK-SPIRV: TypePointer [[Int32LocPtrTy:[0-9]+]] 7 [[Int32Ty]]
// CHECK-SPIRV: TypeDeviceEvent [[EventTy:[0-9]+]]
// CHECK-SPIRV: TypePointer [[EventPtrTy:[0-9]+]] 8 [[EventTy]]
// CHECK-SPIRV: TypeFunction [[BlockTy1:[0-9]+]] [[VoidTy]] [[Int8PtrGenTy]]
// CHECK-SPIRV: TypeFunction [[BlockTy2:[0-9]+]] [[VoidTy]] [[Int8PtrGenTy]]
// CHECK-SPIRV: TypeFunction [[BlockTy3:[0-9]+]] [[VoidTy]] [[Int8PtrGenTy]]
// CHECK-SPIRV: ConstantNull [[EventPtrTy]] [[EventNull:[0-9]+]]

// CHECK-LLVM: [[BlockTy1:%[0-9a-z\.]+]] = type { i32, i32, ptr addrspace(4) }
// CHECK-LLVM: [[BlockTy2:%[0-9a-z\.]+]] = type <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, i8 }>
// CHECK-LLVM: [[BlockTy3:%[0-9a-z\.]+]] = type <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>

// CHECK-LLVM: @__block_literal_global = internal addrspace(1) constant [[BlockTy1]] { i32 12, i32 4, ptr addrspace(4) null }, align 4
// CHECK-LLVM: @__block_literal_global.1 = internal addrspace(1) constant [[BlockTy1]] { i32 12, i32 4, ptr addrspace(4) null }, align 4

typedef struct {int a;} ndrange_t;
#define NULL ((void*)0)

kernel void device_side_enqueue(global int *a, global int *b, int i, char c0) {
  queue_t default_queue;
  unsigned flags = 0;
  ndrange_t ndrange;
  clk_event_t clk_event;
  clk_event_t event_wait_list;
  clk_event_t event_wait_list2[] = {clk_event};

  // Emits block literal on stack and block kernel.

  // CHECK-SPIRV: Bitcast [[Int8PtrGenTy]] [[BlockLit1:[0-9]+]]
  // CHECK-SPIRV: EnqueueKernel [[Int32Ty]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  //                            [[ConstInt0]] [[EventNull]] [[EventNull]]
  //                            [[BlockKer1]] [[BlockLit1]] [[ConstInt17]] [[ConstInt8]]

  // CHECK-LLVM: [[Block2:%[0-9]+]] = addrspacecast ptr %block to ptr addrspace(4)
  // CHECK-LLVM: [[Block2Ptr:%[0-9]+]] = bitcast ptr addrspace(4) [[Block2]] to ptr addrspace(4)
  // CHECK-LLVM: [[BlockInv2:%[0-9]+]] = addrspacecast ptr @__device_side_enqueue_block_invoke_kernel to ptr addrspace(4)
  // CHECK-LLVM: call spir_func i32 @__enqueue_kernel_basic(ptr {{.*}}, i32 {{.*}}, ptr {{.*}}, ptr addrspace(4) [[BlockInv2]], ptr addrspace(4) [[Block2Ptr]])
  // CHECK-SPV-IR: call spir_func i32 @_Z21__spirv_EnqueueKernelP13__spirv_Queuei9ndrange_tiPU3AS4P19__spirv_DeviceEventS5_U13block_pointerFvvEPU3AS4cii(target("spirv.Queue") {{.*}}, i32 {{.*}}, ptr {{.*}}, i32 0, ptr addrspace(4) null, ptr addrspace(4) null, ptr @__device_side_enqueue_block_invoke_kernel, ptr addrspace(4) {{.*}}, i32 {{.*}}, i32 {{.*}})
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(void) {
                   a[i] = c0;
                 });

  // Emits block literal on stack and block kernel.

  // CHECK-SPIRV: PtrCastToGeneric [[EventPtrTy]] [[Event1:[0-9]+]]
  // CHECK-SPIRV: PtrCastToGeneric [[EventPtrTy]] [[Event2:[0-9]+]]

  // CHECK-SPIRV: Bitcast [[Int8PtrGenTy]] [[BlockLit2:[0-9]+]]
  // CHECK-SPIRV: EnqueueKernel [[Int32Ty]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  //                            [[ConstInt2]] [[Event1]] [[Event2]]
  //                            [[BlockKer2]] [[BlockLit2]] [[ConstInt20]] [[ConstInt8]]

  // CHECK-LLVM: [[Block3:%[0-9]+]] = addrspacecast ptr %block4 to ptr addrspace(4)
  // CHECK-LLVM: [[Block3Ptr:%[0-9]+]] = bitcast ptr addrspace(4) [[Block3]] to ptr addrspace(4)
  // CHECK-LLVM: [[BlockInv3:%[0-9]+]] = addrspacecast ptr @__device_side_enqueue_block_invoke_2_kernel to ptr addrspace(4)
  // CHECK-LLVM: call spir_func i32 @__enqueue_kernel_basic_events(ptr {{.*}}, i32 {{.*}}, ptr {{.*}}, i32 2, ptr addrspace(4) {{.*}}, ptr addrspace(4) {{.*}}, ptr addrspace(4) [[BlockInv3]], ptr addrspace(4) [[Block3Ptr]])
  // CHECK-SPV-IR: call spir_func i32 @_Z21__spirv_EnqueueKernelP13__spirv_Queuei9ndrange_tiPU3AS4P19__spirv_DeviceEventS5_U13block_pointerFvvEPU3AS4cii(target("spirv.Queue") {{.*}}, i32 {{.*}}, ptr {{.*}}, i32 2, ptr addrspace(4) {{.*}}, ptr addrspace(4) {{.*}}, ptr @__device_side_enqueue_block_invoke_2_kernel, ptr addrspace(4) %{{.*}}, i32 {{.*}}, i32 {{.*}})
  enqueue_kernel(default_queue, flags, ndrange, 2, &event_wait_list, &clk_event,
                 ^(void) {
                   a[i] = b[i];
                 });

  char c;
  // Emits global block literal and block kernel.

  // CHECK-SPIRV: PtrAccessChain [[Int32LocPtrTy]] [[LocalBuf31:[0-9]+]]
  // CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[BlockLit3Tmp:[0-9]+]] [[BlockGlb1:[0-9]+]]
  // CHECK-SPIRV: Bitcast [[Int8PtrGenTy]] [[BlockLit3:[0-9]+]] [[BlockLit3Tmp]]
  // CHECK-SPIRV: EnqueueKernel [[Int32Ty]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  //                            [[ConstInt2]] [[Event1]] [[Event2]]
  //                            [[BlockKer3]] [[BlockLit3]] [[ConstInt8]] [[ConstInt8]]
  //                            [[LocalBuf31]]

  // CHECK-LLVM: [[Block0Tmp:%[0-9]+]] = addrspacecast ptr addrspace(1) @__block_literal_global to ptr addrspace(4)
  // CHECK-LLVM: [[Block0:%[0-9]+]] = bitcast ptr addrspace(4) [[Block0Tmp]] to ptr addrspace(4)
  // CHECK-LLVM: [[BlockInv0:%[0-9]+]] = addrspacecast ptr @__device_side_enqueue_block_invoke_3_kernel to ptr addrspace(4)
  // CHECK-LLVM: call spir_func i32 @__enqueue_kernel_events_varargs(ptr {{.*}}, i32 {{.*}}, ptr {{.*}}, i32 2, ptr addrspace(4) {{.*}}, ptr addrspace(4) {{.*}}, ptr addrspace(4) [[BlockInv0]], ptr addrspace(4) [[Block0]], i32 1, ptr {{.*}})
  // CHECK-SPV-IR: call spir_func i32 @_Z21__spirv_EnqueueKernelP13__spirv_Queuei9ndrange_tiPU3AS4P19__spirv_DeviceEventS5_U13block_pointerFvvEPU3AS4ciiPi(target("spirv.Queue") {{.*}}, i32 {{.*}}, ptr {{.*}}, i32 2, ptr addrspace(4) {{.*}}, ptr addrspace(4) {{.*}}, ptr @__device_side_enqueue_block_invoke_3_kernel, ptr addrspace(4) {{.*}}, i32 {{.*}}, i32 {{.*}}, ptr {{.*}})
  enqueue_kernel(default_queue, flags, ndrange, 2, event_wait_list2, &clk_event,
                 ^(local void *p) {
                   return;
                 },
                 c);

  // Emits global block literal and block kernel.

  // CHECK-SPIRV: PtrAccessChain [[Int32LocPtrTy]] [[LocalBuf41:[0-9]+]]
  // CHECK-SPIRV: PtrAccessChain [[Int32LocPtrTy]] [[LocalBuf42:[0-9]+]]
  // CHECK-SPIRV: PtrAccessChain [[Int32LocPtrTy]] [[LocalBuf43:[0-9]+]]
  // CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[BlockLit4Tmp:[0-9]+]] [[BlockGlb2:[0-9]+]]
  // CHECK-SPIRV: Bitcast [[Int8PtrGenTy]] [[BlockLit4:[0-9]+]] [[BlockLit4Tmp]]
  // CHECK-SPIRV: EnqueueKernel [[Int32Ty]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  //                            [[ConstInt0]] [[EventNull]] [[EventNull]]
  //                            [[BlockKer4]] [[BlockLit4]] [[ConstInt8]] [[ConstInt8]]
  //                            [[LocalBuf41]] [[LocalBuf42]] [[LocalBuf43]]

  // CHECK-LLVM: [[Block1Tmp:%[0-9]+]] = addrspacecast ptr addrspace(1) @__block_literal_global.1 to ptr addrspace(4)
  // CHECK-LLVM: [[Block1:%[0-9]+]] = bitcast ptr addrspace(4) [[Block1Tmp]] to ptr addrspace(4)
  // CHECK-LLVM: [[BlockInv1:%[0-9]+]] = addrspacecast ptr @__device_side_enqueue_block_invoke_4_kernel to ptr addrspace(4)
  // CHECK-LLVM: call spir_func i32 @__enqueue_kernel_varargs(ptr {{.*}}, i32 {{.*}}, ptr {{.*}}, ptr addrspace(4) [[BlockInv1]], ptr addrspace(4) [[Block1]], i32 3, ptr {{.*}})
  // CHECK-SPV-IR: call spir_func i32 @_Z21__spirv_EnqueueKernelP13__spirv_Queuei9ndrange_tiPU3AS4P19__spirv_DeviceEventS5_U13block_pointerFvvEPU3AS4ciiPiSA_SA_(target("spirv.Queue") {{.*}}, i32 {{.*}}, ptr {{.*}}, i32 0, ptr addrspace(4) null, ptr addrspace(4) null, ptr @__device_side_enqueue_block_invoke_4_kernel, ptr addrspace(4) {{.*}}, i32 {{.*}}, i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *p1, local void *p2, local void *p3) {
                   return;
                 },
                 1, 2, 4);

  // Emits block literal on stack and block kernel.

  // CHECK-SPIRV: PtrCastToGeneric [[EventPtrTy]] [[Event1:[0-9]+]]

  // CHECK-SPIRV: Bitcast [[Int8PtrGenTy]] [[BlockLit2:[0-9]+]]
  // CHECK-SPIRV: EnqueueKernel [[Int32Ty]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  //                            [[ConstInt0]] [[EventNull]] [[Event1]]
  //                            [[BlockKer5]] [[BlockLit5]] [[ConstInt20]] [[ConstInt8]]

  // CHECK-LLVM: [[Block5:%[0-9]+]] = addrspacecast ptr %block15 to ptr addrspace(4)
  // CHECK-LLVM: [[Block5Ptr:%[0-9]+]] = bitcast ptr addrspace(4) [[Block5]] to ptr addrspace(4)
  // CHECK-LLVM: [[BlockInv5:%[0-9]+]] = addrspacecast ptr @__device_side_enqueue_block_invoke_5_kernel to ptr addrspace(4)
  // CHECK-LLVM: call spir_func i32 @__enqueue_kernel_basic_events(ptr {{.*}}, i32 {{.*}}, ptr {{.*}}, i32 0, ptr addrspace(4) null, ptr addrspace(4) {{.*}}, ptr addrspace(4) [[BlockInv5]], ptr addrspace(4) [[Block5Ptr]])
  // CHECK-SPV-IR: call spir_func i32 @_Z21__spirv_EnqueueKernelP13__spirv_Queuei9ndrange_tiPU3AS4P19__spirv_DeviceEventS5_U13block_pointerFvvEPU3AS4cii(target("spirv.Queue") {{.*}}, i32 {{.*}}, ptr {{.*}}, i32 0, ptr addrspace(4) null, ptr addrspace(4) {{.*}}, ptr @__device_side_enqueue_block_invoke_5_kernel, ptr addrspace(4) {{.*}}, i32 {{.*}}, i32 {{.*}})
  enqueue_kernel(default_queue, flags, ndrange, 0, NULL, &clk_event,
                 ^(void) {
                   a[i] = b[i];
                 });
}

// CHECK-SPIRV-DAG: Function [[VoidTy]] [[BlockKer1]] 0 [[BlockTy1]]
// CHECK-SPIRV-DAG: Function [[VoidTy]] [[BlockKer2]] 0 [[BlockTy1]]
// CHECK-SPIRV-DAG: Function [[VoidTy]] [[BlockKer3]] 0 [[BlockTy2]]
// CHECK-SPIRV-DAG: Function [[VoidTy]] [[BlockKer4]] 0 [[BlockTy3]]
// CHECK-SPIRV-DAG: Function [[VoidTy]] [[BlockKer5]] 0 [[BlockTy1]]

// CHECK-LLVM-DAG: define spir_kernel void @__device_side_enqueue_block_invoke_kernel(ptr addrspace(4){{.*}})
// CHECK-LLVM-DAG: define spir_kernel void @__device_side_enqueue_block_invoke_2_kernel(ptr addrspace(4){{.*}})
// CHECK-LLVM-DAG: define spir_kernel void @__device_side_enqueue_block_invoke_3_kernel(ptr addrspace(4){{.*}}, ptr addrspace(3){{.*}})
// CHECK-LLVM-DAG: define spir_kernel void @__device_side_enqueue_block_invoke_4_kernel(ptr addrspace(4){{.*}}, ptr addrspace(3){{.*}}, ptr addrspace(3){{.*}}, ptr addrspace(3){{.*}})
// CHECK-LLVM-DAG: define spir_kernel void @__device_side_enqueue_block_invoke_5_kernel(ptr addrspace(4){{.*}})
