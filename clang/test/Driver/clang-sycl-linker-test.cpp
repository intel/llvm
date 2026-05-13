// Tests the clang-sycl-linker tool.
//
// REQUIRES: spirv-registered-target
//
// Create bitcode files for clang-sycl-linker direct tests
// RUN: %clangxx -emit-llvm -c -target spirv64 %s -o %t_1.bc
// RUN: %clangxx -emit-llvm -c -target spirv64 %s -o %t_2.bc
// Create SYCL offload object files for driver tests
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -c %s -o %t_1.o
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -c %s -o %t_2.o
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-spirv.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SIMPLE-FO
// SIMPLE-FO: sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc{{.*}}libfiles:{{.*}}output: [[LLVMLINKOUT:.*]].bc
// SIMPLE-FO-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: {{.*}}_0.spv
//
// Test that IMG_SPIRV image kind is set for non-AOT compilation.
// RUN: llvm-objdump --offloading %t-spirv.out | FileCheck %s --check-prefix=IMAGE-KIND-SPIRV
// IMAGE-KIND-SPIRV: kind            spir-v
//
// Test the dry run of a simple case with device library files specified.
// RUN: mkdir -p %t.dir
// RUN: touch %t.dir/lib1.bc
// RUN: touch %t.dir/lib2.bc
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc --library-path=%t.dir --device-libs=lib1.bc,lib2.bc -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEVLIBS
// DEVLIBS: sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc{{.*}}libfiles:{{.*}}lib1.bc, {{.*}}lib2.bc{{.*}}output: [[LLVMLINKOUT:.*]].bc
// DEVLIBS-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: a_0.spv
//
// Test a simple case with a random file (not bitcode) as input.
// RUN: touch %t.o
// RUN: not clang-sycl-linker -triple=spirv64 %t.o -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FILETYPEERROR
// FILETYPEERROR: Unsupported file type
//
// Test to see if device library related errors are emitted.
// RUN: not clang-sycl-linker --dry-run -triple=spirv64 %t_1.bc %t_2.bc --library-path=%t.dir --device-libs= -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEVLIBSERR1
// DEVLIBSERR1: Number of device library files cannot be zero
// RUN: not clang-sycl-linker --dry-run -triple=spirv64 %t_1.bc %t_2.bc --library-path=%t.dir --device-libs=lib1.bc,lib2.bc,lib3.bc -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEVLIBSERR2
// DEVLIBSERR2: '{{.*}}lib3.bc' SYCL device library file is not found
//
// Test AOT compilation for an Intel GPU.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 -arch=bmg_g21 %t_1.bc %t_2.bc -o %t-aot-gpu.out 2>&1 \
// RUN:     --ocloc-options="-a -b" \
// RUN:   | FileCheck %s --check-prefix=AOT-INTEL-GPU
// AOT-INTEL-GPU:      sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc{{.*}}libfiles:{{.*}}output: [[LLVMLINKOUT:.*]].bc
// AOT-INTEL-GPU-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: [[SPIRVTRANSLATIONOUT:.*]]_0.spv
// AOT-INTEL-GPU-NEXT: "{{.*}}ocloc{{.*}}" {{.*}}-device bmg_g21 -a -b {{.*}}-output [[SPIRVTRANSLATIONOUT]]_0.out -file [[SPIRVTRANSLATIONOUT]]_0.spv
//
// Test that IMG_Object image kind is set for AOT compilation (Intel GPU).
// RUN: llvm-objdump --offloading %t-aot-gpu.out | FileCheck %s --check-prefix=IMAGE-KIND-OBJECT
// IMAGE-KIND-OBJECT: kind            elf
//
// Test AOT compilation for an Intel CPU.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 -arch=graniterapids %t_1.bc %t_2.bc -o %t-aot-cpu.out 2>&1 \
// RUN:     --opencl-aot-options="-a -b" \
// RUN:   | FileCheck %s --check-prefix=AOT-INTEL-CPU
// AOT-INTEL-CPU:      sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc{{.*}}libfiles:{{.*}}output: [[LLVMLINKOUT:.*]].bc
// AOT-INTEL-CPU-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: [[SPIRVTRANSLATIONOUT:.*]]_0.spv
// AOT-INTEL-CPU-NEXT: "{{.*}}opencl-aot{{.*}}" {{.*}}--device=cpu -a -b {{.*}}-o [[SPIRVTRANSLATIONOUT]]_0.out [[SPIRVTRANSLATIONOUT]]_0.spv
//
// Test that IMG_Object image kind is set for AOT compilation (Intel CPU).
// RUN: llvm-objdump --offloading %t-aot-cpu.out | FileCheck %s --check-prefix=IMAGE-KIND-OBJECT
//
// Check that the output file must be specified.
// RUN: not clang-sycl-linker --dry-run %t_1.bc %t_2.bc 2>&1 \
// RUN: | FileCheck %s --check-prefix=NOOUTPUT
// NOOUTPUT: Output file must be specified
//
// Check that the target triple must be specified.
// RUN: not clang-sycl-linker --dry-run %t_1.bc %t_2.bc -o a.out 2>&1 \
// RUN: | FileCheck %s --check-prefix=NOTARGET
// NOTARGET: Target triple must be specified
//
// Test that driver options are correctly passed through clang-linker-wrapper.
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown \
// RUN:          -Xsycl-target-backend=spir64-unknown-unknown "-backend-opt" \
// RUN:          %t_1.o %t_2.o 2>&1 | FileCheck %s --check-prefix=CHECK-BACKEND-OPT
// CHECK-BACKEND-OPT: clang-linker-wrapper{{.*}} "--device-compiler=sycl:spir64-unknown-unknown=-backend-opt"
//
// Test multiple backend options.
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown \
// RUN:          -Xsycl-target-backend=spir64-unknown-unknown "-opt1 -opt2" \
// RUN:          %t_1.o %t_2.o 2>&1 | FileCheck %s --check-prefix=CHECK-MULTI-OPTS
// CHECK-MULTI-OPTS: clang-linker-wrapper{{.*}} "--device-compiler=sycl:spir64-unknown-unknown=-opt1 -opt2"
//
// Test linker options are forwarded.
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown \
// RUN:          -Xsycl-target-linker=spir64-unknown-unknown "-linker-opt" \
// RUN:          %t_1.o %t_2.o 2>&1 | FileCheck %s --check-prefix=CHECK-LINKER-OPT
// CHECK-LINKER-OPT: clang-linker-wrapper{{.*}} "--device-linker=sycl:spir64-unknown-unknown=-linker-opt"
//
// Test both backend and linker options together.
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown \
// RUN:          -Xsycl-target-backend=spir64-unknown-unknown "-backend-opt" \
// RUN:          -Xsycl-target-linker=spir64-unknown-unknown "-linker-opt" \
// RUN:          %t_1.o %t_2.o 2>&1 | FileCheck %s --check-prefix=CHECK-BOTH-OPTS
// CHECK-BOTH-OPTS: clang-linker-wrapper{{.*}} "--device-compiler=sycl:spir64-unknown-unknown=-backend-opt"
// CHECK-BOTH-OPTS-SAME: "--device-linker=sycl:spir64-unknown-unknown=-linker-opt"
//
// Test AOT GPU with backend options.
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64_gen-unknown-unknown \
// RUN:          -Xsycl-target-backend=spir64_gen "-device pvc" \
// RUN:          %t_1.o %t_2.o 2>&1 | FileCheck %s --check-prefix=CHECK-AOT-GPU
// CHECK-AOT-GPU: clang-linker-wrapper{{.*}} "--device-compiler=sycl:spir64_gen-unknown-unknown=-device pvc"
//
// Test AOT CPU with backend options.
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64_x86_64-unknown-unknown \
// RUN:          -Xsycl-target-backend=spir64_x86_64 "-march=skylake" \
// RUN:          %t_1.o %t_2.o 2>&1 | FileCheck %s --check-prefix=CHECK-AOT-CPU
// CHECK-AOT-CPU: clang-linker-wrapper{{.*}} "--device-compiler=sycl:spir64_x86_64-unknown-unknown=-march=skylake"
//
// Test multi-target with target-specific options (both in same invocation).
// RUN: %clangxx -### -fsycl --offload-new-driver \
// RUN:          -fsycl-targets=spir64_gen-unknown-unknown,spir64_x86_64-unknown-unknown \
// RUN:          -Xsycl-target-backend=spir64_gen "-device pvc" \
// RUN:          -Xsycl-target-backend=spir64_x86_64 "-march=skylake" \
// RUN:          %t_1.o %t_2.o 2>&1 | FileCheck %s --check-prefix=CHECK-MULTI-TARGET
// CHECK-MULTI-TARGET: clang-linker-wrapper{{.*}} "--device-compiler=sycl:spir64_gen-unknown-unknown=-device pvc"
// CHECK-MULTI-TARGET-SAME: "--device-compiler=sycl:spir64_x86_64-unknown-unknown=-march=skylake"
//
// Test that --use-clang-sycl-linker flag is properly forwarded.
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown \
// RUN:          -Xlinker --use-clang-sycl-linker \
// RUN:          %t_1.o %t_2.o 2>&1 | FileCheck %s --check-prefix=CHECK-USE-CSL
// CHECK-USE-CSL: clang-linker-wrapper{{.*}} "--use-clang-sycl-linker"
//
// Tests for appendClangSYCLLinkerArgs functionality when UseClangSYCLLinker=true
//
// Test that backend options are forwarded when using clang-sycl-linker.
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown \
// RUN:          -Xlinker --use-clang-sycl-linker \
// RUN:          -Xsycl-target-backend=spir64-unknown-unknown "-backend-opt" \
// RUN:          %t_1.o %t_2.o 2>&1 | FileCheck %s --check-prefix=CHECK-CSL-BACKEND
// CHECK-CSL-BACKEND: clang-linker-wrapper{{.*}} "--device-compiler=sycl:spir64-unknown-unknown=-backend-opt"{{.*}} "--use-clang-sycl-linker"
//
// Test that linker options are forwarded when using clang-sycl-linker.
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown \
// RUN:          -Xlinker --use-clang-sycl-linker \
// RUN:          -Xsycl-target-linker=spir64-unknown-unknown "-linker-opt" \
// RUN:          %t_1.o %t_2.o 2>&1 | FileCheck %s --check-prefix=CHECK-CSL-LINKER
// CHECK-CSL-LINKER: clang-linker-wrapper{{.*}} "--device-linker=sycl:spir64-unknown-unknown=-linker-opt"{{.*}} "--use-clang-sycl-linker"
//
// Test that both backend and linker options work together with clang-sycl-linker.
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown \
// RUN:          -Xlinker --use-clang-sycl-linker \
// RUN:          -Xsycl-target-backend=spir64-unknown-unknown "-backend-opt" \
// RUN:          -Xsycl-target-linker=spir64-unknown-unknown "-linker-opt" \
// RUN:          %t_1.o %t_2.o 2>&1 | FileCheck %s --check-prefix=CHECK-CSL-BOTH
// CHECK-CSL-BOTH: clang-linker-wrapper{{.*}} "--device-compiler=sycl:spir64-unknown-unknown=-backend-opt"{{.*}} "--device-linker=sycl:spir64-unknown-unknown=-linker-opt"{{.*}} "--use-clang-sycl-linker"
//
// Test AOT GPU with clang-sycl-linker.
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64_gen-unknown-unknown \
// RUN:          -Xlinker --use-clang-sycl-linker \
// RUN:          -Xsycl-target-backend=spir64_gen "-device pvc" \
// RUN:          %t_1.o %t_2.o 2>&1 | FileCheck %s --check-prefix=CHECK-CSL-AOT-GPU
// CHECK-CSL-AOT-GPU: clang-linker-wrapper{{.*}} "--device-compiler=sycl:spir64_gen-unknown-unknown=-device pvc"{{.*}} "--use-clang-sycl-linker"
//
// Test AOT CPU with clang-sycl-linker.
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64_x86_64-unknown-unknown \
// RUN:          -Xlinker --use-clang-sycl-linker \
// RUN:          -Xsycl-target-backend=spir64_x86_64 "-march=skylake" \
// RUN:          %t_1.o %t_2.o 2>&1 | FileCheck %s --check-prefix=CHECK-CSL-AOT-CPU
// CHECK-CSL-AOT-CPU: clang-linker-wrapper{{.*}} "--device-compiler=sycl:spir64_x86_64-unknown-unknown=-march=skylake"{{.*}} "--use-clang-sycl-linker"
//
// Test multi-target with clang-sycl-linker.
// RUN: %clangxx -### -fsycl --offload-new-driver \
// RUN:          -fsycl-targets=spir64_gen-unknown-unknown,spir64_x86_64-unknown-unknown \
// RUN:          -Xlinker --use-clang-sycl-linker \
// RUN:          -Xsycl-target-backend=spir64_gen "-device pvc" \
// RUN:          -Xsycl-target-backend=spir64_x86_64 "-march=skylake" \
// RUN:          %t_1.o %t_2.o 2>&1 | FileCheck %s --check-prefix=CHECK-CSL-MULTI
// CHECK-CSL-MULTI: clang-linker-wrapper{{.*}} "--device-compiler=sycl:spir64_gen-unknown-unknown=-device pvc"{{.*}} "--device-compiler=sycl:spir64_x86_64-unknown-unknown=-march=skylake"{{.*}} "--use-clang-sycl-linker"
//
// Test that --save-temps is forwarded when using clang-sycl-linker.
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown \
// RUN:          -Xlinker --use-clang-sycl-linker -Xlinker --save-temps \
// RUN:          %t_1.o %t_2.o 2>&1 | FileCheck %s --check-prefix=CHECK-CSL-SAVE-TEMPS
// CHECK-CSL-SAVE-TEMPS: clang-linker-wrapper{{.*}} "--use-clang-sycl-linker"{{.*}} "--save-temps"
//
// Test that --dry-run is forwarded when using clang-sycl-linker.
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown \
// RUN:          -Xlinker --use-clang-sycl-linker -Xlinker --dry-run \
// RUN:          %t_1.o %t_2.o 2>&1 | FileCheck %s --check-prefix=CHECK-CSL-DRY
// CHECK-CSL-DRY: clang-linker-wrapper{{.*}} "--use-clang-sycl-linker"{{.*}} "--dry-run"



