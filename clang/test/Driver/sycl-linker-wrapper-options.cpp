// Tests that clang driver correctly forwards options to clang-linker-wrapper
// for SYCL offload compilation.
//
// REQUIRES: spirv-registered-target
//
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -c %s -o %t_1.o
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -c %s -o %t_2.o
//
// Test that --use-clang-sycl-linker is not passed by default.
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown \
// RUN:          %t_1.o 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT
// CHECK-DEFAULT: clang-linker-wrapper
// CHECK-DEFAULT-NOT: "--use-clang-sycl-linker"
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

int foo() { return 42; }
