// Compiler-owned options are serialized once, during compilation. Link
// orchestration interprets the full context but emits only linker options.

// RUN: %clangxx --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/SYCL \
// RUN:   --offload-new-driver -fsycl -fsycl-targets=spir64 \
// RUN:   -g -O0 -ftarget-register-alloc-mode=pvc:large \
// RUN:   -foffload-fp32-prec-div -foffload-fp32-prec-sqrt -ftarget-compile-fast \
// RUN:   -### %s 2>&1 | FileCheck %s --check-prefixes=JIT,LINK

// Compile-only invocations retain the same compiler options.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/SYCL \
// RUN:   --offload-new-driver -fsycl -fsycl-targets=spir64 -c \
// RUN:   -g -O0 -ftarget-register-alloc-mode=pvc:large \
// RUN:   -foffload-fp32-prec-div -foffload-fp32-prec-sqrt -ftarget-compile-fast \
// RUN:   -### %s 2>&1 | FileCheck %s --check-prefix=JIT

// Mixed source/object links do not regenerate compiler options either.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/SYCL \
// RUN:   --offload-new-driver -fsycl -fsycl-targets=spir64 \
// RUN:   -g -O0 -ftarget-register-alloc-mode=pvc:large \
// RUN:   -foffload-fp32-prec-div -foffload-fp32-prec-sqrt -ftarget-compile-fast \
// RUN:   -### %s %S/Inputs/SYCL/objlin64.o 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=JIT,LINK

// Object-only policy: compiler-owned flags do not alter input options.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/SYCL \
// RUN:   --offload-new-driver -fsycl -fsycl-targets=spir64 \
// RUN:   -g -O0 -ftarget-register-alloc-mode=pvc:large \
// RUN:   -foffload-fp32-prec-div -foffload-fp32-prec-sqrt -ftarget-compile-fast \
// RUN:   -### %S/Inputs/SYCL/objlin64.o 2>&1 | FileCheck %s --check-prefix=LINK

// Link orchestration must not interpret or validate compiler-only settings.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/SYCL \
// RUN:   --offload-new-driver -fsycl -fsycl-targets=spir64 \
// RUN:   -ftarget-register-alloc-mode=pvc:invalid \
// RUN:   -### %S/Inputs/SYCL/objlin64.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LINK \
// RUN:       --implicit-check-not="error: unsupported argument 'pvc:invalid' to option '-ftarget-register-alloc-mode='"

// Compilation does validate the setting, exactly once. This also keeps the
// exact diagnostic text above in sync with the driver.
// RUN: not %clangxx --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/SYCL \
// RUN:   --offload-new-driver -fsycl -fsycl-targets=spir64 \
// RUN:   -ftarget-register-alloc-mode=pvc:invalid \
// RUN:   -### %s 2>&1 | FileCheck %s --check-prefix=INVALID_GRF
// INVALID_GRF: error: unsupported argument 'pvc:invalid' to option '-ftarget-register-alloc-mode='
// INVALID_GRF-NOT: error: unsupported argument 'pvc:invalid' to option '-ftarget-register-alloc-mode='

// Defaults belong to compilation too (including PVC's default GRF setting).
// RUN: %clangxx --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/SYCL \
// RUN:   --offload-new-driver -fsycl -fsycl-targets=spir64 \
// RUN:   -### %s 2>&1 | FileCheck %s --check-prefix=LINK
// RUN: %clangxx --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/SYCL \
// RUN:   --offload-new-driver -fsycl -fsycl-targets=spir64 \
// RUN:   -### %S/Inputs/SYCL/objlin64.o 2>&1 | FileCheck %s --check-prefix=LINK

// JIT: llvm-offload-binary{{.*}}compile-opts=-g -ftarget-register-alloc-mode=pvc:-ze-opt-large-register-file -ftarget-compile-fast -foffload-fp32-prec-div -foffload-fp32-prec-sqrt
// LINK: clang-linker-wrapper
// LINK-NOT: --jit-compiler-options=
// LINK-NOT: --jit-linker-options=

// AOT and multiple-target invocations use the same ownership rules. Full
// interpretation still includes fp64 emulation, which is not in the generic
// Clang compiler-option allowlist.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/SYCL \
// RUN:   --offload-new-driver -fsycl -fsycl-targets=spir64,spir64_gen \
// RUN:   -g -O0 -fsycl-fp64-conv-emu \
// RUN:   -foffload-fp32-prec-div -foffload-fp32-prec-sqrt -ftarget-compile-fast \
// RUN:   -### %s 2>&1 | FileCheck %s --check-prefix=AOT
// AOT: llvm-offload-binary{{.*}}triple=spir64_gen-unknown-unknown{{.*}}compile-opts=-options -ze-fp64-gen-conv-emu -g -cl-opt-disable -igc_opts{{.*}}-ze-fp32-correctly-rounded-divide-sqrt
// AOT: clang-linker-wrapper
// AOT-NOT: --jit-compiler-options=
// AOT-NOT: --ocloc-options=

// Explicit backend and linker options remain distinct on object-only links.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/SYCL \
// RUN:   --offload-new-driver -fsycl -fsycl-targets=spir64 \
// RUN:   -Xsycl-target-backend -backend-opt -Xsycl-target-linker -link-opt \
// RUN:   -### %S/Inputs/SYCL/objlin64.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EXPLICIT
// EXPLICIT: clang-linker-wrapper{{.*}}--jit-compiler-options=-backend-opt
// EXPLICIT-SAME: --jit-linker-options=-link-opt

// Exporting symbols is a linker option, not a packaging option.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/SYCL \
// RUN:   --offload-new-driver -fsycl -fsycl-targets=spir64_gen \
// RUN:   -ftarget-export-symbols -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EXPORT
// EXPORT: llvm-offload-binary
// EXPORT-NOT: -library-compilation
// EXPORT: clang-linker-wrapper{{.*}}--ocloc-options=-options
// EXPORT-SAME: --ocloc-options=-library-compilation
// EXPORT-NOT: -library-compilation

// Link-only SYCLBIN settings retain their full-context implications.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/SYCL \
// RUN:   --offload-new-driver -fsycl -fsycl-targets=spir64_gen -fsyclbin=object \
// RUN:   -### %S/Inputs/SYCL/objlin64.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SYCLBIN
// SYCLBIN: clang-linker-wrapper{{.*}}--ocloc-options=-options
// SYCLBIN-SAME: --ocloc-options=-library-compilation

// An explicit opt-out wins over the SYCLBIN implication.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/SYCL \
// RUN:   --offload-new-driver -fsycl -fsycl-targets=spir64_gen -fsyclbin=object \
// RUN:   -fno-target-export-symbols -### %S/Inputs/SYCL/objlin64.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOEXPORT
// NOEXPORT: clang-linker-wrapper
// NOEXPORT-NOT: -library-compilation
