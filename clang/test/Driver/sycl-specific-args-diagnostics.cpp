// This test emits an error or a warning if -fsycl-* options are not used in
// conjunction with -fsycl.

// Error should be emitted when using -fsycl-host-compiler without -fsycl

// RUN:   %clang -### -fsycl-host-compiler=g++  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL %s
// RUN:   %clang_cl -### -fsycl-host-compiler=g++  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL %s
// CHK-NO-FSYCL: error: '-fsycl-host-compiler' must be used in conjunction with '-fsycl' to enable offloading

// Warning should be emitted when using -fsycl-host-compiler-options without -fsycl-host-compiler

// RUN:   %clang -###  -fsycl -fsycl-host-compiler-options="-g"  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-WARNING %s
// RUN:   %clang_cl -###  -fsycl -fsycl-host-compiler-options="-g"  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-WARNING %s
// CHK-FSYCL-WARNING: warning: '-fsycl-host-compiler-options=' should be used only in conjunction with '-fsycl-host-compiler'

// Warning should be emitted when using -fsycl-dead-args-optimization without -fsycl
// RUN:   %clang -### -fsycl-dead-args-optimization  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-DEAD-ARGS %s
// WARNING-DEAD-ARGS: warning: argument unused during compilation: '-fsycl-dead-args-optimization' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-dead-args-optimization  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-DEAD-ARGS-CL %s
// WARNING-DEAD-ARGS-CL: warning: argument unused during compilation: '-fsycl-dead-args-optimization' [-Wunused-command-line-argument]

// Warning should be emitted when using -fsycl-device-lib-jit-link without -fsycl
// RUN:   %clang -### -fsycl-device-lib-jit-link  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-JIT-LINK %s
// WARNING-JIT-LINK: warning: argument unused during compilation: '-fsycl-device-lib-jit-link' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-device-lib-jit-link  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-JIT-LINK-CL %s
// WARNING-JIT-LINK-CL: warning: argument unused during compilation: '-fsycl-device-lib-jit-link' [-Wunused-command-line-argument]

// Warning should be emitted when using -fsycl-default-sub-group-size= without -fsycl
// RUN:   %clang -### -fsycl-default-sub-group-size=10  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-DSS %s
// WARNING-DSS: warning: argument unused during compilation: '-fsycl-default-sub-group-size=10' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-default-sub-group-size=10  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-DSS-CL %s
// WARNING-DSS-CL: unknown argument ignored in clang-cl: '-fsycl-default-sub-group-size=10' [-Wunknown-argument]

// Warning should be emitted when using -fsycl-device-code-split-esimd without -fsycl
// RUN:   %clang -### -fsycl-device-code-split-esimd  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-DCSE %s
// WARNING-DCSE: warning: argument unused during compilation: '-fsycl-device-code-split-esimd' [-Wunused-command-line-argument]

// Warning should be emitted when using -fsycl-device-lib=libc without -fsycl
// RUN:   %clang -### -fsycl-device-lib=libc  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-DEVICE-LIB %s
// WARNING-DEVICE-LIB: warning: argument unused during compilation: '-fsycl-device-lib=libc' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-device-lib=libc  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-DEVICE-LIB-CL %s
// WARNING-DEVICE-LIB-CL: warning: argument unused during compilation: '-fsycl-device-lib=libc' [-Wunused-command-line-argument]

// Warning should be emitted when using -fsycl-device-obj=spirv without -fsycl
// RUN:   %clang -### -fsycl-device-obj=spirv  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-DEVICE-OBJ %s
// WARNING-DEVICE-OBJ: warning: argument unused during compilation: '-fsycl-device-obj=spirv' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-device-obj=spirv  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-DEVICE-OBJ-CL %s
// WARNING-DEVICE-OBJ-CL: warning: argument unused during compilation: '-fsycl-device-obj=spirv' [-Wunused-command-line-argument]

// Warning should be emitted when using -fsycl-device-only without -fsycl
// RUN:   %clang -### -fsycl-device-only  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-DEVICE-ONLY %s
// WARNING-DEVICE-ONLY-NOT: warning: argument unused during compilation: '-fsycl-device-only' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-device-only  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-DEVICE-ONLY %s
// WARNING-DEVICE-ONLY-NOT: warning: argument unused during compilation: '-fsycl-device-only' [-Wunused-command-line-argument]

// Warning should be emitted when using -fsycl-early-optimizations without -fsycl
// RUN:   %clang -### -fsycl-early-optimizations  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-EARLY-OPT %s
// WARNING-EARLY-OPT: warning: argument unused during compilation: '-fsycl-early-optimizations' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-early-optimizations  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-EARLY-OPT-CL %s
// WARNING-EARLY-OPT-CL: warning: argument unused during compilation: '-fsycl-early-optimizations' [-Wunused-command-line-argument]

// Warning should be emitted when using -fsycl-embed-ir without -fsycl
// RUN:   %clang -### -fsycl-embed-ir  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-EMBED-IR %s
// WARNING-EMBED-IR: warning: argument unused during compilation: '-fsycl-embed-ir' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-embed-ir  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-EMBED-IR-CL %s
// WARNING-EMBED-IR-CL: warning: argument unused during compilation: '-fsycl-embed-ir' [-Wunused-command-line-argument]

// Warning should be emitted when using -fsycl-esimd-force-stateless-mem without -fsycl
// RUN:   %clang -### -fsycl-esimd-force-stateless-mem  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-FSM %s
// WARNING-FSM: warning: argument unused during compilation: '-fsycl-esimd-force-stateless-mem' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-esimd-force-stateless-mem  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-FSM-CL %s
// WARNING-FSM-CL: warning: argument unused during compilation: '-fsycl-esimd-force-stateless-mem' [-Wunused-command-line-argument]

// Warning should be emitted when using -fsycl-force-inline-kernel-lambda without -fsycl
// RUN:   %clang -### -fsycl-force-inline-kernel-lambda  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-FORCE-INLINE %s
// WARNING-FORCE-INLINE: warning: argument unused during compilation: '-fsycl-force-inline-kernel-lambda' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-force-inline-kernel-lambda  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-FORCE-INLINE-CL %s
// WARNING-FORCE-INLINE-CL: warning: argument unused during compilation: '-fsycl-force-inline-kernel-lambda' [-Wunused-command-line-argument]

// Warning should be emitted when using -fsycl-fp32-prec-sqrt without -fsycl
// RUN:   %clang -### -fsycl-fp32-prec-sqrt  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-FP32 %s
// WARNING-FP32: warning: argument unused during compilation: '-fsycl-fp32-prec-sqrt' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-fp32-prec-sqrt  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-FP32-CL %s
// WARNING-FP32-CL: warning: unknown argument ignored in clang-cl: '-fsycl-fp32-prec-sqrt' [-Wunknown-argument]

// Warning should be emitted when using -fsycl-id-queries-fit-in-int without -fsycl
// RUN:   %clang -### -fsycl-id-queries-fit-in-int  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-ID %s
// WARNING-ID: warning: argument unused during compilation: '-fsycl-id-queries-fit-in-int' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-id-queries-fit-in-int  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-ID-CL %s
// WARNING-ID-CL: warning: argument unused during compilation: '-fsycl-id-queries-fit-in-int' [-Wunused-command-line-argument]

// Warning should be emitted when using -fsycl-instrument-device-code without -fsycl
// RUN:   %clang -### -fsycl-instrument-device-code  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-INS-DEV %s
// WARNING-INS-DEV: warning: argument unused during compilation: '-fsycl-instrument-device-code' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-instrument-device-code  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-INS-DEV-CL %s
// WARNING-INS-DEV-CL: warning: argument unused during compilation: '-fsycl-instrument-device-code' [-Wunused-command-line-argument]

// Warning should be emitted when using -fsycl-libspirv-path without -fsycl
// RUN:   %clang -### -fsycl-libspirv-path=libspirv.bc  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-LIBSPRIV %s
// WARNING-LIBSPRIV: warning: argument unused during compilation: '-fsycl-libspirv-path=libspirv.bc' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-libspirv-path=libspirv.bc  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-LIBSPRIV-CL %s
// WARNING-LIBSPRIV-CL: warning: argument unused during compilation: '-fsycl-libspirv-path=libspirv.bc' [-Wunused-command-line-argument]

// Warning should be emitted when using -fsycl-max-parallel-link-jobs without -fyscl
// RUN:   %clang -### -fsycl-max-parallel-link-jobs=4  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-PARALLEL %s
// WARNING-PARALLEL: warning: argument unused during compilation: '-fsycl-max-parallel-link-jobs=4' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-max-parallel-link-jobs=4  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-PARALLEL-CL %s
// WARNING-PARALLEL-CL: warning: argument unused during compilation: '-fsycl-max-parallel-link-jobs=4' [-Wunused-command-line-argument]

// Warning should be emitted when using -fsycl-optimize-non-user-code without -fsycl
// RUN:   %clang -### -fsycl-optimize-non-user-code  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-OPTIMIZE %s
// WARNING-OPTIMIZE: warning: argument unused during compilation: '-fsycl-optimize-non-user-code' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-optimize-non-user-code  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-OPTIMIZE-CL %s
// WARNING-OPTIMIZE-CL: warning: argument unused during compilation: '-fsycl-optimize-non-user-code' [-Wunused-command-line-argument]

// Warning should be emitted when using -fsycl-unnamed-lambda without -fsycl
// RUN:   %clang -### -fsycl-unnamed-lambda  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNNAMED-LAMBDA %s
// WARNING-UNNAMED-LAMBDA: warning: argument unused during compilation: '-fsycl-unnamed-lambda' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-unnamed-lambda  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNNAMED-LAMBDA-CL %s
// WARNING-UNNAMED-LAMBDA-CL: warning: argument unused during compilation: '-fsycl-unnamed-lambda' [-Wunused-command-line-argument]

// Warning should be emitted when using -fsycl-use-bitcode without -fsycl
// RUN:   %clang -### -fsycl-use-bitcode  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-USE-BITCODE %s
// WARNING-USE-BITCODE: warning: argument unused during compilation: '-fsycl-use-bitcode' [-Wunused-command-line-argument]
// RUN:   %clang_cl -### -fsycl-use-bitcode  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-USE-BITCODE-CL %s
// WARNING-USE-BITCODE-CL: warning: argument unused during compilation: '-fsycl-use-bitcode' [-Wunused-command-line-argument]
