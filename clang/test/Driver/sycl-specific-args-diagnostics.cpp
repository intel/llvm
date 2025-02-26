// This test emits an error or a warning if -fsycl-* options are not used in
// conjunction with -fsycl.

// Error should be emitted when using -fsycl-host-compiler without -fsycl

// RUN:   not %clang -### -fsycl-host-compiler=g++  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL %s
// RUN:   not %clang_cl -### -fsycl-host-compiler=g++  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL %s
// CHK-NO-FSYCL: error: '-fsycl-host-compiler' must be used in conjunction with '-fsycl' to enable offloading

// Warning should be emitted when using -fsycl-host-compiler-options without -fsycl-host-compiler
// RUN:   %clang -###  -fsycl -fsycl-host-compiler-options="-g"  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-WARNING %s
// RUN:   %clang_cl -###  -fsycl -fsycl-host-compiler-options="-g"  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-WARNING %s
// CHK-FSYCL-WARNING: warning: '-fsycl-host-compiler-options' should be used only in conjunction with '-fsycl-host-compiler'

// Warning should be emitted when using -fsycl-dead-args-optimization without -fsycl
// RUN:   %clang -### -fsycl-dead-args-optimization  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-dead-args-optimization %s
// RUN:   %clang_cl -### -fsycl-dead-args-optimization  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-dead-args-optimization %s

// Warning should be emitted when using -fsycl-device-lib-jit-link without -fsycl
// RUN:   %clang -### -fsycl-device-lib-jit-link  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-device-lib-jit-link %s
// RUN:   %clang_cl -### -fsycl-device-lib-jit-link  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-device-lib-jit-link %s

// Warning should be emitted when using -fsycl-default-sub-group-size= without -fsycl
// RUN:   %clang -### -fsycl-default-sub-group-size=10  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-default-sub-group-size=10 %s
// RUN:   %clang_cl -### -fsycl-default-sub-group-size=10  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-default-sub-group-size=10 %s

// Warning should be emitted when using -fsycl-device-code-split-esimd without -fsycl
// RUN:   %clang -### -fsycl-device-code-split-esimd  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-device-code-split-esimd %s

// Warning should be emitted when using -fsycl-device-lib=libc without -fsycl
// RUN:   %clang -### -fsycl-device-lib=libc  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-device-lib=libc %s
// RUN:   %clang_cl -### -fsycl-device-lib=libc  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-device-lib=libc %s

// Warning should be emitted when using -fsycl-device-obj=spirv without -fsycl
// RUN:   %clang -### -fsycl-device-obj=spirv  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-device-obj=spirv  %s
// RUN:   %clang_cl -### -fsycl-device-obj=spirv  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-device-obj=spirv %s

// Warning should not be emitted when using -fsycl-device-only without -fsycl
// RUN:   %clang -### -fsycl-device-only  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-DEVICE-ONLY -DOPT=-fsycl-device-only %s
// RUN:   %clang_cl -### -fsycl-device-only  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-DEVICE-ONLY -DOPT=-fsycl-device-only %s
// WARNING-DEVICE-ONLY-NOT: warning: argument unused during compilation: '[[OPT]]' [-Wunused-command-line-argument]

// Warning should be emitted when using -fsycl-early-optimizations without -fsycl
// RUN:   %clang -### -fsycl-early-optimizations  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-early-optimizations %s
// RUN:   %clang_cl -### -fsycl-early-optimizations  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-early-optimizations %s

// Warning should be emitted when using -fsycl-embed-ir without -fsycl
// RUN:   %clang -### -fsycl-embed-ir  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-embed-ir %s
// RUN:   %clang_cl -### -fsycl-embed-ir  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-embed-ir %s

// Warning should be emitted when using -fsycl-esimd-force-stateless-mem without -fsycl
// RUN:   %clang -### -fsycl-esimd-force-stateless-mem  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-esimd-force-stateless-mem %s
// RUN:   %clang_cl -### -fsycl-esimd-force-stateless-mem  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-esimd-force-stateless-mem %s

// Warning should be emitted when using -fsycl-force-inline-kernel-lambda without -fsycl
// RUN:   %clang -### -fsycl-force-inline-kernel-lambda  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-force-inline-kernel-lambda %s
// RUN:   %clang_cl -### -fsycl-force-inline-kernel-lambda  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-force-inline-kernel-lambda %s

// Warning should be emitted when using -fsycl-fp32-prec-sqrt without -fsycl
// RUN:   %clang -### -fsycl-fp32-prec-sqrt  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-fp32-prec-sqrt %s
// RUN:   %clang_cl -### -fsycl-fp32-prec-sqrt  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-fp32-prec-sqrt %s

// Warning should be emitted when using -fsycl-id-queries-fit-in-int without -fsycl
// RUN:   %clang -### -fsycl-id-queries-fit-in-int  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-id-queries-fit-in-int %s
// RUN:   %clang_cl -### -fsycl-id-queries-fit-in-int  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-id-queries-fit-in-int %s

// Warning should be emitted when using -fsycl-instrument-device-code without -fsycl
// RUN:   %clang -### -fsycl-instrument-device-code  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-instrument-device-code %s
// RUN:   %clang_cl -### -fsycl-instrument-device-code  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-instrument-device-code %s

// Warning should be emitted when using -fsycl-libspirv-path without -fsycl
// RUN:   %clang -### -fsycl-libspirv-path=libspirv.bc  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-libspirv-path=libspirv.bc  %s
// RUN:   %clang_cl -### -fsycl-libspirv-path=libspirv.bc  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-libspirv-path=libspirv.bc %s

// Warning should be emitted when using -fsycl-max-parallel-link-jobs without -fyscl
// RUN:   %clang -### -fsycl-max-parallel-link-jobs=4  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-max-parallel-link-jobs=4 %s
// RUN:   %clang_cl -### -fsycl-max-parallel-link-jobs=4  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-max-parallel-link-jobs=4 %s

// Warning should be emitted when using -fsycl-optimize-non-user-code without -fsycl
// RUN:   %clang -### -fsycl-optimize-non-user-code  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-optimize-non-user-code %s
// RUN:   %clang_cl -### -fsycl-optimize-non-user-code  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-optimize-non-user-code %s

// Warning should be emitted when using -fsycl-unnamed-lambda without -fsycl
// RUN:   %clang -### -fsycl-unnamed-lambda  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-unnamed-lambda %s
// RUN:   %clang_cl -### -fsycl-unnamed-lambda  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-unnamed-lambda %s

// Warning should be emitted when using -fsycl-use-bitcode without -fsycl
// RUN:   %clang -### -fsycl-use-bitcode  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-use-bitcode %s
// RUN:   %clang_cl -### -fsycl-use-bitcode  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WARNING-UNUSED-ARG -DOPT=-fsycl-use-bitcode %s

// WARNING-UNUSED-ARG: warning: argument unused during compilation: '[[OPT]]' [-Wunused-command-line-argument]
