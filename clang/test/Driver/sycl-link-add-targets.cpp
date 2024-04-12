///
/// Perform several driver tests for SYCL offloading using -fsycl-link-targets
/// and -fsycl-add-targets
///

/// Check whether an invalid SYCL target is specified:
// RUN:   not %clang -### -fsycl -fsycl-add-targets=dummy-target:dummy-file %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET-ADD %s
// RUN:   not %clang_cl -### -fsycl -fsycl-add-targets=dummy-target:dummy-file %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET-ADD %s
// CHK-INVALID-TARGET-ADD: error: SYCL target is invalid: 'dummy-target'

/// Check error for no -fsycl option
// RUN:   not %clang -### -fsycl-link-targets=spir64-unknown-unknown  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL-LINK-TGTS %s
// CHK-NO-FSYCL-LINK-TGTS: error: '-fsycl-link-targets' must be used in conjunction with '-fsycl' to enable offloading

// RUN:   not %clang -### -fsycl-add-targets=spir64-unknown-unknown  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL-ADD %s
// CHK-NO-FSYCL-ADD: error: '-fsycl-add-targets' must be used in conjunction with '-fsycl' to enable offloading

/// Check error for -fsycl-add-targets -fsycl-link-targets conflict
// RUN:   not %clang -### -fsycl-link-targets=spir64-unknown-unknown -fsycl-add-targets=spir64:dummy.spv -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-ADD-LINK %s
// RUN:   not %clang_cl -### -fsycl-link-targets=spir64-unknown-unknown -fsycl-add-targets=spir64:dummy.spv -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-ADD-LINK %s
// CHK-SYCL-ADD-LINK: error: The option -fsycl-link-targets= conflicts with -fsycl-add-targets=

/// Check error for -fsycl-targets -fsycl-link-targets conflict
// RUN:   not %clang -### -fsycl-link-targets=spir64-unknown-unknown -fsycl-targets=spir64-unknown-unknown -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-LINK-CONFLICT %s
// RUN:   not %clang_cl -### -fsycl-link-targets=spir64-unknown-unknown -fsycl-targets=spir64-unknown-unknown -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-LINK-CONFLICT %s
// CHK-SYCL-LINK-CONFLICT: error: The option -fsycl-targets= conflicts with -fsycl-link-targets=

/// Check error for -fsycl-[add|link]-targets with bad triple
// RUN:   not %clang -### -fsycl-add-targets=spir64_bad-unknown-unknown:dummy.spv -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-BAD-ADDLINK-TRIPLE %s
// RUN:   not %clang_cl -### -fsycl-add-targets=spir64_bad-unknown-unknown:dummy.spv -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-BAD-ADDLINK-TRIPLE %s
// RUN:   not %clang -### -fsycl-link-targets=spir64_bad-unknown-unknown -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-BAD-ADDLINK-TRIPLE %s
// RUN:   not %clang_cl -### -fsycl-link-targets=spir64_bad-unknown-unknown -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-BAD-ADDLINK-TRIPLE %s
// CHK-SYCL-FPGA-BAD-ADDLINK-TRIPLE: error: SYCL target is invalid: 'spir64_bad-unknown-unknown'

/// Check no error for -fsycl-[add|link]-targets with good triple
// RUN:   %clang -### -fsycl-add-targets=spir64-unknown-unknown:dummy.spv -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-ADDLINK-TRIPLE %s
// RUN:   %clang_cl -### -fsycl-add-targets=spir64-unknown-unknown:dummy.spv -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-ADDLINK-TRIPLE %s
// RUN:   %clang -### -fsycl-add-targets=spir64_gen-unknown-unknown:dummy.spv -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-ADDLINK-TRIPLE %s
// RUN:   %clang -### -fsycl-add-targets=spir64_fpga-unknown-unknown:dummy.spv -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-ADDLINK-TRIPLE %s
// RUN:   %clang -### -fsycl-add-targets=spir64_x86_64-unknown-unknown:dummy.spv -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-ADDLINK-TRIPLE %s
// RUN:   %clang -### -fsycl-link-targets=spir64-unknown-unknown -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-ADDLINK-TRIPLE %s
// RUN:   %clang_cl -### -fsycl-link-targets=spir64-unknown-unknown -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-ADDLINK-TRIPLE %s
// RUN:   %clang -### -fsycl-link-targets=spir64_gen-unknown-unknown -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-ADDLINK-TRIPLE %s
// RUN:   %clang -### -fsycl-link-targets=spir64_fpga-unknown-unknown -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-ADDLINK-TRIPLE %s
// RUN:   %clang -### -fsycl-link-targets=spir64_x86_64-unknown-unknown -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-ADDLINK-TRIPLE %s
// CHK-SYCL-ADDLINK-TRIPLE-NOT: error: SYCL target is invalid

/// Check -fsycl-link-targets=<triple> behaviors unbundle
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-unknown %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS-UB %s
// RUN:   %clang_cl -### -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-unknown %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS-UB %s
// CHK-LINK-TARGETS-UB: 0: input, "[[INPUT:.+\.o]]", object
// CHK-LINK-TARGETS-UB: 1: clang-offload-unbundler, {0}, object
// CHK-LINK-TARGETS-UB: 2: linker, {1}, image, (device-sycl)
// CHK-LINK-TARGETS-UB: 3: llvm-spirv, {2}, image, (device-sycl)
// CHK-LINK-TARGETS-UB: 4: offload, "device-sycl (spir64-unknown-unknown)" {3}, image
// CHK-LINK-TARGETS-UB-NOT: offload

// RUN: %clangxx -ccc-print-bindings --target=x86_64-unknown-linux-gnu \
// RUN:          -fsycl -o checkme.out -fsycl-link-targets=spir64 %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-LINK-TARGETS-BINDINGS %s
// CHK-LINK-TARGETS-BINDINGS: "spir64-unknown-unknown" - "clang", inputs: [{{.*}}], output: "[[IR_OUTPUT_BC:.+\.bc]]"
// CHK-LINK-TARGETS-BINDINGS: "spir64-unknown-unknown" - "SYCL::Linker", inputs: ["[[IR_OUTPUT_BC]]"], output: "[[LLVM_LINK_OUTPUT:.+\.out]]"
// CHK-LINK-TARGETS-BINDINGS: "spir64-unknown-unknown" - "SPIR-V translator", inputs: ["[[LLVM_LINK_OUTPUT]]"], output: "checkme.out"

/// Check -fsycl-link-targets=<triple> behaviors unbundle multiple objects
// RUN:   touch %t-a.o
// RUN:   touch %t-b.o
// RUN:   touch %t-c.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-unknown %t-a.o %t-b.o %t-c.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS-UB2 %s
// RUN:   %clang_cl -### -ccc-print-phases --target=x86_64-pc-windws-msvc -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-unknown %t-a.o %t-b.o %t-c.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS-UB2 %s
// CHK-LINK-TARGETS-UB2: 0: input, "[[INPUT:.+\a.o]]", object
// CHK-LINK-TARGETS-UB2: 1: clang-offload-unbundler, {0}, object
// CHK-LINK-TARGETS-UB2: 2: input, "[[INPUT:.+\b.o]]", object
// CHK-LINK-TARGETS-UB2: 3: clang-offload-unbundler, {2}, object
// CHK-LINK-TARGETS-UB2: 4: input, "[[INPUT:.+\c.o]]", object
// CHK-LINK-TARGETS-UB2: 5: clang-offload-unbundler, {4}, object
// CHK-LINK-TARGETS-UB2: 6: linker, {1, 3, 5}, image, (device-sycl)
// CHK-LINK-TARGETS-UB2: 7: llvm-spirv, {6}, image, (device-sycl)
// CHK-LINK-TARGETS-UB2: 8: offload, "device-sycl (spir64-unknown-unknown)" {7}, image

/// Check -fsycl-link-targets=<triple> behaviors from source
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS %s -DSUBARCH=
// RUN:   %clang_cl -### -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS %s -DSUBARCH=
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link-targets=spir64_gen-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS %s -DSUBARCH=_gen
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link-targets=spir64_fpga-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS %s -DSUBARCH=_fpga
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link-targets=spir64_x86_64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS %s -DSUBARCH=_x86_64
// CHK-LINK-TARGETS: 0: input, "[[INPUT:.+\.cpp]]", c++, (device-sycl)
// CHK-LINK-TARGETS: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHK-LINK-TARGETS: 2: compiler, {1}, ir, (device-sycl)
// CHK-LINK-TARGETS: 3: linker, {2}, image, (device-sycl)
// CHK-LINK-TARGETS: 4: llvm-spirv, {3}, image, (device-sycl)
// CHK-LINK-TARGETS: 5: offload, "device-sycl (spir64[[SUBARCH]]-unknown-unknown)" {4}, image

/// Check -fsycl-add-targets=<triple> behaviors unbundle
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-add-targets=spir64-unknown-unknown:dummy.spv %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-ADD-TARGETS-UB %s
// CHK-ADD-TARGETS-UB: 0: input, "[[INPUT:.+\.o]]", object, (host-sycl)
// CHK-ADD-TARGETS-UB: 1: clang-offload-unbundler, {0}, object, (host-sycl)
// CHK-ADD-TARGETS-UB: 2: linker, {1}, image, (host-sycl)
// CHK-ADD-TARGETS-UB: 3: input, "dummy.spv", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-UB: 4: clang-offload-wrapper, {3}, object, (device-sycl)
// CHK-ADD-TARGETS-UB: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown)" {4}, image

/// Check offload with multiple triples, multiple binaries passed through -fsycl-add-targets
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-add-targets=spir64-unknown-unknown:dummy.spv,spir64_fpga-unknown-unknown:dummy.aocx,spir64_gen-unknown-unknown:dummy_Gen9core.bin %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-ADD-TARGETS-MUL %s
// CHK-ADD-TARGETS-MUL: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHK-ADD-TARGETS-MUL: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-ADD-TARGETS-MUL: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-ADD-TARGETS-MUL: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-ADD-TARGETS-MUL: 4: compiler, {3}, ir, (device-sycl)
// CHK-ADD-TARGETS-MUL: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-unknown)" {4}, c++-cpp-output
// CHK-ADD-TARGETS-MUL: 6: compiler, {5}, ir, (host-sycl)
// CHK-ADD-TARGETS-MUL: 7: backend, {6}, assembler, (host-sycl)
// CHK-ADD-TARGETS-MUL: 8: assembler, {7}, object, (host-sycl)
// CHK-ADD-TARGETS-MUL: 9: linker, {8}, image, (host-sycl)
// CHK-ADD-TARGETS-MUL: 10: input, "dummy.spv", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-MUL: 11: clang-offload-wrapper, {10}, object, (device-sycl)
// CHK-ADD-TARGETS-MUL: 12: input, "dummy.aocx", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-MUL: 13: clang-offload-wrapper, {12}, object, (device-sycl)
// CHK-ADD-TARGETS-MUL: 14: input, "dummy_Gen9core.bin", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-MUL: 15: clang-offload-wrapper, {14}, object, (device-sycl)
// CHK-ADD-TARGETS-MUL: 16: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64-unknown-unknown)" {11}, "device-sycl (spir64_fpga-unknown-unknown)" {13}, "device-sycl (spir64_gen-unknown-unknown)" {15}, image

/// Check offload with single triple, multiple binaries passed through -fsycl-add-targets
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-add-targets=spir64-unknown-unknown:dummy0.spv,spir64-unknown-unknown:dummy1.spv,spir64-unknown-unknown:dummy2.spv %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-ADD-TARGETS-MUL-BINS %s
// CHK-ADD-TARGETS-MUL-BINS: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 4: compiler, {3}, ir, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-unknown)" {4}, c++-cpp-output
// CHK-ADD-TARGETS-MUL-BINS: 6: compiler, {5}, ir, (host-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 7: backend, {6}, assembler, (host-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 8: assembler, {7}, object, (host-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 9: linker, {8}, image, (host-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 10: input, "dummy0.spv", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 11: clang-offload-wrapper, {10}, object, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 12: input, "dummy1.spv", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 13: clang-offload-wrapper, {12}, object, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 14: input, "dummy2.spv", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 15: clang-offload-wrapper, {14}, object, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 16: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64-unknown-unknown)" {11}, "device-sycl (spir64-unknown-unknown)" {13}, "device-sycl (spir64-unknown-unknown)" {15}, image

/// Check regular offload with an additional AOT binary passed through -fsycl-add-targets (same triple)
// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64-unknown-unknown -fsycl-add-targets=spir64-unknown-unknown:dummy.spv -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-ADD-TARGETS-REG %s
// CHK-ADD-TARGETS-REG: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHK-ADD-TARGETS-REG: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-ADD-TARGETS-REG: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-ADD-TARGETS-REG: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-ADD-TARGETS-REG: 4: compiler, {3}, ir, (device-sycl)
// CHK-ADD-TARGETS-REG: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-unknown)" {4}, c++-cpp-output
// CHK-ADD-TARGETS-REG: 6: compiler, {5}, ir, (host-sycl)
// CHK-ADD-TARGETS-REG: 7: backend, {6}, assembler, (host-sycl)
// CHK-ADD-TARGETS-REG: 8: assembler, {7}, object, (host-sycl)
// CHK-ADD-TARGETS-REG: 9: linker, {4}, ir, (device-sycl)
// CHK-ADD-TARGETS-REG: 10: sycl-post-link, {9}, tempfiletable, (device-sycl)
// CHK-ADD-TARGETS-REG: 11: file-table-tform, {10}, tempfilelist, (device-sycl)
// CHK-ADD-TARGETS-REG: 12: llvm-spirv, {11}, tempfilelist, (device-sycl)
// CHK-ADD-TARGETS-REG: 13: file-table-tform, {10, 12}, tempfiletable, (device-sycl)
// CHK-ADD-TARGETS-REG: 14: clang-offload-wrapper, {13}, object, (device-sycl)
// CHK-ADD-TARGETS-REG: 15: offload, "device-sycl (spir64-unknown-unknown)" {14}, object
// CHK-ADD-TARGETS-REG: 16: linker, {8, 15}, image, (host-sycl)
// CHK-ADD-TARGETS-REG: 17: input, "dummy.spv", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-REG: 18: clang-offload-wrapper, {17}, object, (device-sycl)
// CHK-ADD-TARGETS-REG: 19: offload, "host-sycl (x86_64-unknown-linux-gnu)" {16}, "device-sycl (spir64-unknown-unknown)" {18}, image

/// Check regular offload with multiple additional AOT binaries passed through -fsycl-add-targets
// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64-unknown-unknown -fsycl-add-targets=spir64_fpga-unknown-unknown:dummy.aocx,spir64_gen-unknown-unknown:dummy_Gen9core.bin,spir64_x86_64-unknown-unknown:dummy.ir -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-ADD-TARGETS-REG-MUL %s
// CHK-ADD-TARGETS-REG-MUL: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHK-ADD-TARGETS-REG-MUL: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-ADD-TARGETS-REG-MUL: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 4: compiler, {3}, ir, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-unknown)" {4}, c++-cpp-output
// CHK-ADD-TARGETS-REG-MUL: 6: compiler, {5}, ir, (host-sycl)
// CHK-ADD-TARGETS-REG-MUL: 7: backend, {6}, assembler, (host-sycl)
// CHK-ADD-TARGETS-REG-MUL: 8: assembler, {7}, object, (host-sycl)
// CHK-ADD-TARGETS-REG-MUL: 9: linker, {4}, ir, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 10: sycl-post-link, {9}, tempfiletable, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 11: file-table-tform, {10}, tempfilelist, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 12: llvm-spirv, {11}, tempfilelist, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 13: file-table-tform, {10, 12}, tempfiletable, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 14: clang-offload-wrapper, {13}, object, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 15: offload, "device-sycl (spir64-unknown-unknown)" {14}, object
// CHK-ADD-TARGETS-REG-MUL: 16: linker, {8, 15}, image, (host-sycl)
// CHK-ADD-TARGETS-REG-MUL: 17: input, "dummy.aocx", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 18: clang-offload-wrapper, {17}, object, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 19: input, "dummy_Gen9core.bin", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 20: clang-offload-wrapper, {19}, object, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 21: input, "dummy.ir", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 22: clang-offload-wrapper, {21}, object, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 23: offload, "host-sycl (x86_64-unknown-linux-gnu)" {16}, "device-sycl (spir64_fpga-unknown-unknown)" {18}, "device-sycl (spir64_gen-unknown-unknown)" {20}, "device-sycl (spir64_x86_64-unknown-unknown)" {22}, image
