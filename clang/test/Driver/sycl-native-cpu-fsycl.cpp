//RUN: %clang -fsycl -fsycl-targets=native_cpu -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -ccc-print-phases %s 2>&1 | FileCheck %s --check-prefix=CHECK_ACTIONS
//RUN: %clang -fsycl -fsycl-targets=native_cpu -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -ccc-print-bindings %s 2>&1 | FileCheck %s --check-prefix=CHECK_BINDINGS
//RUN: %clang -fsycl -fsycl-targets=native_cpu -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK_INVO
//RUN: %clang -fsycl -fsycl-targets=native_cpu -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -target aarch64-unknown-linux-gnu -ccc-print-phases %s 2>&1 | FileCheck %s --check-prefix=CHECK_ACTIONS-AARCH64

//Link together multiple TUs.
//RUN: touch %t_1.o 
//RUN: touch %t_2.o 
//RUN: %clang -fsycl -fsycl-targets=native_cpu -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc %t_1.o %t_2.o -ccc-print-bindings 2>&1 | FileCheck %s --check-prefix=CHECK_BINDINGS_MULTI_TU


//CHECK_ACTIONS:                     +- 0: input, "{{.*}}sycl-native-cpu-fsycl.cpp", c++, (host-sycl)
//CHECK_ACTIONS:                  +- 1: append-footer, {0}, c++, (host-sycl)
//CHECK_ACTIONS:               +- 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
//CHECK_ACTIONS:               |     +- 3: input, "{{.*}}sycl-native-cpu-fsycl.cpp", c++, (device-sycl)
//CHECK_ACTIONS:               |  +- 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
//CHECK_ACTIONS:               |- 5: compiler, {4}, ir, (device-sycl)
//CHECK_ACTIONS:            +- 6: offload, "host-sycl ({{.*}})" {2}, "device-sycl ({{.*}})" {5}, c++-cpp-output
//CHECK_ACTIONS:         +- 7: compiler, {6}, ir, (host-sycl)
//CHECK_ACTIONS:      +- 8: backend, {7}, assembler, (host-sycl)
//CHECK_ACTIONS:   +- 9: assembler, {8}, object, (host-sycl)

//CHECK_ACTIONS:|           +- 10: linker, {5}, ir, (device-sycl)
//CHECK_ACTIONS:|           |- 11: input, "{{.*}}libspirv{{.*}}", ir, (device-sycl)
//CHECK_ACTIONS:|        +- 12: linker, {10, 11}, ir, (device-sycl)
//CHECK_ACTIONS:|     +- 13: backend, {12}, assembler, (device-sycl)
//CHECK_ACTIONS:|  +- 14: assembler, {13}, object, (device-sycl)
//CHECK_ACTIONS:|- 15: offload, "device-sycl ({{.*}})" {14}, object
//CHECK_ACTIONS:|     +- 16: sycl-post-link, {12}, tempfiletable, (device-sycl)
//CHECK_ACTIONS:|  +- 17: clang-offload-wrapper, {16}, object, (device-sycl)
//CHECK_ACTIONS:|- 18: offload, "device-sycl ({{.*}})" {17}, object
//CHECK_ACTIONS:19: linker, {9, 15, 18}, image, (host-sycl)

//CHECK_BINDINGS:# "{{.*}}" - "clang", inputs: ["{{.*}}sycl-native-cpu-fsycl.cpp"], output: "[[KERNELIR:.*]].bc"
//CHECK_BINDINGS:# "{{.*}}" - "Append Footer to source", inputs: ["{{.*}}sycl-native-cpu-fsycl.cpp"], output: "[[SRCWFOOTER:.*]].cpp"
//CHECK_BINDINGS:# "{{.*}}" - "clang", inputs: ["[[SRCWFOOTER]].cpp", "[[KERNELIR]].bc"], output: "[[HOSTOBJ:.*]].o"
//CHECK_BINDINGS:# "{{.*}}" - "SYCL::Linker", inputs: ["[[KERNELIR]].bc"], output: "[[KERNELLINK:.*]].bc"
//CHECK_BINDINGS:# "{{.*}}" - "SYCL::Linker", inputs: ["[[KERNELLINK]].bc", "{{.*}}.bc"], output: "[[KERNELLINKWLIB:.*]].bc"
//CHECK_BINDINGS:# "{{.*}}" - "clang", inputs: ["[[KERNELLINKWLIB]].bc"], output: "[[KERNELOBJ:.*]].o"
//CHECK_BINDINGS:# "{{.*}}" - "SYCL post link", inputs: ["[[KERNELLINKWLIB]].bc"], output: "[[TABLEFILE:.*]].table"
//CHECK_BINDINGS:# "{{.*}}" - "offload wrapper", inputs: ["[[TABLEFILE]].table"], output: "[[WRAPPEROBJ:.*]].o"
//CHECK_BINDINGS:# "{{.*}}" - "{{.*}}::Linker", inputs: ["[[HOSTOBJ]].o", "[[KERNELOBJ]].o", "[[WRAPPEROBJ]].o"], output: "a.{{.*}}"

//CHECK_INVO:{{.*}}clang{{.*}}-fsycl-is-device{{.*}}"-fsycl-is-native-cpu" "-D" "__SYCL_NATIVE_CPU__" 
//CHECK_INVO:{{.*}}clang{{.*}}"-fsycl-is-host"{{.*}}
//CHECK_INVO:{{.*}}clang{{.*}}"-x" "ir"
//CHECK_INVO:{{.*}}sycl-post-link{{.*}}"-emit-program-metadata"
//CHECK_INVO-NOT:{{.*}}sycl-post-link{{.*}}-emit-only-kernels-as-entry-points

// checks that the device and host triple is correct in the generated actions when it is set explicitly
//CHECK_ACTIONS-AARCH64:            +- 6: offload, "host-sycl (aarch64-unknown-linux-gnu)" {2}, "device-sycl (aarch64-unknown-linux-gnu)" {5}, c++-cpp-output
//CHECK_ACTIONS-AARCH64:|- 15: offload, "device-sycl (aarch64-unknown-linux-gnu)" {14}, object
//CHECK_ACTIONS-AARCH64:|- 18: offload, "device-sycl (aarch64-unknown-linux-gnu)" {17}, object

// checks that bindings are correct when linking together multiple TUs on native cpu
//CHECK_BINDINGS_MULTI_TU:# "{{.*}}" - "offload bundler", inputs: ["{{.*}}.o"], outputs: ["[[FILE1HOST:.*]].o", "[[FILE1DEV:.*]].o"] 
//CHECK_BINDINGS_MULTI_TU:# "{{.*}}" - "offload bundler", inputs: ["{{.*}}.o"], outputs: ["[[FILE2HOST:.*]].o", "[[FILE2DEV:.*]].o"] 
//CHECK_BINDINGS_MULTI_TU:# "{{.*}}" - "Convert SPIR-V to LLVM-IR if needed", inputs: ["[[FILE1DEV]].o"], output: "[[FILE1SPV:.*]].bc"
//CHECK_BINDINGS_MULTI_TU:# "{{.*}}" - "Convert SPIR-V to LLVM-IR if needed", inputs: ["[[FILE2DEV]].o"], output: "[[FILE2SPV:.*]].bc"
//CHECK_BINDINGS_MULTI_TU:# "{{.*}}" - "SYCL::Linker", inputs: ["[[FILE1SPV]].bc", "[[FILE2SPV]].bc"], output: "[[LINK1:.*]].bc"
//CHECK_BINDINGS_MULTI_TU:# "{{.*}}" - "SYCL::Linker", inputs: ["[[LINK1]].bc", "{{.*}}.bc"], output: "[[LINK2:.*]].bc"
//CHECK_BINDINGS_MULTI_TU:# "{{.*}}" - "clang", inputs: ["[[LINK2]].bc"], output: "[[KERNELO:.*]].o"
//CHECK_BINDINGS_MULTI_TU:# "{{.*}}" - "SYCL post link", inputs: ["[[LINK2]].bc"], output: "[[POSTL:.*]].table"
//CHECK_BINDINGS_MULTI_TU:# "{{.*}}" - "offload wrapper", inputs: ["[[POSTL]].table"], output: "[[WRAP:.*]].o"
//CHECK_BINDINGS_MULTI_TU:# "{{.*}}" - "{{.*}}::Linker", inputs: ["[[FILE1HOST]].o", "[[FILE2HOST]].o", "[[KERNELO]].o", "[[WRAP]].o"], output: "{{.*}}"

