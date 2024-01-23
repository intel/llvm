//RUN: %clang -fsycl -fsycl-targets=native_cpu -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -ccc-print-phases %s 2>&1 | FileCheck %s --check-prefix=CHECK_ACTIONS
//RUN: %clang -fsycl -fsycl-targets=native_cpu -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -ccc-print-bindings %s 2>&1 | FileCheck %s --check-prefix=CHECK_BINDINGS
//RUN: %clang -fsycl -fsycl-targets=native_cpu -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK_INVO
//RUN: %clang -fsycl -fsycl-targets=native_cpu -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -target aarch64-unknown-linux-gnu -ccc-print-phases %s 2>&1 | FileCheck %s --check-prefix=CHECK_ACTIONS-AARCH64


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
//CHECK_ACTIONS:+- 10: linker, {9}, image, (host-sycl)
//CHECK_ACTIONS:        +- 11: linker, {5}, ir, (device-sycl)
//CHECK_ACTIONS:        |- 75: input, "{{.*}}libspirv{{.*}}", ir, (device-sycl)
//CHECK_ACTIONS:     +- 76: linker, {11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 75}, ir, (device-sycl)
//this is where we compile the device code to a shared lib, and we link the host shared lib and the device shared lib
//CHECK_ACTIONS:|  +- 81: backend, {80}, assembler, (device-sycl)
//CHECK_ACTIONS:|- 82: assembler, {81}, object, (device-sycl)
//call sycl-post-link and clang-offload-wrapper
//CHECK_ACTIONS:|  +- 83: sycl-post-link, {76}, tempfiletable, (device-sycl)
//CHECK_ACTIONS:|- 84: clang-offload-wrapper, {83}, object, (device-sycl)
//CHECK_ACTIONS:85: offload, "host-sycl ({{.*}})" {10}, "device-sycl ({{.*}})" {82}, "device-sycl ({{.*}})" {84}, image


//CHECK_BINDINGS:# "{{.*}}" - "clang", inputs: ["{{.*}}sycl-native-cpu-fsycl.cpp"], output: "[[KERNELIR:.*]].bc"
//CHECK_BINDINGS:# "{{.*}}" - "SYCL::Linker", inputs: ["[[KERNELIR]].bc"], output: "[[KERNELLINK:.*]].bc"
//CHECK_BINDINGS:# "{{.*}}" - "SYCL::Linker", inputs: ["[[KERNELLINK]].bc", "{{.*}}.bc"], output: "[[KERNELLINKWLIB:.*]].bc"
//CHECK_BINDINGS:# "{{.*}}" - "clang", inputs: ["[[KERNELLINKWLIB2:.*]].bc"], output: "[[KERNELOBJ:.*]].o"
//CHECK_BINDINGS:# "{{.*}}" - "SYCL post link", inputs: ["[[KERNELLINKWLIB]].bc"], output: "[[TABLEFILE:.*]].table"
//CHECK_BINDINGS:# "{{.*}}" - "offload wrapper", inputs: ["[[TABLEFILE]].table"], output: "[[WRAPPEROBJ:.*]].o"
//CHECK_BINDINGS:# "{{.*}}" - "Append Footer to source", inputs: ["{{.*}}sycl-native-cpu-fsycl.cpp"], output: "[[SRCWFOOTER:.*]].cpp"
//CHECK_BINDINGS:# "{{.*}}" - "clang", inputs: ["[[SRCWFOOTER]].cpp", "[[KERNELIR]].bc"], output: "[[HOSTOBJ:.*]].o"
//CHECK_BINDINGS:# "{{.*}}" - "{{.*}}::Linker", inputs: ["[[HOSTOBJ]].o", "[[KERNELOBJ]].o", "[[WRAPPEROBJ]].o"], output: "a.{{.*}}"

//CHECK_INVO:{{.*}}clang{{.*}}-fsycl-is-device{{.*}}"-fsycl-is-native-cpu" "-D" "__SYCL_NATIVE_CPU__" 
//CHECK_INVO:{{.*}}clang{{.*}}"-x" "ir"
//CHECK_INVO:{{.*}}clang{{.*}}"-fsycl-is-host"{{.*}}

// checks that the device and host triple is correct in the generated actions when it is set explicitly
//CHECK_ACTIONS-AARCH64:            +- 6: offload, "host-sycl (aarch64-unknown-linux-gnu)" {2}, "device-sycl (aarch64-unknown-linux-gnu)" {5}, c++-cpp-output
//CHECK_ACTIONS-AARCH64:{{[0-9]*}}: offload, "host-sycl (aarch64-unknown-linux-gnu)" {{{[0-9]*}}}, "device-sycl (aarch64-unknown-linux-gnu)" {{{[0-9]*}}}, "device-sycl (aarch64-unknown-linux-gnu)" {{{[0-9]*}}}, image
