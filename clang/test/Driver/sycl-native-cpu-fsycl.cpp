//RUN: %clang -fsycl -fsycl-native-cpu -ccc-print-phases %s 2>&1 | FileCheck %s --check-prefix=CHECK_ACTIONS
//RUN: %clang -fsycl -fsycl-native-cpu -ccc-print-bindings %s 2>&1 | FileCheck %s --check-prefix=CHECK_BINDINGS
//RUN: %clang -fsycl -fsycl-native-cpu -### %s 2>&1 | FileCheck %s --check-prefix=CHECK_INVO
//RUN: %clang -fsycl -fsycl-native-cpu -target aarch64-unknown-linux-gnu -ccc-print-phases %s 2>&1 | FileCheck %s --check-prefix=CHECK_ACTIONS-AARCH64


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
//this is where we compile the device code to a shared lib, and we link the host shared lib and the device shared lib
//CHECK_ACTIONS:|     +- 11: linker, {5}, ir, (device-sycl)
//CHECK_ACTIONS:|  +- 12: backend, {11}, assembler, (device-sycl)
//CHECK_ACTIONS:|- 13: assembler, {12}, object, (device-sycl)
//CHECK_ACTIONS:14: offload, "host-sycl ({{.*}})" {10}, "device-sycl ({{.*}})" {13}, image


//CHECK_BINDINGS:# "{{.*}}" - "clang", inputs: ["{{.*}}sycl-native-cpu-fsycl.cpp"], output: "[[KERNELIR:.*]].bc"
//CHECK_BINDINGS:# "{{.*}}" - "SYCL::Linker", inputs: ["[[KERNELIR]].bc"], output: "[[KERNELLINK:.*]].bc"
//CHECK_BINDINGS:# "{{.*}}" - "clang", inputs: ["[[KERNELLINK]].bc"], output: "[[KERNELOBJ:.*]].o"
//CHECK_BINDINGS:# "{{.*}}" - "Append Footer to source", inputs: ["{{.*}}sycl-native-cpu-fsycl.cpp"], output: "[[SRCWFOOTER:.*]].cpp"
//CHECK_BINDINGS:# "{{.*}}" - "clang", inputs: ["[[SRCWFOOTER]].cpp", "[[KERNELIR]].bc"], output: "[[HOSTOBJ:.*]].o"
//CHECK_BINDINGS:# "{{.*}}" - "GNU::Linker", inputs: ["[[HOSTOBJ]].o", "[[KERNELOBJ]].o"], output: "a.out"

//CHECK_INVO:{{.*}}clang{{.*}}-fsycl-is-device{{.*}}"-mllvm" "-sycl-native-cpu" "-D" "__SYCL_NATIVE_CPU__" 
//CHECK_INVO:{{.*}}clang{{.*}}"-x" "ir"
//CHECK_INVO:{{.*}}clang{{.*}}"-fsycl-is-host"{{.*}}

// checkes that the device and host triple is correct in the generated actions when it is set explicitly
//CHECK_ACTIONS-AARCH64:                     +- 0: input, "{{.*}}sycl-native-cpu-fsycl.cpp", c++, (host-sycl)
//CHECK_ACTIONS-AARCH64:                  +- 1: append-footer, {0}, c++, (host-sycl)
//CHECK_ACTIONS-AARCH64:               +- 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
//CHECK_ACTIONS-AARCH64:               |     +- 3: input, "{{.*}}sycl-native-cpu-fsycl.cpp", c++, (device-sycl)
//CHECK_ACTIONS-AARCH64:               |  +- 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
//CHECK_ACTIONS-AARCH64:               |- 5: compiler, {4}, ir, (device-sycl)
//CHECK_ACTIONS-AARCH64:            +- 6: offload, "host-sycl (aarch64-unknown-linux-gnu)" {2}, "device-sycl (aarch64-unknown-linux-gnu)" {5}, c++-cpp-output
//CHECK_ACTIONS-AARCH64:         +- 7: compiler, {6}, ir, (host-sycl)
//CHECK_ACTIONS-AARCH64:      +- 8: backend, {7}, assembler, (host-sycl)
//CHECK_ACTIONS-AARCH64:   +- 9: assembler, {8}, object, (host-sycl)
//CHECK_ACTIONS-AARCH64:+- 10: linker, {9}, image, (host-sycl)
//CHECK_ACTIONS-AARCH64:|     +- 11: linker, {5}, ir, (device-sycl)
//CHECK_ACTIONS-AARCH64:|  +- 12: backend, {11}, assembler, (device-sycl)
//CHECK_ACTIONS-AARCH64:|- 13: assembler, {12}, object, (device-sycl)
//CHECK_ACTIONS-AARCH64:14: offload, "host-sycl (aarch64-unknown-linux-gnu)" {10}, "device-sycl (aarch64-unknown-linux-gnu)" {13}, image
