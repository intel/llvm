//UNSUPPORTED: system-windows
//RUN: %clang -fsycl --sysroot=%S/Inputs/SYCL -fsycl-targets=native_cpu -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -ccc-print-phases %s 2>&1 | FileCheck %s --check-prefix=CHECK_ACTIONS
//RUN: %clang -fsycl --sysroot=%S/Inputs/SYCL -fsycl-targets=native_cpu -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -ccc-print-bindings %s 2>&1 | FileCheck %s --check-prefix=CHECK_BINDINGS
//RUN: %clang -fsycl --sysroot=%S/Inputs/SYCL -fsycl-targets=native_cpu -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK_INVO
//RUN: %clang -fsycl --sysroot=%S/Inputs/SYCL -fsycl-targets=native_cpu -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -target aarch64-unknown-linux-gnu -ccc-print-phases %s 2>&1 | FileCheck %s --check-prefix=CHECK_ACTIONS-AARCH64

//Link together multiple TUs.
//RUN: touch %t_1.o
//RUN: touch %t_2.o
//RUN: %clang -fsycl -fsycl-targets=native_cpu --sysroot=%S/Inputs/SYCL -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc %t_1.o %t_2.o -ccc-print-bindings 2>&1 | FileCheck %s --check-prefix=CHECK_BINDINGS_MULTI_TU

//CHECK_ACTIONS:               +- 0: input, "{{.*}}sycl-native-cpu-fsycl.cpp", c++, (host-sycl)
//CHECK_ACTIONS:            +- 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
//CHECK_ACTIONS:            |     +- 2: input, "{{.*}}sycl-native-cpu-fsycl.cpp", c++, (device-sycl)
//CHECK_ACTIONS:            |  +- 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
//CHECK_ACTIONS:            |- 4: compiler, {3}, ir, (device-sycl)
//CHECK_ACTIONS:         +- 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (native_cpu)" {4}, c++-cpp-output
//CHECK_ACTIONS:      +- 6: compiler, {5}, ir, (host-sycl)
//CHECK_ACTIONS:   +- 7: backend, {6}, assembler, (host-sycl)
//CHECK_ACTIONS:+- 8: assembler, {7}, object, (host-sycl)
//CHECK_ACTIONS:|              +- 9: linker, {4}, ir, (device-sycl)
//CHECK_ACTIONS:|              |- [[SPIRVLIB:.*]]: input, "{{.*}}libspirv{{.*}}", ir, (device-sycl)
//different libraries may be linked on different platforms, so just check the common stages
//CHECK_ACTIONS:|           +- [[LINKALL:.*]]: linker, {9, [[SPIRVLIB]]}, ir, (device-sycl)
//CHECK_ACTIONS:|           |- [[NCPUIMP:.*]]: input, "{{.*}}nativecpu{{.*}}", ir, (device-sycl)
//CHECK_ACTIONS:|        +- [[NCPULINK:.*]]: linker, {[[LINKALL]], [[NCPUIMP]]}, ir, (device-sycl)
//this is where we compile the device code to a shared lib, and we link the host shared lib and the device shared lib
//CHECK_ACTIONS:|     +- [[VAL81:.*]]: backend, {[[NCPULINK]]}, assembler, (device-sycl)
//CHECK_ACTIONS:|  +- [[VAL82:.*]]: assembler, {[[VAL81]]}, object, (device-sycl)
//CHECK_ACTIONS:|- [[VAL822:.*]]: offload, "device-sycl (native_cpu)" {[[VAL82]]}, object
//call sycl-post-link and clang-offload-wrapper
//CHECK_ACTIONS:|     +- [[VAL83:.*]]: sycl-post-link, {[[LINKALL]]}, tempfiletable, (device-sycl)
//CHECK_ACTIONS:|  +- [[VAL84:.*]]: clang-offload-wrapper, {[[VAL83]]}, object, (device-sycl)
//CHECK_ACTIONS:|- [[VAL85:.*]]: offload, "device-sycl ({{.*}})" {[[VAL84]]}, object
//CHECK_ACTIONS:[[VAL86:.*]]: linker, {8, [[VAL822]], [[VAL85]]}, image, (host-sycl)

//CHECK_BINDINGS:# "{{.*}}" - "clang", inputs: ["{{.*}}sycl-native-cpu-fsycl.cpp"], output: "[[KERNELIR:.*]].bc"
//CHECK_BINDINGS:# "{{.*}}" - "clang", inputs: ["{{.*}}sycl-native-cpu-fsycl.cpp", "[[KERNELIR]].bc"], output: "[[HOSTOBJ:.*]].o"
//CHECK_BINDINGS:# "{{.*}}" - "SYCL::Linker", inputs: ["[[KERNELIR]].bc"], output: "[[KERNELLINK:.*]].bc"
//CHECK_BINDINGS:# "{{.*}}" - "SYCL::Linker", inputs: ["[[KERNELLINK]].bc", "{{.*}}.bc"], output: "[[KERNELLINKWLIB:.*]].bc"
//CHECK_BINDINGS:# "{{.*}}" - "SYCL::Linker", inputs: ["[[KERNELLINKWLIB]].bc", "[[UNBUNDLEDNCPU:.*]].bc"], output: "[[KERNELLINKWLIB12:.*]].bc"
//CHECK_BINDINGS:# "{{.*}}" - "clang", inputs: ["[[KERNELLINKWLIB12]].bc"], output: "[[KERNELOBJ:.*]].o"
//CHECK_BINDINGS:# "{{.*}}" - "SYCL post link", inputs: ["[[KERNELLINKWLIB]].bc"], output: "[[TABLEFILE:.*]].table"
//CHECK_BINDINGS:# "{{.*}}" - "offload wrapper", inputs: ["[[TABLEFILE]].table"], output: "[[WRAPPEROBJ:.*]].o"
//CHECK_BINDINGS:# "{{.*}}" - "{{.*}}::Linker", inputs: ["[[HOSTOBJ]].o", "[[KERNELOBJ]].o", "[[WRAPPEROBJ]].o"], output: "a.{{.*}}"

//CHECK_INVO:{{.*}}clang{{.*}}-fsycl-is-device{{.*}}"-fsycl-is-native-cpu" "-D" "__SYCL_NATIVE_CPU__"
//CHECK_INVO:{{.*}}clang{{.*}}"-fsycl-is-host"{{.*}}
//CHECK_INVO:{{.*}}clang{{.*}}"-x" "ir"
//CHECK_INVO:{{.*}}sycl-post-link{{.*}}"-emit-program-metadata"

// checks that the device and host triple is correct in the generated actions when it is set explicitly
//CHECK_ACTIONS-AARCH64:        +- 5: offload, "host-sycl (aarch64-unknown-linux-gnu)" {1}, "device-sycl (native_cpu)" {4}, c++-cpp-output
//CHECK_ACTIONS-AARCH64:|- 16: offload, "device-sycl (native_cpu)" {15}, object
//CHECK_ACTIONS-AARCH64:|- 19: offload, "device-sycl (native_cpu)" {18}, object

// checks that bindings are correct when linking together multiple TUs on native cpu
//CHECK_BINDINGS_MULTI_TU:# "{{.*}}" - "offload bundler", inputs: ["{{.*}}.o"], outputs: ["[[FILE1HOST:.*\.o]]", "{{.*\.o}}"]
//CHECK_BINDINGS_MULTI_TU:# "{{.*}}" - "offload bundler", inputs: ["{{.*}}.o"], outputs: ["[[FILE2HOST:.*\.o]]", "{{.*\.o}}"]
//CHECK_BINDINGS_MULTI_TU:# "native_cpu" - "offload bundler", inputs: ["{{.*}}.o"], outputs: ["{{.*\.o}}", "[[FILE1DEV:.*\.o]]"]
//CHECK_BINDINGS_MULTI_TU:# "native_cpu" - "Convert SPIR-V to LLVM-IR if needed", inputs: ["[[FILE1DEV]]"], output: "[[FILE1SPV:.*\.bc]]"
//CHECK_BINDINGS_MULTI_TU:# "native_cpu" - "offload bundler", inputs: ["{{.*}}.o"], outputs: ["{{.*\.o}}", "[[FILE2DEV:.*\.o]]"]
//CHECK_BINDINGS_MULTI_TU:# "native_cpu" - "Convert SPIR-V to LLVM-IR if needed", inputs: ["[[FILE2DEV]]"], output: "[[FILE2SPV:.*\.bc]]"
//CHECK_BINDINGS_MULTI_TU:# "native_cpu" - "SYCL::Linker", inputs: ["[[FILE1SPV]]", "[[FILE2SPV]]"], output: "[[LINK1:.*\.bc]]"
//CHECK_BINDINGS_MULTI_TU:# "native_cpu" - "SYCL::Linker", inputs: ["[[LINK1]]", "{{.*\.bc}}"], output: "[[LINK2:.*\.bc]]"
//CHECK_BINDINGS_MULTI_TU:# "native_cpu" - "clang", inputs: ["{{.*}}.bc"], output: "[[KERNELO:.*\.o]]"
//CHECK_BINDINGS_MULTI_TU:# "native_cpu" - "SYCL post link", inputs: ["[[LINK2]]"], output: "[[POSTL:.*\.table]]"
//CHECK_BINDINGS_MULTI_TU:# "native_cpu" - "offload wrapper", inputs: ["[[POSTL]]"], output: "[[WRAP:.*\.o]]"
//CHECK_BINDINGS_MULTI_TU:# "{{.*}}" - "{{.*}}::Linker", inputs: ["[[FILE1HOST]]", "[[FILE2HOST]]", "[[KERNELO]]", "[[WRAP]]"], output: "{{.*}}"
