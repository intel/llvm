// UNSUPPORTED: system-windows

/// -fintelfpga static lib (aoco)
// RUN:  echo "Dummy AOCO image" > %t.aoco
// RUN:  echo "void foo() {}" > %t.c
// RUN:  echo "void foo2() {}" > %t2.c
// RUN:  %clang -c -o %t.o %t.c
// RUN:  %clang -fsycl -c -o %t2.o %t2.c
// RUN:  %clang_cl -fsycl -c -o %t2_cl.o %t2.c
// RUN:  clang-offload-wrapper -o %t-aoco.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=fpga_aoco-intel-unknown-sycldevice %t.aoco
// RUN:  llc -filetype=obj -o %t-aoco.o %t-aoco.bc
// RUN:  clang-offload-wrapper -o %t-aoco_cl.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=fpga_aoco-intel-unknown-sycldevice %t.aoco
// RUN:  llc -filetype=obj -o %t-aoco_cl.o %t-aoco_cl.bc
// RUN:  llvm-ar crv %t_aoco.a %t.o %t2.o %t-aoco.o
// RUN:  llvm-ar crv %t_aoco_cl.a %t.o %t2_cl.o %t-aoco_cl.o
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga %t_aoco.a %s -### -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-AOCO-PHASES %s
// CHK-FPGA-AOCO-PHASES: 0: input, "[[INPUTA:.+\.a]]", object, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 1: input, "[[INPUTCPP:.+\.cpp]]", c++, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 3: input, "[[INPUTCPP]]", c++, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 5: compiler, {4}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// CHK-FPGA-AOCO-PHASES: 7: compiler, {6}, ir, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 8: backend, {7}, assembler, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 9: assembler, {8}, object, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 10: linker, {0, 9}, image, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 11: linker, {0, 9}, image, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 12: clang-offload-deps, {11}, ir, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 13: input, "[[INPUTA]]", archive
// CHK-FPGA-AOCO-PHASES: 14: clang-offload-unbundler, {13}, archive
// CHK-FPGA-AOCO-PHASES: 15: linker, {5, 12, 14}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 16: sycl-post-link, {15}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 17: llvm-spirv, {16}, spirv, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 18: input, "[[INPUTA]]", archive
// CHK-FPGA-AOCO-PHASES: 19: clang-offload-unbundler, {18}, fpga_dependencies_list
// CHK-FPGA-AOCO-PHASES: 20: input, "[[INPUTA]]", fpga_aoco
// CHK-FPGA-AOCO-PHASES: 21: clang-offload-unbundler, {20}, fpga_aoco
// CHK-FPGA-AOCO-PHASES: 22: backend-compiler, {17, 19, 21}, fpga_aocx, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 23: clang-offload-wrapper, {22}, object, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 24: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {23}, image

/// FPGA AOCO Windows phases check
// RUN:  %clang_cl -fsycl -fno-sycl-device-lib=all -fintelfpga -foffload-static-lib=%t_aoco_cl.a %s -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO-PHASES-WIN %s
// CHK-FPGA-AOCO-PHASES-WIN: 0: input, "[[INPUTA:.+\.a]]", object, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 1: input, "[[INPUTSRC:.+\.cpp]]", c++, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 3: input, "[[INPUTSRC]]", c++, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 5: compiler, {4}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 6: offload, "host-sycl (x86_64-pc-windows-msvc)" {2}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// CHK-FPGA-AOCO-PHASES-WIN: 7: compiler, {6}, ir, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 8: backend, {7}, assembler, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 9: assembler, {8}, object, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 10: linker, {0, 9}, image, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 11: linker, {0, 9}, image, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 12: clang-offload-deps, {11}, ir, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 13: input, "[[INPUTA]]", archive
// CHK-FPGA-AOCO-PHASES-WIN: 14: clang-offload-unbundler, {13}, archive
// CHK-FPGA-AOCO-PHASES-WIN: 15: linker, {5, 12, 14}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 16: sycl-post-link, {15}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 17: llvm-spirv, {16}, spirv, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 18: input, "[[INPUTA]]", archive
// CHK-FPGA-AOCO-PHASES-WIN: 19: clang-offload-unbundler, {18}, fpga_dependencies_list
// CHK-FPGA-AOCO-PHASES-WIN: 20: input, "[[INPUTA]]", fpga_aoco
// CHK-FPGA-AOCO-PHASES-WIN: 21: clang-offload-unbundler, {20}, fpga_aoco
// CHK-FPGA-AOCO-PHASES-WIN: 22: backend-compiler, {17, 19, 21}, fpga_aocx, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 23: clang-offload-wrapper, {22}, object, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 24: offload, "host-sycl (x86_64-pc-windows-msvc)" {10}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {23}, image

/// aoco test, checking tools
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga -foffload-static-lib=%t_aoco.a -### %s 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO,CHK-FPGA-AOCO-LIN %s
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga %t_aoco.a -### %s 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO,CHK-FPGA-AOCO-LIN %s
// RUN:  %clang_cl -fsycl -fno-sycl-device-lib=all -fintelfpga -foffload-static-lib=%t_aoco_cl.a -### %s 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO,CHK-FPGA-AOCO-WIN %s
// RUN:  %clang_cl -fsycl -fno-sycl-device-lib=all -fintelfpga %t_aoco_cl.a -### %s 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO,CHK-FPGA-AOCO-WIN %s
// CHK-FPGA-AOCO: clang-offload-bundler{{.*}} "-type=a" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice" "-inputs=[[INPUTLIB:.+\.a]]" "-outputs=[[OUTLIB:.+\.a]]" "-unbundle"
// CHK-FPGA-AOCO: llvm-link{{.*}} "[[OUTLIB]]" "-o" "[[LINKEDBC:.+\.bc]]"
// CHK-FPGA-AOCO: sycl-post-link
// CHK-FPGA-AOCO-NOT: "-split-esimd"
// CHK-FPGA-AOCO: "-ir-output-only" "-O2" "-spec-const=default" "-o" "[[PLINKEDBC:.+\.bc]]" "[[LINKEDBC]]"
// CHK-FPGA-AOCO: llvm-spirv{{.*}} "-o" "[[TARGSPV:.+\.spv]]" {{.*}} "[[PLINKEDBC]]"
// CHK-FPGA-AOCO: clang-offload-bundler{{.*}} "-type=aoo" "-targets=sycl-fpga_aoco-intel-unknown-sycldevice" "-inputs=[[INPUTLIB]]" "-outputs=[[AOCOLIST:.+\.txt]]" "-unbundle"
// CHK-FPGA-AOCO: aoc{{.*}} "-o" "[[AOCXOUT:.+\.aocx]]" "[[TARGSPV]]" "-library-list=[[AOCOLIST]]" "-sycl"
// CHK-FPGA-AOCO: clang-offload-wrapper{{.*}} "-o=[[FINALBC:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "[[AOCXOUT]]"
// CHK-FPGA-AOCO-LIN: llc{{.*}} "-filetype=obj" "-o" "[[FINALOBJL:.+\.o]]" "[[FINALBC]]"
// CHK-FPGA-AOCO-WIN: llc{{.*}} "-filetype=obj" "-o" "[[FINALOBJW:.+\.obj]]" "[[FINALBC]]"
// CHK-FPGA-AOCO-LIN: ld{{.*}} "[[INPUTLIB]]" {{.*}} "[[FINALOBJL]]"
// CHK-FPGA-AOCO-WIN: link.exe{{.*}} "{{.*}}[[INPUTLIB]]" {{.*}} "[[FINALOBJW]]"
