/// -fintelfpga static lib (aoco)
// RUN:  echo "Dummy AOCO image" > %t.aoco
// RUN:  echo "void foo() {}" > %t.c
// RUN:  echo "void foo2() {}" > %t2.c
// RUN:  %clang -c -o %t.o %t.c
// RUN:  %clang_cl -fsycl -c -o %t2.o %t2.c
// RUN:  clang-offload-wrapper -o %t-aoco.bc -host=x86_64-pc-windows-msvc -kind=sycl -target=fpga_aoco-intel-unknown-sycldevice %t.aoco
// RUN:  llc -filetype=obj -o %t-aoco.o %t-aoco.bc
// RUN:  llvm-ar crv %t_aoco.a %t.o %t2.o %t-aoco.o
// RUN:  %clang_cl -fsycl -fno-sycl-device-lib=all -fintelfpga -foffload-static-lib=%t_aoco.a %s -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO-PHASES-WIN %s
// RUN:  %clangxx -target x86_64-pc-windows-msvc -fsycl -fno-sycl-device-lib=all -fintelfpga -foffload-static-lib=%t_aoco.a %s -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO-PHASES-WIN %s
// CHK-FPGA-AOCO-PHASES-WIN: 0: input, "[[INPUTA:.+\.a]]", object, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 1: input, "[[INPUTSRC:.+\.cpp]]", c++, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 3: input, "[[INPUTSRC]]", c++, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 5: compiler, {4}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 6: offload, "host-sycl (x86_64-pc-windows-msvc)" {2}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// CHK-FPGA-AOCO-PHASEES-WIN: 7: append-footer, {6}, c++, (host-sycl)
// CHK-FPGA-AOCO-PHASEES-WIN: 8: preprocessor, {7}, c++-cpp-output, (host-sycl)
// CHK-FPGA-AOCO-PHASEES-WIN: 9: compiler, {8}, ir, (host-sycl)
// CHK-FPGA-AOCO-PHASEES-WIN: 10: backend, {9}, assembler, (host-sycl)
// CHK-FPGA-AOCO-PHASEES-WIN: 11: assembler, {10}, object, (host-sycl)
// CHK-FPGA-AOCO-PHASEES-WIN: 12: linker, {0, 11}, image, (host-sycl)
// CHK-FPGA-AOCO-PHASEES-WIN: 13: linker, {0, 11}, image, (host-sycl)
// CHK-FPGA-AOCO-PHASEES-WIN: 14: clang-offload-deps, {13}, ir, (host-sycl)
// CHK-FPGA-AOCO-PHASEES-WIN: 15: input, "[[INPUTA]]", archive
// CHK-FPGA-AOCO-PHASEES-WIN: 16: clang-offload-unbundler, {15}, archive
// CHK-FPGA-AOCO-PHASEES-WIN: 17: linker, {5, 14, 16}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASEES-WIN: 18: sycl-post-link, {17}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASEES-WIN: 19: llvm-spirv, {18}, spirv, (device-sycl)
// CHK-FPGA-AOCO-PHASEES-WIN: 20: input, "[[INPUTA]]", archive
// CHK-FPGA-AOCO-PHASEES-WIN: 21: clang-offload-unbundler, {20}, fpga_dep_list
// CHK-FPGA-AOCO-PHASEES-WIN: 22: input, "[[INPUTA]]", fpga_aoco
// CHK-FPGA-AOCO-PHASEES-WIN: 23: clang-offload-unbundler, {22}, fpga_aoco
// CHK-FPGA-AOCO-PHASEES-WIN: 24: backend-compiler, {19, 21, 23}, fpga_aocx, (device-sycl)
// CHK-FPGA-AOCO-PHASEES-WIN: 25: clang-offload-wrapper, {24}, object, (device-sycl)
// CHK-FPGA-AOCO-PHASEES-WIN: 26: offload, "host-sycl (x86_64-pc-windows-msvc)" {12}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {25}, image

/// aoco test, checking tools
// RUN:  %clang_cl -fsycl -fno-sycl-device-lib=all -fintelfpga -foffload-static-lib=%t_aoco.a -Xshardware -### %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-AOCO %s
// RUN:  %clang_cl -fsycl -fno-sycl-device-lib=all -fintelfpga %t_aoco.a -Xshardware -### %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-AOCO %s
// CHK-FPGA-AOCO: clang-offload-bundler{{.*}} "-type=a" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice" "-inputs=[[INPUTLIB:.+\.a]]" "-outputs=[[OUTLIB:.+\.a]]" "-unbundle"
// CHK-FPGA-AOCO: llvm-link{{.*}} "[[OUTLIB]]" "-o" "[[LINKEDBC:.+\.bc]]"
// CHK-FPGA-AOCO: sycl-post-link
// CHK-FPGA-AOCO-NOT: -split-esimd
// CHK-FPGA-AOCO: "-ir-output-only" "-O2" "-spec-const=default" "-o" "[[PLINKEDBC:.+\.bc]]" "[[LINKEDBC]]"
// CHK-FPGA-AOCO: llvm-spirv{{.*}} "-o" "[[TARGSPV:.+\.spv]]" {{.*}} "[[PLINKEDBC]]"
// CHK-FPGA-AOCO: clang-offload-bundler{{.*}} "-type=aoo" "-targets=sycl-fpga_aoco-intel-unknown-sycldevice" "-inputs=[[INPUTLIB]]" "-outputs=[[AOCOLIST:.+\.txt]]" "-unbundle"
// CHK-FPGA-AOCO: aoc{{.*}} "-o" "[[AOCXOUT:.+\.aocx]]" "[[TARGSPV]]" "-library-list=[[AOCOLIST]]" "-sycl"
// CHK-FPGA-AOCO: clang-offload-wrapper{{.*}} "-o=[[FINALBC:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "[[AOCXOUT]]"
// CHK-FPGA-AOCO: llc{{.*}} "-filetype=obj" "-o" "[[FINALOBJW:.+\.obj]]" "[[FINALBC]]"
// CHK-FPGA-AOCO: link.exe{{.*}} "{{.*}}[[INPUTLIB]]" {{.*}} "[[FINALOBJW]]"
