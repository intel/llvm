/// -fintelfpga static lib (aoco)
// RUN:  echo "Dummy AOCO image" > %t.aoco
// RUN:  echo "void foo() {}" > %t.c
// RUN:  echo "void foo2() {}" > %t2.c
// RUN:  %clang -target x86_64-pc-windows-msvc -c -o %t.o %t.c
// RUN:  %clang_cl --target=x86_64-pc-windows-msvc -fsycl -fintelfpga -c -o %t2.o %t2.c
// RUN:  clang-offload-wrapper -o %t-aoco.bc -host=x86_64-pc-windows-msvc -kind=sycl -target=fpga_aoco-intel-unknown %t.aoco
// RUN:  llc -filetype=obj -o %t-aoco.o %t-aoco.bc
// RUN:  llvm-ar crv %t_aoco.a %t.o %t2.o %t-aoco.o
// RUN:  %clang_cl --target=x86_64-pc-windows-msvc -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga -Xshardware %t_aoco.a %s -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO-PHASES-WIN %s
// RUN:  %clangxx -target x86_64-pc-windows-msvc -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga -Xshardware %t_aoco.a %s -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO-PHASES-WIN %s
// CHK-FPGA-AOCO-PHASES-WIN: 0: input, "[[INPUTA:.+\.a]]", object, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 1: input, "[[INPUTSRC:.+\.cpp]]", c++, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 2: append-footer, {1}, c++, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 3: preprocessor, {2}, c++-cpp-output, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 4: input, "[[INPUTSRC]]", c++, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 5: preprocessor, {4}, c++-cpp-output, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 6: compiler, {5}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 7: offload, "host-sycl (x86_64-pc-windows-msvc)" {3}, "device-sycl (spir64_fpga-unknown-unknown)" {6}, c++-cpp-output
// CHK-FPGA-AOCO-PHASES-WIN: 8: compiler, {7}, ir, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 9: backend, {8}, assembler, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 10: assembler, {9}, object, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 11: linker, {0, 10}, image, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 12: linker, {0, 10}, host_dep_image, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 13: clang-offload-deps, {12}, ir, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 14: linker, {6, 13}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 15: sycl-post-link, {14}, tempfiletable, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 16: file-table-tform, {15}, tempfilelist, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 17: llvm-spirv, {16}, tempfilelist, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 18: input, "[[INPUTA]]", archive
// CHK-FPGA-AOCO-PHASES-WIN: 19: clang-offload-unbundler, {18}, fpga_dep_list
// CHK-FPGA-AOCO-PHASES-WIN: 20: input, "[[INPUTA]]", fpga_aoco
// CHK-FPGA-AOCO-PHASES-WIN: 21: clang-offload-unbundler, {20}, fpga_aoco
// CHK-FPGA-AOCO-PHASES-WIN: 22: backend-compiler, {17, 19, 21}, fpga_aocx, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 23: file-table-tform, {15, 22}, tempfiletable, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 24: clang-offload-wrapper, {23}, object, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 25: offload, "host-sycl (x86_64-pc-windows-msvc)" {11}, "device-sycl (spir64_fpga-unknown-unknown)" {24}, image

/// aoco test, checking tools
// RUN:  %clang_cl --target=x86_64-pc-windows-msvc -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga %t_aoco.a -Xshardware -### %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-AOCO %s
// RUN:  %clang_cl --target=x86_64-pc-windows-msvc -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga %t_aoco.a -Xshardware -### %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-AOCO %s
// CHK-FPGA-AOCO: llvm-link{{.*}} "-o" "[[LINKEDBC:.+\.bc]]"
// CHK-FPGA-AOCO: sycl-post-link{{.*}} "-split-esimd"{{.*}} "-O2" "-spec-const=default" {{.*}} "-o" "[[SPLTABLE:.+\.table]]" "[[LINKEDBC]]"
// CHK-FPGA-AOCO: file-table-tform{{.*}} "-o" "[[TABLEOUT:.+\.txt]]" "[[SPLTABLE]]"
// CHK-FPGA-AOCO: llvm-spirv{{.*}} "-o" "[[TARGSPV:.+\.txt]]" {{.*}} "[[TABLEOUT]]"
// CHK-FPGA-AOCO: clang-offload-bundler{{.*}} "-type=aoo" "-targets=sycl-fpga_aoco-intel-unknown" "-input=[[INPUTLIB:.+\.a]]" "-output=[[AOCOLIST:.+\.txt]]" "-unbundle"
// CHK-FPGA-AOCO: aoc{{.*}} "-o" "[[AOCXOUT:.+\.aocx]]" "[[TARGSPV]]" "-library-list=[[AOCOLIST]]" "-sycl"
// CHK-FPGA-AOCO: file-table-tform{{.*}} "-o" "[[TABLEOUT2:.+\.table]]" "[[SPLTABLE]]" "[[AOCXOUT]]"
// CHK-FPGA-AOCO: clang-offload-wrapper{{.*}} "-o=[[FINALBC:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "-batch" "[[TABLEOUT2]]"
// CHK-FPGA-AOCO: llc{{.*}} "-filetype=obj" "-o" "[[FINALOBJW:.+\.obj]]" "[[FINALBC]]"
// CHK-FPGA-AOCO: link.exe{{.*}} "{{.*}}[[INPUTLIB]]" {{.*}} "[[FINALOBJW]]"
