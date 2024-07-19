// UNSUPPORTED: system-windows

/// -fintelfpga static lib (aoco)
// RUN:  echo "Dummy AOCO image" > %t.aoco
// RUN:  echo "void foo() {}" > %t.c
// RUN:  echo "void foo2() {}" > %t2.c
// RUN:  %clang -c -o %t.o %t.c
// RUN:  %clang -fno-sycl-use-footer -fintelfpga -c -o %t2.o %t2.c
// RUN:  %clang_cl -fno-sycl-use-footer -fintelfpga -c -o %t2_cl.o %t2.c
// RUN:  clang-offload-wrapper -o %t-aoco.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=fpga_aoco-intel-unknown %t.aoco
// RUN:  llc -filetype=obj -o %t-aoco.o %t-aoco.bc
// RUN:  clang-offload-wrapper -o %t-aoco_cl.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=fpga_aoco-intel-unknown %t.aoco
// RUN:  llc -filetype=obj -o %t-aoco_cl.o %t-aoco_cl.bc
// RUN:  llvm-ar crv %t_aoco.a %t.o %t2.o %t-aoco.o
// RUN:  llvm-ar crv %t_aoco_cl.a %t.o %t2_cl.o %t-aoco_cl.o
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga -Xshardware %t_aoco.a %s -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-AOCO-PHASES %s
// CHK-FPGA-AOCO-PHASES: 0: input, "[[INPUTA:.+\.a]]", object, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 1: input, "[[INPUTCPP:.+\.cpp]]", c++, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 3: input, "[[INPUTCPP]]", c++, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 5: compiler, {4}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64_fpga-unknown-unknown)" {5}, c++-cpp-output
// CHK-FPGA-AOCO-PHASES: 7: compiler, {6}, ir, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 8: backend, {7}, assembler, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 9: assembler, {8}, object, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 10: linker, {0, 9}, host_dep_image, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 11: clang-offload-deps, {10}, ir, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 12: input, "[[INPUTA]]", archive
// CHK-FPGA-AOCO-PHASES: 13: clang-offload-unbundler, {12}, tempfilelist
// CHK-FPGA-AOCO-PHASES: 14: spirv-to-ir-wrapper, {13}, tempfilelist, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 15: linker, {5, 11, 14}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 16: sycl-post-link, {15}, tempfiletable, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 17: file-table-tform, {16}, tempfilelist, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 18: llvm-spirv, {17}, tempfilelist, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 19: input, "[[INPUTA]]", archive
// CHK-FPGA-AOCO-PHASES: 20: clang-offload-unbundler, {19}, fpga_dep_list
// CHK-FPGA-AOCO-PHASES: 21: input, "[[INPUTA]]", fpga_aoco
// CHK-FPGA-AOCO-PHASES: 22: clang-offload-unbundler, {21}, fpga_aoco
// CHK-FPGA-AOCO-PHASES: 23: backend-compiler, {18, 20, 22}, fpga_aocx, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 24: file-table-tform, {16, 23}, tempfiletable, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 25: clang-offload-wrapper, {24}, object, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 26: offload, "device-sycl (spir64_fpga-unknown-unknown)" {25}, object
// CHK-FPGA-AOCO-PHASES: 27: linker, {0, 9, 26}, image, (host-sycl)

/// aoco test, checking tools
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga -Xshardware %t_aoco.a -### %s 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO,CHK-FPGA-AOCO-LIN %s
// RUN:  %clang_cl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga -Xshardware %t_aoco_cl.a -### %s 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO,CHK-FPGA-AOCO-WIN %s
// CHK-FPGA-AOCO: clang-offload-bundler{{.*}} "-type=aoo" "-excluded-targets=sycl-fpga_aoco-intel-unknown" "-targets=sycl-spir64_fpga-unknown-unknown" "-input=[[INPUTLIB:.+\.a]]" "-output=[[LIBLIST:.+\.txt]]" "-unbundle"
// CHK-FPGA-AOCO: spirv-to-ir-wrapper{{.*}} "[[LIBLIST]]" "-o" "[[LIBLIST2:.+\.txt]]"
// CHK-FPGA-AOCO: llvm-link{{.*}} "-o" "[[LINKEDBC:.+\.bc]]"
// CHK-FPGA-AOCO: llvm-link{{.*}} "--only-needed" "[[LINKEDBC]]" "@[[LIBLIST2]]" "-o" "[[LINKEDBC2:.+\.bc]]"
// CHK-FPGA-AOCO: sycl-post-link{{.*}} "-device-globals"{{.*}} "-spec-const=emulation"{{.*}} "-o" "[[SPLTABLE:.+\.table]]" "[[LINKEDBC2]]"
// CHK-FPGA-AOCO: file-table-tform{{.*}} "-o" "[[TABLEOUT:.+\.txt]]" "[[SPLTABLE]]"
// CHK-FPGA-AOCO: llvm-spirv{{.*}} "-o" "[[TARGSPV:.+\.txt]]" {{.*}} "[[TABLEOUT]]"
// CHK-FPGA-AOCO: clang-offload-bundler{{.*}} "-type=aoo" "-targets=sycl-fpga_aoco-intel-unknown" "-input=[[INPUTLIB]]" "-output=[[AOCOLIST:.+\.txt]]" "-unbundle"
// CHK-FPGA-AOCO: aoc{{.*}} "-o" "[[AOCXOUT:.+\.aocx]]" "[[TARGSPV]]" "-library-list=[[AOCOLIST]]" "-sycl"
// CHK-FPGA-AOCO: file-table-tform{{.*}} "-o" "[[TABLEOUT2:.+\.table]]" "[[SPLTABLE]]" "[[AOCXOUT]]"
// CHK-FPGA-AOCO: clang-offload-wrapper{{.*}} "-o=[[FINALBC:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "-batch" "[[TABLEOUT2]]"
// CHK-FPGA-AOCO-LIN: llc{{.*}} "-filetype=obj" "-o" "[[FINALOBJL:.+\.o]]" "[[FINALBC]]"
// CHK-FPGA-AOCO-WIN: llc{{.*}} "-filetype=obj" "-o" "[[FINALOBJW:.+\.obj]]" "[[FINALBC]]"
// CHK-FPGA-AOCO-LIN: ld{{.*}} "[[INPUTLIB]]" {{.*}} "[[FINALOBJL]]"
// CHK-FPGA-AOCO-WIN: link.exe{{.*}} "{{.*}}[[INPUTLIB]]" {{.*}} "[[FINALOBJW]]"

/// aoco archive check with emulation
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga %t_aoco.a %s -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-AOCO-PHASES-EMU %s
// CHK-FPGA-AOCO-PHASES-EMU: 0: input, "[[INPUTA:.+\.a]]", object, (host-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 1: input, "[[INPUTCPP:.+\.cpp]]", c++, (host-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 3: input, "[[INPUTCPP]]", c++, (device-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 5: compiler, {4}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64_fpga-unknown-unknown)" {5}, c++-cpp-output
// CHK-FPGA-AOCO-PHASES-EMU: 7: compiler, {6}, ir, (host-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 8: backend, {7}, assembler, (host-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 9: assembler, {8}, object, (host-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 10: linker, {0, 9}, host_dep_image, (host-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 11: clang-offload-deps, {10}, ir, (host-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 12: input, "[[INPUTA]]", archive
// CHK-FPGA-AOCO-PHASES-EMU: 13: clang-offload-unbundler, {12}, tempfilelist
// CHK-FPGA-AOCO-PHASES-EMU: 14: spirv-to-ir-wrapper, {13}, tempfilelist, (device-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 15: linker, {5, 11, 14}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 16: sycl-post-link, {15}, tempfiletable, (device-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 17: file-table-tform, {16}, tempfilelist, (device-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 18: llvm-spirv, {17}, tempfilelist, (device-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 19: input, "[[INPUTA]]", archive
// CHK-FPGA-AOCO-PHASES-EMU: 20: clang-offload-unbundler, {19}, fpga_dep_list
// CHK-FPGA-AOCO-PHASES-EMU: 21: backend-compiler, {18, 20}, fpga_aocx, (device-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 22: file-table-tform, {16, 21}, tempfiletable, (device-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 23: clang-offload-wrapper, {22}, object, (device-sycl)
// CHK-FPGA-AOCO-PHASES-EMU: 24: offload, "device-sycl (spir64_fpga-unknown-unknown)" {23}, object
// CHK-FPGA-AOCO-PHASES-EMU: 25: linker, {0, 9, 24}, image, (host-sycl)

/// aoco emulation test, checking tools
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga %t_aoco.a -### %s 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO-EMU,CHK-FPGA-AOCO-EMU-LIN %s
// RUN:  %clang_cl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga %t_aoco_cl.a -### %s 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO-EMU,CHK-FPGA-AOCO-EMU-WIN %s
// CHK-FPGA-AOCO-EMU: clang-offload-bundler{{.*}} "-type=aoo" "-targets=sycl-spir64_fpga-unknown-unknown" "-input=[[INPUTLIB:.+\.a]]" "-output=[[OUTLIB:.+\.txt]]" "-unbundle"
// CHK-FPGA-AOCO-EMU: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTLIB]]" "--in-replace=[[OUTLIB]]" "--out-file-list=[[DEVICELIST:.+\.txt]]" "--out-replace=[[DEVICELIST]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTLIB]]" "-o" "[[DEVICELIST]]"
// CHK-FPGA-AOCO-EMU: llvm-link{{.*}} "@[[DEVICELIST]]" "-o" "[[LINKEDBC:.+\.bc]]"
// CHK-FPGA-AOCO-EMU: sycl-post-link{{.*}} "-O2" "-device-globals"{{.*}} "-spec-const=emulation"{{.*}} "-o" "[[SPLTABLE:.+\.table]]" "[[LINKEDBC]]"
// CHK-FPGA-AOCO-EMU: file-table-tform{{.*}} "-o" "[[TABLEOUT:.+\.txt]]" "[[SPLTABLE]]"
// CHK-FPGA-AOCO-EMU: llvm-spirv{{.*}} "-o" "[[TARGSPV:.+\.txt]]" {{.*}} "[[TABLEOUT]]"
// CHK-FPGA-AOCO-EMU: opencl-aot{{.*}} "-device=fpga_fast_emu" "-spv=[[TARGSPV]]" "-ir=[[AOCXOUT:.+\.aocx]]"
// CHK-FPGA-AOCO-EMU: file-table-tform{{.*}} "-o" "[[TABLEOUT2:.+\.table]]" "[[SPLTABLE]]" "[[AOCXOUT]]"
// CHK-FPGA-AOCO-EMU: clang-offload-wrapper{{.*}} "-o=[[FINALBC:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "-batch" "[[TABLEOUT2]]"
// CHK-FPGA-AOCO-EMU-LIN: llc{{.*}} "-filetype=obj" "-o" "[[FINALOBJL:.+\.o]]" "[[FINALBC]]"
// CHK-FPGA-AOCO-EMU-WIN: llc{{.*}} "-filetype=obj" "-o" "[[FINALOBJW:.+\.obj]]" "[[FINALBC]]"
// CHK-FPGA-AOCO-EMU-LIN: ld{{.*}} "[[INPUTLIB]]" {{.*}} "[[FINALOBJL]]"
// CHK-FPGA-AOCO-EMU-WIN: link.exe{{.*}} "{{.*}}[[INPUTLIB]]" {{.*}} "[[FINALOBJW]]"
