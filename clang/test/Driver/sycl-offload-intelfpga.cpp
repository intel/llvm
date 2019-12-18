///
/// tests specific to -fintelfpga -fsycl
///
// REQUIRES: clang-driver

/// -fintelfpga implies -g and -MMD
// RUN:   %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-INTELFPGA %s
// CHK-TOOLS-INTELFPGA: clang{{.*}} "-debug-info-kind=limited" {{.*}} "-dependency-file"
// CHK-TOOLS-INTELFPGA: aoc{{.*}} "-dep-files={{.*}}"

/// -fintelfpga implies -g but -g0 should override
// RUN:   %clangxx -### -target x86_64-unknown-linux-gnu -g0 -fsycl -fintelfpga %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-INTELFPGA-G0 %s
// CHK-TOOLS-INTELFPGA-G0-NOT: clang{{.*}} "-debug-info-kind=limited"

/// -fintelfpga -fsycl-link tests
// RUN:  touch %t.o
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link %t.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-EARLY %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=early %t.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-EARLY %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=image %t.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-IMAGE %s
// CHK-FPGA-LINK-NOT: clang-offload-bundler{{.*}} "-check-section"
// CHK-FPGA-LINK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice" "-inputs=[[INPUT:.+\.o]]" "-outputs=[[OUTPUT1:.+\.o]]" "-unbundle"
// CHK-FPGA-LINK: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2:.+\.bc]]"
// CHK-FPGA-LINK: llvm-spirv{{.*}} "-o" "[[OUTPUT3:.+\.spv]]" "-spirv-max-version=1.1" "-spirv-ext=+all" "[[OUTPUT2]]"
// CHK-FPGA-EARLY: aoc{{.*}} "-o" "[[OUTPUT4:.+\.aocr]]" "[[OUTPUT3]]" "-sycl" "-rtl"
// CHK-FPGA-IMAGE: aoc{{.*}} "-o" "[[OUTPUT5:.+\.aocx]]" "[[OUTPUT3]]" "-sycl"
// CHK-FPGA-LINK: {{lib|llvm-ar}}{{.*}} "[[INPUT]]"

/// -fintelfpga -fsycl-link clang-cl specific
// RUN:  touch %t.obj
// RUN:  %clang_cl -### -fsycl -fintelfpga -fsycl-link %t.obj 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-WIN %s
// CHK-FPGA-LINK-WIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice{{.*}}" "-inputs=[[INPUT:.+\.obj]]" "-outputs=[[OUTPUT1:.+\.obj]]" "-unbundle"
// CHK-FPGA-LINK-WIN: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2:.+\.bc]]"
// CHK-FPGA-LINK-WIN: llvm-spirv{{.*}} "-o" "[[OUTPUT3:.+\.spv]]" "-spirv-max-version=1.1" "-spirv-ext=+all" "[[OUTPUT2]]"
// CHK-FPGA-LINK-WIN: aoc{{.*}} "-o" "[[OUTPUT5:.+\.aocr]]" "[[OUTPUT3]]" "-sycl" "-rtl"
// CHK-FPGA-LINK-WIN: lib.exe{{.*}} "[[INPUT]]"


/// Check -fintelfpga -fsycl-link with an FPGA archive
// Create the dummy archive
// RUN:  echo "Dummy AOCR image" > %t.aocr
// RUN:  echo "void foo() {}" > %t.c
// RUN:  %clang -c %t.c
// RUN:  clang-offload-wrapper -o %t-aocr.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=fpga_aocr-intel-unknown-sycldevice %t.aocr
// RUN:  llc -filetype=obj -o %t-aocr.o %t-aocr.bc
// RUN:  llvm-ar crv %t.a %t.o %t-aocr.o
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=image %t.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-LIB,CHK-FPGA-LINK-LIB-IMAGE %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=early %t.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-LIB,CHK-FPGA-LINK-LIB-EARLY %s
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocx-intel-unknown-sycldevice" "-inputs=[[INPUT:.+\.a]]" "-check-section"
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocr-intel-unknown-sycldevice" "-inputs=[[INPUT]]" "-check-section"
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocr-intel-unknown-sycldevice" "-inputs=[[INPUT]]" "-outputs=[[OUTPUT2:.+\.aocr]]" "-unbundle"
// CHK-FPGA-LINK-LIB-IMAGE: aoc{{.*}} "-o" "[[OUTPUT3:.+\.aocx]]" "[[OUTPUT2]]" "-sycl"
// CHK-FPGA-LINK-LIB-EARLY: aoc{{.*}} "-o" "[[OUTPUT4:.+\.aocr]]" "[[OUTPUT2]]" "-sycl" "-rtl"
// CHK-FPGA-LINK-LIB-IMAGE: clang-offload-wrapper{{.*}} "-host=x86_64-unknown-linux-gnu" "--emit-reg-funcs=0" "-target=fpga_aocx-intel-unknown-sycldevice" "-kind=sycl" "[[OUTPUT3]]"
// CHK-FPGA-LINK-LIB-EARLY: clang-offload-wrapper{{.*}} "-host=x86_64-unknown-linux-gnu" "-target=fpga_aocr-intel-unknown-sycldevice" "-kind=sycl" "[[OUTPUT4]]"
// CHK-FPGA-LINK-LIB: llc{{.*}} "-filetype=obj" "-o" "[[OUTPUT5:.+\.o]]"
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=aoo" "-targets=host-x86_64-unknown-linux-gnu" "-inputs=[[INPUT]]" "-outputs=[[OUTPUT1:.+\.txt]]" "-unbundle"
// CHK-FPGA-LINK-LIB: llvm-ar{{.*}} "cr" {{.*}} "@[[OUTPUT1]]"

/// -fintelfpga with AOCR library and additional object
// RUN:  touch %t2.o
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %t.a %t2.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA %s
// CHK-FPGA: aoc{{.*}} "-o" {{.*}} "-sycl"
// CHK-FPGA: llc{{.*}} "-filetype=obj" "-o" "[[FINALLINK:.*\.o]]"
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=o" "-targets=host-x86_64-unknown-linux-gnu,sycl-spir64_fpga-unknown-unknown-sycldevice" {{.*}} "-outputs=[[FINALLINK2:.+\.o]],[[OUTPUT1:.+\.o]]" "-unbundle"
// CHK-FPGA: llvm-no-spir-kernel{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT3:.+\.o]]"
// CHK-FPGA: llvm-link{{.*}} "[[OUTPUT3]]" "-o" "[[OUTPUT4:.+\.bc]]"
// CHK-FPGA: llvm-spirv{{.*}} "-o" "[[OUTPUT5:.+\.spv]]" "-spirv-max-version=1.1" "-spirv-ext=+all" "[[OUTPUT4]]"
// CHK-FPGA: clang-offload-wrapper{{.*}} "-o=[[OUTPUT6:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_fpga" "-kind=sycl" "[[OUTPUT5]]"
// CHK-FPGA: llc{{.*}} "-filetype=obj" "-o" "[[FINALLINK3:.+\.o]]" "[[OUTPUT6]]"
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=aoo" "-targets=host-x86_64-unknown-linux-gnu" {{.*}} "-outputs=[[FINALLINK4:.+\.txt]]" "-unbundle"
// CHK-FPGA: {{link|ld}}{{.*}} "@[[FINALLINK4]]" "[[FINALLINK2]]" "[[FINALLINK]]" "[[FINALLINK3]]"

/// -fintelfpga with AOCX library
// Create the dummy archive
// RUN:  echo "Dummy AOCX image" > %t.aocx
// RUN:  echo "void foo() {}" > %t.c
// RUN:  %clang -c %t.c
// RUN:  clang-offload-wrapper -o %t-aocx.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=fpga_aocx-intel-unknown-sycldevice %t.aocx
// RUN:  llc -filetype=obj -o %t-aocx.o %t-aocx.bc
// RUN:  llvm-ar crv %t_aocx.a %t.o %t-aocx.o
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %t_aocx.a -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-PHASES %s
// RUN:  %clang_cl -fsycl -fintelfpga %t_aocx.a -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-PHASES %s
// CHK-FPGA-AOCX-PHASES: 0: input, "{{.*}}", object, (host-sycl)
// CHK-FPGA-AOCX-PHASES: 1: linker, {0}, image, (host-sycl)

// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX,CHK-FPGA-AOCX-LIN %s
// RUN:  %clang_cl -fsycl -fintelfpga %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX,CHK-FPGA-AOCX-WIN %s
// CHK-FPGA-AOCX: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocx-intel-unknown-sycldevice" "-inputs=[[LIBINPUT:.+\.a]]" "-outputs=[[BUNDLEOUT:.+\.aocx]]" "-unbundle"
// CHK-FPGA-AOCX: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "[[BUNDLEOUT]]"
// CHK-FPGA-AOCX-LIN: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUT:.+\.o]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-WIN: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUT2:.+\.obj]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-LIN: ld{{.*}} "[[LIBINPUT]]" "[[LLCOUT]]"
// CHK-FPGA-AOCX-WIN: link{{.*}} "[[LIBINPUT]]" "[[LLCOUT2]]"

/// -fintelfpga -fsycl-link from source
// RUN: touch %t.cpp
// RUN: %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=early %t.cpp -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-SRC,CHK-FPGA-LINK-SRC-DEFAULT %s
// RUN: %clang_cl -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=early %t.cpp -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-SRC,CHK-FPGA-LINK-SRC-CL %s
// CHK-FPGA-LINK-SRC: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHK-FPGA-LINK-SRC: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-FPGA-LINK-SRC: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-FPGA-LINK-SRC: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-FPGA-LINK-SRC: 4: compiler, {3}, sycl-header, (device-sycl)
// CHK-FPGA-LINK-SRC-DEFAULT: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {4}, c++-cpp-output
// CHK-FPGA-LINK-SRC-CL: 5: offload, "host-sycl (x86_64-pc-windows-msvc)" {1}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice-coff)" {4}, c++-cpp-output
// CHK-FPGA-LINK-SRC: 6: compiler, {5}, ir, (host-sycl)
// CHK-FPGA-LINK-SRC: 7: backend, {6}, assembler, (host-sycl)
// CHK-FPGA-LINK-SRC: 8: assembler, {7}, object, (host-sycl)
// CHK-FPGA-LINK-SRC: 9: linker, {8}, archive, (host-sycl)
// CHK-FPGA-LINK-SRC: 10: compiler, {3}, ir, (device-sycl)
// CHK-FPGA-LINK-SRC: 11: backend, {10}, assembler, (device-sycl)
// CHK-FPGA-LINK-SRC: 12: assembler, {11}, object, (device-sycl)
// CHK-FPGA-LINK-SRC: 13: linker, {12}, ir, (device-sycl)
// CHK-FPGA-LINK-SRC: 14: llvm-spirv, {13}, spirv, (device-sycl)
// CHK-FPGA-LINK-SRC: 15: backend-compiler, {14}, fpga_aocr, (device-sycl)
// CHK-FPGA-LINK-SRC: 16: clang-offload-wrapper, {15}, object, (device-sycl)
// CHK-FPGA-LINK-SRC-DEFAULT: 17: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {16}, archive
// CHK-FPGA-LINK-SRC-CL: 17: offload, "host-sycl (x86_64-pc-windows-msvc)" {9}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice-coff)" {16}, archive

/// -fintelfpga with -reuse-exe=
// RUN:  touch %t.cpp
// RUN:  %clangxx -### -reuse-exe=testing -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %t.cpp 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-REUSE-EXE %s
// CHK-FPGA-REUSE-EXE: aoc{{.*}} "-o" {{.*}} "-sycl" {{.*}} "-reuse-exe=testing"

/// -fintelfpga dependency file generation test
// RUN: touch %t-1.cpp
// RUN: touch %t-2.cpp
// RUN: %clangxx -### -fsycl -fintelfpga %t-1.cpp %t-2.cpp -o %t.out 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES %s
// RUN: %clangxx -### -fsycl -fintelfpga %t-1.cpp %t-2.cpp 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES %s
// CHK-FPGA-DEP-FILES: clang{{.*}} "-dependency-file" "[[INPUT1:.+\.d]]"
// CHK-FPGA-DEP-FILES: clang{{.*}} "-dependency-file" "[[INPUT2:.+\.d]]"
// CHK-FPGA-DEP-FILES: aoc{{.*}} "-dep-files={{.*}}[[INPUT1]],{{.*}}[[INPUT2]]"

/// -fintelfpga output report file test
// RUN: mkdir -p %t_dir
// RUN: %clangxx -### -fsycl -fintelfpga %s -o %t_dir/file.out 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga %s -o %t_dir/file.out 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga %s -Fe%t_dir/file.out 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// CHK-FPGA-REPORT-OPT: aoc{{.*}} "-sycl" {{.*}} "-output-report-folder=[[OUTDIR]]{{/|\\\\}}file.prj"

/// -fintelfpga static lib (aoco)
// RUN:  echo "Dummy AOCO image" > %t.aoco
// RUN:  echo "void foo() {}" > %t.c
// RUN:  %clang -c %t.c
// RUN:  clang-offload-wrapper -o %t-aoco.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=fpga_aoco-intel-unknown-sycldevice %t.aoco
// RUN:  llc -filetype=obj -o %t-aoco.o %t-aoco.bc
// RUN:  llvm-ar crv %t_aoco.a %t.o %t-aoco.o
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -foffload-static-lib=%t_aocx.a %s -### -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO-PHASES,CHK-FPGA-AOCO-PHASES-LIN %s
// RUN:  %clang_cl -fsycl -fintelfpga -foffload-static-lib=%t_aoco.a %s -### -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO-PHASES,CHK-FPGA-AOCO-PHASES-WIN %s
// CHK-FPGA-AOCO-PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 4: compiler, {3}, sycl-header, (device-sycl)
// CHK-FPGA-AOCO-PHASES-LIN: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {4}, c++-cpp-output
// CHK-FPGA-AOCO-PHASES-WIN: 5: offload, "host-sycl (x86_64-pc-windows-msvc)" {1}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice-coff)" {4}, c++-cpp-output
// CHK-FPGA-AOCO-PHASES: 6: compiler, {5}, ir, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 7: backend, {6}, assembler, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 8: assembler, {7}, object, (host-sycl)
// CHK-FPGA-AOCO-PHASES-LIN: 9: clang-offload-unbundler, {8}, object, (host-sycl)
// CHK-FPGA-AOCO-PHASES-LIN: 10: linker, {9}, image, (host-sycl)
// CHK-FPGA-AOCO-PHASES-LIN: 11: compiler, {3}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES-LIN: 12: backend, {11}, assembler, (device-sycl)
// CHK-FPGA-AOCO-PHASES-LIN: 13: assembler, {12}, object, (device-sycl)
// CHK-FPGA-AOCO-PHASES-LIN: 14: linker, {13, 9}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES-LIN: 15: llvm-spirv, {14}, spirv, (device-sycl)
// CHK-FPGA-AOCO-PHASES-LIN: 16: backend-compiler, {15}, fpga_aocx, (device-sycl)
// CHK-FPGA-AOCO-PHASES-LIN: 17: clang-offload-wrapper, {16}, object, (device-sycl)
// CHK-FPGA-AOCO-PHASES-LIN: 18: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {17}, image

// CHK-FPGA-AOCO-PHASES-WIN: 9: linker, {8}, image, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 10: compiler, {3}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 11: backend, {10}, assembler, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 12: assembler, {11}, object, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 13: input, "{{.*}}", archive
// CHK-FPGA-AOCO-PHASES-WIN: 14: clang-offload-unbundler, {13}, archive
// CHK-FPGA-AOCO-PHASES-WIN: 15: linker, {12, 14}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 16: llvm-spirv, {15}, spirv, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 17: backend-compiler, {16}, fpga_aocx, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 18: clang-offload-wrapper, {17}, object, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 19: offload, "host-sycl (x86_64-pc-windows-msvc)" {9}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice-coff)" {18}, image

/// aoco test, checking tools
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -foffload-static-lib=%t_aoco.a -### %s 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO,CHK-FPGA-AOCO-LIN %s
// RUN:  %clang_cl -fsycl -fintelfpga -foffload-static-lib=%t_aoco.a -### %s 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO,CHK-FPGA-AOCO-WIN %s
// CHK-FPGA-AOCO-LIN: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aoco-intel-unknown-sycldevice" "-inputs=[[INPUTLIB:.+\.a]]" "-check-section"
// CHK-FPGA-AOCO-LIN: ld{{.*}} "-r" "-o" "[[PARTLINKOBJ:.+\.o]]" {{.*}} "[[INPUTLIB]]"
// CHK-FPGA-AOCO-LIN: clang-offload-bundler{{.*}} "-type=oo" "-targets=host-x86_64-unknown-linux-gnu,sycl-spir64_fpga-unknown-unknown-sycldevice" "-inputs=[[PARTLINKOBJ]]" "-outputs=[[HOSTOBJLIB:.+\.o]],{{.*}}" "-unbundle"
// CHK-FPGA-AOCO-WIN: clang-offload-bundler{{.*}} "-type=aoo" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice-coff" "-inputs=[[INPUTLIB:.+\.a]]" "-outputs={{.*}}" "-unbundle"                               
// CHK-FPGA-AOCO: llvm-link{{.*}} "@{{.*}}" "-o" "[[LINKEDBC:.+\.bc]]"
// CHK-FPGA-AOCO: llvm-spirv{{.*}} "-o" "[[TARGSPV:.+\.spv]]" {{.*}} "[[LINKEDBC]]"
// CHK-FPGA-AOCO: clang-offload-bundler{{.*}} "-type=aoo" "-targets=sycl-fpga_aoco-intel-unknown-sycldevice" "-inputs=[[INPUTLIB]]" "-outputs=[[AOCOLIST:.+\.txt]]" "-unbundle"
// CHK-FPGA-AOCO: aoc{{.*}} "-o" "[[AOCXOUT:.+\.aocx]]" "[[TARGSPV]]" "-library-list=[[AOCOLIST]]" "-sycl"
// CHK-FPGA-AOCO: clang-offload-wrapper{{.*}} "-o=[[FINALBC:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "[[AOCXOUT]]"
// CHK-FPGA-AOCO-LIN: llc{{.*}} "-filetype=obj" "-o" "[[FINALOBJL:.+\.o]]" "[[FINALBC]]"
// CHK-FPGA-AOCO-WIN: llc{{.*}} "-filetype=obj" "-o" "[[FINALOBJW:.+\.obj]]" "[[FINALBC]]"
// CHK-FPGA-AOCO-LIN: ld{{.*}} "[[HOSTOBJLIB]]" "[[FINALOBJL]]"
// CHK-FPGA-AOCO-WIN: link.exe{{.*}} "-defaultlib:[[INPUTLIB]]" {{.*}} "[[FINALOBJW]]"

// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows-msvc
