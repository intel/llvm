///
/// tests specific to -fintelfpga -fsycl
///
// REQUIRES: clang-driver

/// Check SYCL headers path
// RUN:   %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-HEADERS-INTELFPGA %s
// CHK-HEADERS-INTELFPGA: clang{{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"

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
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-EARLY %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=early %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-EARLY %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=image %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-IMAGE %s
// CHK-FPGA-LINK-NOT: clang-offload-bundler{{.*}} "-check-section"
// CHK-FPGA-LINK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice" "-inputs=[[INPUT:.+\.o]]" "-outputs=[[OUTPUT1:.+\.o]]" "-unbundle"
// CHK-FPGA-LINK-NOT: clang-offload-bundler{{.*}}
// CHK-FPGA-LINK: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2:.+\.bc]]"
// CHK-FPGA-LINK: llvm-spirv{{.*}} "-o" "[[OUTPUT3:.+\.spv]]" "-spirv-max-version=1.1" "-spirv-ext=+all" "[[OUTPUT2]]"
// CHK-FPGA-EARLY: aoc{{.*}} "-o" "[[OUTPUT4:.+\.aocr]]" "[[OUTPUT3]]" "-sycl" "-rtl"
// CHK-FPGA-IMAGE: aoc{{.*}} "-o" "[[OUTPUT5:.+\.aocx]]" "[[OUTPUT3]]" "-sycl"
// CHK-FPGA-LINK: llvm-ar{{.*}} "cr" "libfoo.a" "[[INPUT]]"

// Output designation should not be used for unbundling step
// RUN:  touch %t.o
// RUN:  touch %t.obj
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-LINK-OUT %s
// RUN:  %clang_cl -### -fsycl -fintelfpga -fsycl-link %t.obj -Folibfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-OUT %s
// RUN:  %clang_cl -### -fsycl -fintelfpga -fsycl-link %t.obj -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-OUT %s
// CHK-FPGA-LINK-OUT-NOT: clang-offload-bundler{{.*}} "-outputs=libfoo.a" "-unbundle"

/// -fintelfpga -fsycl-link clang-cl specific
// RUN:  touch %t.obj
// RUN:  %clang_cl -### -fsycl -fintelfpga -fsycl-link %t.obj -Folibfoo.lib 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-WIN %s
// RUN:  %clang_cl -### -fsycl -fintelfpga -fsycl-link %t.obj -o libfoo.lib 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-WIN %s
// CHK-FPGA-LINK-WIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice{{.*}}" "-inputs=[[INPUT:.+\.obj]]" "-outputs=[[OUTPUT1:.+\.obj]]" "-unbundle"
// CHK-FPGA-LINK-WIN-NOT: clang-offload-bundler{{.*}}
// CHK-FPGA-LINK-WIN: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2:.+\.bc]]"
// CHK-FPGA-LINK-WIN: llvm-spirv{{.*}} "-o" "[[OUTPUT3:.+\.spv]]" "-spirv-max-version=1.1" "-spirv-ext=+all" "[[OUTPUT2]]"
// CHK-FPGA-LINK-WIN: aoc{{.*}} "-o" "[[OUTPUT5:.+\.aocr]]" "[[OUTPUT3]]" "-sycl" "-rtl"
// CHK-FPGA-LINK-WIN: lib.exe{{.*}} "[[INPUT]]" {{.*}} "-OUT:libfoo.lib"

/// Check -fintelfpga -fsycl-link with an FPGA archive
// Create the dummy archive
// RUN:  echo "Dummy AOCR image" > %t.aocr
// RUN:  echo "void foo() {}" > %t.c
// RUN:  %clang -c -o %t.o %t.c
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

/// Check the warning's emission for -fsycl-link's appending behavior
// RUN: touch dummy.a
// RUN: %clangxx -fsycl -fintelfpga -fsycl-link=image %s -o dummy.a -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHK-FPGA-LINK-WARN
// RUN: %clangxx -fsycl -fintelfpga -fsycl-link=early %s -o dummy.a -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHK-FPGA-LINK-WARN
// CHK-FPGA-LINK-WARN: warning: appending to an existing archive 'dummy.a'

/// -fintelfpga -fsycl-link name creation without output file specified
// RUN: mkdir -p %t_dir
// RUN: touch %t_dir/dummy_file.cpp
// RUN: %clang++ -### -fsycl -fintelfpga -fsycl-link %t_dir/dummy_file.cpp 2>&1 \
// RUN: | FileCheck -check-prefixes=CHK-SYCL-LINK-LIN -DINPUTSRC=dummy_file %s
// RUN: %clang_cl -### -fsycl -fintelfpga -fsycl-link %t_dir/dummy_file.cpp 2>&1 \
// RUN: | FileCheck -check-prefixes=CHK-SYCL-LINK-WIN -DINPUTSRC=dummy_file %s
// CHK-SYCL-LINK-LIN: llvm-ar{{.*}} "cr" "[[INPUTSRC]].a"
// CHK-SYCL-LINK-WIN: lib.exe{{.*}} "-OUT:[[INPUTSRC]].a"

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
// RUN:  %clang -c -o %t.o %t.c
// RUN:  clang-offload-wrapper -o %t-aocx.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=fpga_aocx-intel-unknown-sycldevice %t.aocx
// RUN:  llc -filetype=obj -o %t-aocx.o %t-aocx.bc
// RUN:  llvm-ar crv %t_aocx.a %t.o %t-aocx.o
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %t_aocx.a -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-PHASES %s
// RUN:  %clang_cl -fsycl -fintelfpga %t_aocx.a -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-PHASES %s
// CHK-FPGA-AOCX-PHASES: 0: input, "{{.*}}", fpga_aocx, (host-sycl)
// CHK-FPGA-AOCX-PHASES: 1: linker, {0}, image, (host-sycl)
// CHK-FPGA-AOCX-PHASES: 2: clang-offload-unbundler, {0}, fpga_aocx
// CHK-FPGA-AOCX-PHASES: 3: clang-offload-wrapper, {2}, object, (device-sycl)
// CHK-FPGA-AOCX-PHASES: 4: offload, "host-sycl ({{.*}}x86_64{{.*}})" {1}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice{{(-coff)?}})" {3}, image

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
// CHK-FPGA-LINK-SRC: 11: linker, {10}, ir, (device-sycl)
// CHK-FPGA-LINK-SRC: 12: llvm-spirv, {11}, spirv, (device-sycl)
// CHK-FPGA-LINK-SRC: 13: backend-compiler, {12}, fpga_aocr, (device-sycl)
// CHK-FPGA-LINK-SRC: 14: clang-offload-wrapper, {13}, object, (device-sycl)
// CHK-FPGA-LINK-SRC-DEFAULT: 15: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {14}, archive
// CHK-FPGA-LINK-SRC-CL: 15: offload, "host-sycl (x86_64-pc-windows-msvc)" {9}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice-coff)" {14}, archive

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
// RUN: %clang_cl -### -fsycl -fintelfpga %t-1.cpp %t-2.cpp -o %t.out 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES %s
// RUN: %clangxx -### -fsycl -fintelfpga %t-1.cpp %t-2.cpp 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES %s
// RUN: %clang_cl -### -fsycl -fintelfpga %t-1.cpp %t-2.cpp 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES %s
// CHK-FPGA-DEP-FILES: clang{{.*}} "-dependency-file" "[[INPUT1:.+\.d]]"
// CHK-FPGA-DEP-FILES: clang{{.*}} "-dependency-file" "[[INPUT2:.+\.d]]"
// CHK-FPGA-DEP-FILES: aoc{{.*}} "-dep-files={{.*}}[[INPUT1]],{{.*}}[[INPUT2]]"

/// -fintelfpga dependency file generation test to object
// RUN: %clangxx -### -fsycl -fintelfpga %t-1.cpp %t-2.cpp -c 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-DEP-FILES2,CHK-FPGA-DEP-FILES2-LIN %s
// RUN: %clang_cl -### -fsycl -fintelfpga %t-1.cpp %t-2.cpp -c 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-DEP-FILES2,CHK-FPGA-DEP-FILES2-WIN %s
// CHK-FPGA-DEP-FILES2: clang{{.*}} "-dependency-file" "[[INPUT1:.+\.d]]"
// CHK-FPGA-DEP-FILES2-LIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice,host-x86_64-unknown-linux-gnu,sycl-fpga_dep" {{.*}} "-inputs={{.*}}.bc,{{.*}}.o,[[INPUT1]]"
// CHK-FPGA-DEP-FILES2-WIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice-coff,host-x86_64-pc-windows-msvc,sycl-fpga_dep" {{.*}} "-inputs={{.*}}.bc,{{.*}}.obj,[[INPUT1]]"
// CHK-FPGA-DEP-FILES2: clang{{.*}} "-dependency-file" "[[INPUT2:.+\.d]]"
// CHK-FPGA-DEP-FILES2-LIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice,host-x86_64-unknown-linux-gnu,sycl-fpga_dep" {{.*}} "-inputs={{.*}}.bc,{{.*}}.o,[[INPUT2]]"
// CHK-FPGA-DEP-FILES2-WIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice-coff,host-x86_64-pc-windows-msvc,sycl-fpga_dep" {{.*}} "-inputs={{.*}}.bc,{{.*}}.obj,[[INPUT2]]"

/// -fintelfpga dependency file test to object with output designator
// RUN: touch %t-1.cpp
// RUN: %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %t-1.cpp -c -o dummy.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-DEP-FILES3,CHK-FPGA-DEP-FILES3-LIN %s
// RUN: %clang_cl -### -fsycl -fintelfpga %t-1.cpp -c -Fodummy.obj 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-DEP-FILES3,CHK-FPGA-DEP-FILES3-WIN %s
// CHK-FPGA-DEP-FILES3: clang{{.*}} "-dependency-file" "[[OUTPUT:.+\.d]]"
// CHK-FPGA-DEP-FILES3-LIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice,host-x86_64-unknown-linux-gnu,sycl-fpga_dep" {{.*}} "-inputs={{.*}}.bc,{{.*}}.o,[[OUTPUT]]"
// CHK-FPGA-DEP-FILES3-WIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice-coff,host-x86_64-pc-windows-msvc,sycl-fpga_dep" {{.*}} "-inputs={{.*}}.bc,{{.*}}.obj,[[OUTPUT]]"

/// -fintelfpga dependency obj use test
// RUN: touch %t-1.o
// RUN: touch %t-2.o
// RUN: %clangxx -### -fsycl -fintelfpga %t-1.o %t-2.o 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES-OBJ -DINPUT1=%t-1.o -DINPUT2=%t-2.o %s
// RUN: %clang_cl -### -fsycl -fintelfpga %t-1.o %t-2.o 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES-OBJ -DINPUT1=%t-1.o -DINPUT2=%t-2.o %s
// CHK-FPGA-DEP-FILES-OBJ: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-fpga_dep" "-inputs=[[INPUT1]]" "-outputs=[[DEPFILE1:.+\.d]]" "-unbundle"
// CHK-FPGA-DEP-FILES-OBJ: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-fpga_dep" "-inputs=[[INPUT2]]" "-outputs=[[DEPFILE2:.+\.d]]" "-unbundle"
// CHK-FPGA-DEP-FILES-OBJ: aoc{{.*}} "-dep-files=[[DEPFILE1]],[[DEPFILE2]]

/// -fintelfpga dependency file use from object phases test
// RUN: touch %t-1.o
// RUN: %clangxx -fsycl -fintelfpga -ccc-print-phases -### %t-1.o 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES-OBJ-PHASES -DINPUT=%t-1.o %s
// RUN: %clang_cl -fsycl -fintelfpga -ccc-print-phases -### %t-1.o 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES-OBJ-PHASES -DINPUT=%t-1.o %s
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 0: input, "[[INPUT]]", object, (host-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 1: clang-offload-unbundler, {0}, object, (host-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 2: linker, {1}, image, (host-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 3: linker, {1}, ir, (device-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 4: llvm-spirv, {3}, spirv, (device-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 5: clang-offload-unbundler, {0}, fpga_dependencies
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 6: backend-compiler, {4, 5}, fpga_aocx, (device-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 7: clang-offload-wrapper, {6}, object, (device-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 8: offload, "host-sycl (x86_64-{{unknown-linux-gnu|pc-windows-msvc}})" {2}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice{{(-coff)?}})" {7}, image

/// -fintelfpga output report file test
// RUN: mkdir -p %t_dir
// RUN: %clangxx -### -fsycl -fintelfpga %s -o %t_dir/file.out 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga %s -o %t_dir/file.out 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga %s -Fe%t_dir/file.out 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// CHK-FPGA-REPORT-OPT: aoc{{.*}} "-sycl" {{.*}} "-output-report-folder=[[OUTDIR]]{{/|\\\\}}file.prj"

/// -fintelfpga output report file from dir/source
/// check dependency file from dir/source
// RUN: mkdir -p %t_dir
// RUN: touch %t_dir/dummy.cpp
// RUN: %clangxx -### -fsycl -fintelfpga %t_dir/dummy.cpp  2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT2 %s
// RUN: %clang_cl -### -fsycl -fintelfpga %t_dir/dummy.cpp 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT2 %s
// CHK-FPGA-REPORT-OPT2: aoc{{.*}} "-sycl"{{.*}} "-dep-files=dummy.d" "-output-report-folder=dummy.prj"
// CHK-FPGA-REPORT-OPT2-NOT: aoc{{.*}} "-sycl" {{.*}}[[OUTDIR]]{{.*}}

/// -fintelfpga output report file should be based on first input (src/obj)
// RUN: mkdir -p %t_dir
// RUN: touch %t_dir/dummy1.cpp
// RUN: touch %t_dir/dummy2.cpp
// RUN: touch %t_dir/dummy1.o
// RUN: touch %t_dir/dummy2.o
// RUN: %clangxx -### -fsycl -fintelfpga %t_dir/dummy2.o %t_dir/dummy1.cpp  2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-REPORT-NAME %s
// RUN: %clangxx -### -fsycl -fintelfpga %t_dir/dummy2.cpp %t_dir/dummy1.o  2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-REPORT-NAME %s
// RUN: %clang_cl -### -fsycl -fintelfpga %t_dir/dummy2.o %t_dir/dummy1.cpp 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-REPORT-NAME %s
// RUN: %clang_cl -### -fsycl -fintelfpga %t_dir/dummy2.cpp %t_dir/dummy1.o 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-REPORT-NAME %s
// CHK-FPGA-REPORT-NAME: aoc{{.*}} "-sycl"{{.*}} "-output-report-folder=dummy2.prj"

/// -fintelfpga output dep file using -Fo<dir>
// RUN: mkdir -p %t_dir
// RUN: %clang_cl -### -c -fsycl -fintelfpga -Fo%t_dir/ %s 2>&1 \
// RUN:  | FileCheck -DDEPDIR=%t_dir/ -check-prefix=CHK-FPGA-DEP-DIR %s
// CHK-FPGA-DEP-DIR: clang{{.*}} "-dependency-file" "[[DEPDIR]][[DEPFILE:.+\.d]]"
// CHK-FPGA-DEP-DIR: clang-offload-bundler{{.*}} "-inputs={{.*}}.bc,{{.*}}.obj,[[DEPDIR]][[DEPFILE]]"

/// -fintelfpga static lib (aoco)
// RUN:  echo "Dummy AOCO image" > %t.aoco
// RUN:  echo "void foo() {}" > %t.c
// RUN:  echo "void foo2() {}" > %t2.c
// RUN:  %clang -c -o %t.o %t.c
// RUN:  %clang -fsycl -c -o %t2.o %t2.c
// RUN:  %clang_cl -fsycl -c -o %t2_cl.o %t2.c
// RUN:  clang-offload-wrapper -o %t-aoco.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=fpga_aoco-intel-unknown-sycldevice %t.aoco
// RUN:  llc -filetype=obj -o %t-aoco.o %t-aoco.bc
// RUN:  llvm-ar crv %t_aoco.a %t.o %t2.o %t-aoco.o
// RUN:  llvm-ar crv %t_aoco_cl.a %t.o %t2_cl.o %t-aoco.o
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -foffload-static-lib=%t_aoco.a %s -### -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-AOCO-PHASES %s
// CHK-FPGA-AOCO-PHASES: 0: input, "[[INPUTA:.+\.a]]", object, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 1: input, "[[INPUTCPP:.+\.cpp]]", c++, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 3: input, "[[INPUTCPP]]", c++, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 5: compiler, {4}, sycl-header, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// CHK-FPGA-AOCO-PHASES: 7: compiler, {6}, ir, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 8: backend, {7}, assembler, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 9: assembler, {8}, object, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 10: linker, {0, 9}, image, (host-sycl)
// CHK-FPGA-AOCO-PHASES: 11: compiler, {4}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 12: input, "[[INPUTA]]", archive
// CHK-FPGA-AOCO-PHASES: 13: partial-link, {9, 12}, object
// CHK-FPGA-AOCO-PHASES: 14: clang-offload-unbundler, {13}, object
// CHK-FPGA-AOCO-PHASES: 15: linker, {11, 14}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 16: llvm-spirv, {15}, spirv, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 17: input, "[[INPUTA]]", fpga_aoco
// CHK-FPGA-AOCO-PHASES: 18: clang-offload-unbundler, {17}, fpga_aoco
// CHK-FPGA-AOCO-PHASES: 19: backend-compiler, {16, 18}, fpga_aocx, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 20: clang-offload-wrapper, {19}, object, (device-sycl)
// CHK-FPGA-AOCO-PHASES: 21: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {20}, image

/// FPGA AOCO Windows phases check
// RUN:  %clang_cl -fsycl -fintelfpga -foffload-static-lib=%t_aoco_cl.a %s -### -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO-PHASES-WIN %s
// CHK-FPGA-AOCO-PHASES-WIN: 0: input, "{{.*}}", object, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 1: input, "[[INPUTSRC:.+\.cpp]]", c++, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 3: input, "[[INPUTSRC]]", c++, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 5: compiler, {4}, sycl-header, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 6: offload, "host-sycl (x86_64-pc-windows-msvc)" {2}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice-coff)" {5}, c++-cpp-output
// CHK-FPGA-AOCO-PHASES-WIN: 7: compiler, {6}, ir, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 8: backend, {7}, assembler, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 9: assembler, {8}, object, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 10: linker, {0, 9}, image, (host-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 11: compiler, {4}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 12: input, "[[INPUTA:.+\.a]]", archive
// CHK-FPGA-AOCO-PHASES-WIN: 13: clang-offload-unbundler, {12}, archive
// CHK-FPGA-AOCO-PHASES-WIN: 14: linker, {11, 13}, ir, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 15: llvm-spirv, {14}, spirv, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 16: input, "[[INPUTA]]", fpga_aoco
// CHK-FPGA-AOCO-PHASES-WIN: 17: clang-offload-unbundler, {16}, fpga_aoco
// CHK-FPGA-AOCO-PHASES-WIN: 18: backend-compiler, {15, 17}, fpga_aocx, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 19: clang-offload-wrapper, {18}, object, (device-sycl)
// CHK-FPGA-AOCO-PHASES-WIN: 20: offload, "host-sycl (x86_64-pc-windows-msvc)" {10}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice-coff)" {19}, image

/// aoco test, checking tools
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -foffload-static-lib=%t_aoco.a -### %s 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO,CHK-FPGA-AOCO-LIN %s
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %t_aoco.a -### %s 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO,CHK-FPGA-AOCO-LIN %s
// RUN:  %clang_cl -fsycl -fintelfpga -foffload-static-lib=%t_aoco_cl.a -### %s 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO,CHK-FPGA-AOCO-WIN %s
// RUN:  %clang_cl -fsycl -fintelfpga %t_aoco_cl.a -### %s 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCO,CHK-FPGA-AOCO-WIN %s
// CHK-FPGA-AOCO-LIN: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aoco-intel-unknown-sycldevice" "-inputs=[[INPUTLIB:.+\.a]]" "-check-section"
// CHK-FPGA-AOCO-LIN: clang{{.*}} "-emit-obj" {{.*}} "-o" "[[HOSTOBJ:.+\.o]]"
// CHK-FPGA-AOCO-LIN: ld{{.*}} "-r" "-o" "[[PARTLINKOBJ:.+\.o]]" "{{.*}}crt1.o" "{{.*}}crti.o" {{.*}} "[[HOSTOBJ]]" "[[INPUTLIB]]" "{{.*}}crtn.o"
// CHK-FPGA-AOCO-LIN: clang-offload-bundler{{.*}} "-type=oo" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice" "-inputs=[[PARTLINKOBJ]]" "-outputs={{.*}}" "-unbundle"
// CHK-FPGA-AOCO-WIN: clang-offload-bundler{{.*}} "-type=aoo" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice{{(-coff)?}}" "-inputs=[[INPUTLIB:.+\.a]]" "-outputs={{.*}}" "-unbundle"
// CHK-FPGA-AOCO: llvm-link{{.*}} "@{{.*}}" "-o" "[[LINKEDBC:.+\.bc]]"
// CHK-FPGA-AOCO: llvm-spirv{{.*}} "-o" "[[TARGSPV:.+\.spv]]" {{.*}} "[[LINKEDBC]]"
// CHK-FPGA-AOCO: clang-offload-bundler{{.*}} "-type=aoo" "-targets=sycl-fpga_aoco-intel-unknown-sycldevice" "-inputs=[[INPUTLIB]]" "-outputs=[[AOCOLIST:.+\.txt]]" "-unbundle"
// CHK-FPGA-AOCO: aoc{{.*}} "-o" "[[AOCXOUT:.+\.aocx]]" "[[TARGSPV]]" "-library-list=[[AOCOLIST]]" "-sycl"
// CHK-FPGA-AOCO: clang-offload-wrapper{{.*}} "-o=[[FINALBC:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "[[AOCXOUT]]"
// CHK-FPGA-AOCO-LIN: llc{{.*}} "-filetype=obj" "-o" "[[FINALOBJL:.+\.o]]" "[[FINALBC]]"
// CHK-FPGA-AOCO-WIN: llc{{.*}} "-filetype=obj" "-o" "[[FINALOBJW:.+\.obj]]" "[[FINALBC]]"
// CHK-FPGA-AOCO-LIN: ld{{.*}} "[[INPUTLIB]]" {{.*}} "[[FINALOBJL]]"
// CHK-FPGA-AOCO-WIN: link.exe{{.*}} "{{.*}}[[INPUTLIB]]" {{.*}} "[[FINALOBJW]]"

// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows-msvc
