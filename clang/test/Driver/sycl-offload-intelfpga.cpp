///
/// tests specific to -fintelfpga -fsycl
///
// REQUIRES: clang-driver

/// Check SYCL headers path
// RUN:   %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-HEADERS-INTELFPGA %s
// RUN:   %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-HEADERS-INTELFPGA %s
// CHK-HEADERS-INTELFPGA: clang{{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"

/// -fintelfpga implies -g and -MMD
// RUN:   %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -Xshardware %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-INTELFPGA %s
// CHK-TOOLS-INTELFPGA: clang{{.*}} "-debug-info-kind=constructor" {{.*}} "-dependency-file"
// CHK-TOOLS-INTELFPGA: aoc{{.*}} "-dep-files={{.*}}"

/// -fintelfpga implies -g but -g0 should override
// RUN:   %clangxx -### -target x86_64-unknown-linux-gnu -g0 -fsycl -fintelfpga %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-INTELFPGA-G0 %s
// CHK-TOOLS-INTELFPGA-G0-NOT: clang{{.*}} "-debug-info-kind=constructor"

/// FPGA target implies -fsycl-disable-range-rounding
// RUN:   %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RANGE-ROUNDING %s
// RUN:   %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RANGE-ROUNDING %s
// CHK-RANGE-ROUNDING: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-disable-range-rounding"
// CHK-RANGE-ROUNDING: clang{{.*}} "-fsycl-disable-range-rounding"{{.*}} "-fsycl-is-host"

/// -fsycl-disable-range-rounding is applied to all compilations if fpga is used
// RUN:   %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice,spir64_gen-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RANGE-ROUNDING-MULTI %s
// CHK-RANGE-ROUNDING-MULTI: clang{{.*}} "-triple" "spir64_fpga-unknown-unknown-sycldevice"{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-disable-range-rounding"
// CHK-RANGE-ROUNDING-MULTI: clang{{.*}} "-triple" "spir64_gen-unknown-unknown-sycldevice"{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-disable-range-rounding"
// CHK-RANGE-ROUNDING-MULTI: clang{{.*}} "-fsycl-disable-range-rounding"{{.*}} "-fsycl-is-host"

/// -fintelfpga -fsycl-link tests
// RUN:  touch %t.o
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga -fsycl-link -Xshardware %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-EARLY %s
// RUN:  %clangxx -### -O2 -target x86_64-unknown-linux-gnu -fsycl  -fno-sycl-device-lib=all -fintelfpga -fsycl-link=early -Xshardware %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-EARLY %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga  -fsycl-link=image -Xshardware %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-IMAGE %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice -fsycl-link=image -Xshardware %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-IMAGE %s
// CHK-FPGA-LINK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice" "-inputs=[[INPUT:.+\.o]]" "-outputs=[[OUTPUT1:.+\.o]]" "-unbundle"
// CHK-FPGA-LINK-NOT: clang-offload-bundler{{.*}}
// CHK-FPGA-LINK: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2_1:.+\.bc]]"
// CHK-FPGA-LINK: sycl-post-link{{.*}} "-O2" "-spec-const=default" "-o" "[[OUTPUT2:.+\.table]]" "[[OUTPUT2_1]]"
// CHK-FPGA-LINK: file-table-tform{{.*}} "-o" "[[TABLEOUT:.+\.txt]]" "[[OUTPUT2]]"
// CHK-FPGA-LINK: llvm-spirv{{.*}} "-o" "[[OUTPUT3:.+\.txt]]" "-spirv-max-version={{.*}}"{{.*}} "[[TABLEOUT]]"
// CHK-FPGA-EARLY: aoc{{.*}} "-o" "[[OUTPUT4:.+\.aocr]]" "[[OUTPUT3]]" "-sycl" "-rtl"
// CHK-FPGA-IMAGE: aoc{{.*}} "-o" "[[OUTPUT5:.+\.aocx]]" "[[OUTPUT3]]" "-sycl"
// CHK-FPGA-LINK: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" {{.*}} "-kind=sycl"
// CHK-FPGA-LINK: llc{{.*}} "-o" "[[OBJOUTDEV:.+\.o]]" "[[WRAPOUT]]"
// CHK-FPGA-EARLY: clang-offload-wrapper{{.*}} "-host" "x86_64-unknown-linux-gnu" "-o" "[[WRAPOUTHOST:.+\.bc]]" "-kind=host"
// CHK-FPGA-EARLY-NOT: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-O2"
// CHK-FPGA-EARLY: "-o" "[[OBJOUT:.+\.o]]" {{.*}} "[[WRAPOUTHOST]]"
// CHK-FPGA-EARLY: llvm-ar{{.*}} "cr" "libfoo.a" "[[OBJOUT]]" "[[OBJOUTDEV]]"
// CHK-FPGA-IMAGE: llvm-ar{{.*}} "cr" "libfoo.a" "[[INPUT]]" "[[OBJOUTDEV]]"

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
// RUN:  %clang_cl -### -fsycl -fintelfpga -fno-sycl-device-lib=all -fsycl-link -Xshardware %t.obj -Folibfoo.lib 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-WIN %s
// RUN:  %clang_cl -### -fsycl -fintelfpga -fno-sycl-device-lib=all -fsycl-link -Xshardware %t.obj -o libfoo.lib 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-WIN %s
// CHK-FPGA-LINK-WIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice{{.*}}" "-inputs=[[INPUT:.+\.obj]]" "-outputs=[[OUTPUT1:.+\.obj]]" "-unbundle"
// CHK-FPGA-LINK-WIN-NOT: clang-offload-bundler{{.*}}
// CHK-FPGA-LINK-WIN: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2_1:.+\.bc]]"
// CHK-FPGA-LINK-WIN: sycl-post-link{{.*}} "-O2" "-spec-const=default" "-o" "[[OUTPUT2:.+\.table]]" "[[OUTPUT2_1]]"
// CHK-FPGA-LINK-WIN: file-table-tform{{.*}} "-o" "[[TABLEOUT:.+\.txt]]" "[[OUTPUT2]]"
// CHK-FPGA-LINK-WIN: llvm-spirv{{.*}} "-o" "[[OUTPUT3:.+\.txt]]" "-spirv-max-version={{.*}}"{{.*}} "[[TABLEOUT]]"
// CHK-FPGA-LINK-WIN: aoc{{.*}} "-o" "[[OUTPUT5:.+\.aocr]]" "[[OUTPUT3]]" "-sycl" "-rtl"
// CHK-FPGA-LINK-WIN: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-kind=sycl"
// CHK-FPGA-LINK-WIN: llc{{.*}} "-o" "[[OBJOUTDEV:.+\.obj]]" "[[WRAPOUT]]"
// CHK-FPGA-LINK-WIN: clang-offload-wrapper{{.*}} "-o" "[[WRAPOUTHOST:.+\.bc]]" "-kind=host"
// CHK-FPGA-LINK-WIN: clang{{.*}} "-o" "[[OBJOUT:.+\.obj]]" {{.*}} "[[WRAPOUTHOST]]"
// CHK-FPGA-LINK-WIN: lib.exe{{.*}} "[[OBJOUT]]" "[[OBJOUTDEV]]" {{.*}} "-OUT:libfoo.lib"

/// Check -fintelfpga -fsycl-link with an FPGA archive
// Create the dummy archive
// RUN:  echo "Dummy AOCR image" > %t.aocr
// RUN:  echo "void foo() {}" > %t.c
// RUN:  %clang -c -o %t.o %t.c
// RUN:  clang-offload-wrapper -o %t-aocr.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=fpga_aocr-intel-unknown-sycldevice %t.aocr
// RUN:  llc -filetype=obj -o %t-aocr.o %t-aocr.bc
// RUN:  llvm-ar crv %t-aocr.a %t.o %t-aocr.o
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=image -Xshardware %t-aocr.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-LIB,CHK-FPGA-LINK-LIB-IMAGE %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=early -Xshardware %t-aocr.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-LIB,CHK-FPGA-LINK-LIB-EARLY %s
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocx-intel-unknown-sycldevice" "-inputs={{.*}}" "-check-section"
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocr-intel-unknown-sycldevice" "-inputs={{.*}}" "-check-section"
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=aocr" "-targets=sycl-fpga_aocr-intel-unknown-sycldevice" "-inputs=[[INPUT:.+\.a]]" "-outputs=[[OUTPUT2:.+\.aocr]]" "-unbundle"
// CHK-FPGA-LINK-LIB-IMAGE: llvm-foreach{{.*}} "--out-ext=aocx" "--in-file-list=[[OUTPUT2]]" "--in-replace=[[OUTPUT2]]" "--out-file-list=[[OUTPUT3:.+\.aocx]]" "--out-replace=[[OUTPUT3]]" "--out-increment={{.*}}-aocr.prj" "--" "{{.*}}aoc{{.*}}" "-o" "[[OUTPUT3]]" "[[OUTPUT2]]" "-sycl" "-output-report-folder={{.*}}-aocr.prj" "-g"
// CHK-FPGA-LINK-LIB-IMAGE: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[OUTPUT4:.+\.txt]]" "[[OUTPUT3]]"
// CHK-FPGA-LINK-LIB-IMAGE: clang-offload-wrapper{{.*}} "-host=x86_64-unknown-linux-gnu" "--emit-reg-funcs=0" "-target=fpga_aocx-intel-unknown-sycldevice" "-kind=sycl" "-batch" "[[OUTPUT4]]"
// CHK-FPGA-LINK-LIB-EARLY: llvm-foreach{{.*}} "--out-ext=aocr" "--in-file-list=[[OUTPUT2]]" "--in-replace=[[OUTPUT2]]" "--out-file-list=[[OUTPUT3:.+\.aocr]]" "--out-replace=[[OUTPUT3]]" "--out-increment={{.*}}-aocr.prj" "--" "{{.*}}aoc{{.*}}" "-o" "[[OUTPUT3]]" "[[OUTPUT2]]" "-sycl" "-rtl" "-output-report-folder={{.*}}-aocr.prj" "-g"
// CHK-FPGA-LINK-LIB-EARLY: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[OUTPUT4:.+\.txt]]" "[[OUTPUT3]]"
// CHK-FPGA-LINK-LIB-EARLY: clang-offload-wrapper{{.*}} "-host=x86_64-unknown-linux-gnu" "-target=fpga_aocr-intel-unknown-sycldevice" "-kind=sycl" "-batch" "[[OUTPUT4]]"
// CHK-FPGA-LINK-LIB: llc{{.*}} "-filetype=obj" "-o" "[[OUTPUT5:.+\.o]]"
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=aoo" "-targets=host-x86_64-unknown-linux-gnu" "-inputs=[[INPUT]]" "-outputs=[[OUTPUT1:.+\.txt]]" "-unbundle"
// CHK-FPGA-LINK-LIB-IMAGE: llvm-ar{{.*}} "cr" {{.*}} "@[[OUTPUT1]]"

/// Check the warning's emission for -fsycl-link's appending behavior
// RUN: touch dummy.a
// RUN: %clangxx -fsycl -fintelfpga -fsycl-link=image -target x86_64-unknown-linux-gnu %s -o dummy.a -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHK-FPGA-LINK-WARN
// RUN: %clangxx -fsycl -fintelfpga -fsycl-link=early -target x86_64-unknown-linux-gnu %s -o dummy.a -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHK-FPGA-LINK-WARN
// CHK-FPGA-LINK-WARN: warning: appending to an existing archive 'dummy.a'

/// -fintelfpga -fsycl-link name creation without output file specified
// RUN: mkdir -p %t_dir
// RUN: touch %t_dir/dummy_file.cpp
// RUN: %clangxx -### -fsycl -fintelfpga -fsycl-link -target x86_64-unknown-linux-gnu %t_dir/dummy_file.cpp 2>&1 \
// RUN: | FileCheck -check-prefixes=CHK-SYCL-LINK-LIN -DINPUTSRC=a %s
// RUN: %clang_cl -### -fsycl -fintelfpga -fsycl-link %t_dir/dummy_file.cpp 2>&1 \
// RUN: | FileCheck -check-prefixes=CHK-SYCL-LINK-WIN -DINPUTSRC=a %s
// CHK-SYCL-LINK-LIN: llvm-ar{{.*}} "cr" "[[INPUTSRC]].a"
// CHK-SYCL-LINK-WIN: lib.exe{{.*}} "-OUT:[[INPUTSRC]].a"

/// -fintelfpga with AOCR library and additional object
// RUN:  touch %t2.o
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fno-sycl-device-lib=all -fsycl -fintelfpga -Xshardware %t-aocr.a %t2.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA %s
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=aocr" "-targets=sycl-fpga_aocr-intel-unknown-sycldevice" "-inputs=[[INPUT:.+\.a]]" "-outputs=[[OUTPUT2:.+\.aocr]]" "-unbundle"
// CHK-FPGA: llvm-foreach{{.*}} "--out-ext=aocx" "--in-file-list=[[OUTPUT2]]" "--in-replace=[[OUTPUT2]]" "--out-file-list=[[OUTPUT3:.+\.aocx]]" "--out-replace=[[OUTPUT3]]" "--out-increment={{.*}}" "--" "{{.*}}aoc{{.*}}" "-o" "[[OUTPUT3]]" "[[OUTPUT2]]" "-sycl" {{.*}} "-g"
// CHK-FPGA: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[OUTPUT4:.+\.txt]]" "[[OUTPUT3]]"
// CHK-FPGA: clang-offload-wrapper{{.*}} "-o=[[OUTPUT_AOCX_BC:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_fpga" "-kind=sycl" "-batch" "[[OUTPUT4]]"
// CHK-FPGA: llc{{.*}} "-filetype=obj" "-o" "[[FINALLINK:.+\.o]]" "[[OUTPUT_AOCX_BC]]"
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=o" "-targets=host-x86_64-unknown-linux-gnu,sycl-spir64_fpga-unknown-unknown-sycldevice" {{.*}} "-outputs=[[FINALLINK2:.+\.o]],[[OUTPUT1:.+\.o]]" "-unbundle"
// CHK-FPGA: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2_BC:.+\.bc]]"
// CHK-FPGA: sycl-post-link{{.*}} "-O2" "-spec-const=default" "-o" "[[OUTPUT3_TABLE:.+\.table]]" "[[OUTPUT2_BC]]"
// CHK-FPGA: file-table-tform{{.*}} "-o" "[[TABLEOUT:.+\.txt]]" "[[OUTPUT3_TABLE]]"
// CHK-FPGA: llvm-spirv{{.*}} "-o" "[[OUTPUT5:.+\.txt]]" "-spirv-max-version={{.*}}"{{.*}} "[[TABLEOUT]]"
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-fpga_dep" {{.*}} "-outputs=[[DEPFILE:.+\.d]]" "-unbundle"
// CHK-FPGA: aoc{{.*}} "-o" "[[OUTPUT6:.+\.aocx]]" "[[OUTPUT5]]" "-sycl" "-dep-files=[[DEPFILE]]"
// CHK-FPGA: file-table-tform{{.*}} "-replace=Code,Code" "-o" "[[TABLEOUT2:.+\.table]]" "[[OUTPUT3_TABLE]]" "[[OUTPUT6]]"
// CHK-FPGA: clang-offload-wrapper{{.*}} "-o=[[OUTPUT7:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_fpga" "-kind=sycl" "-batch" "[[TABLEOUT2]]"
// CHK-FPGA: llc{{.*}} "-filetype=obj" "-o" "[[FINALLINK3:.+\.o]]" "[[OUTPUT7]]"
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=aoo" "-targets=host-x86_64-unknown-linux-gnu" {{.*}} "-outputs=[[FINALLINK4:.+\.txt]]" "-unbundle"
// CHK-FPGA: {{link|ld}}{{.*}} "@[[FINALLINK4]]" "[[FINALLINK2]]" "[[FINALLINK]]" "[[FINALLINK3]]"

/// Check the warning's emission for conflicting emulation/hardware (AOCR)
// RUN: touch %t-aocr.a
// RUN: %clangxx -fsycl -fintelfpga -fsycl-link=image -target x86_64-unknown-linux-gnu %t-aocr.a %s -### 2>&1 \
// RUN:  | FileCheck %s --check-prefix=CHK-FPGA-LINK-WARN-AOCR
// CHK-FPGA-LINK-WARN-AOCR: warning: FPGA archive '{{.*}}-aocr.a' does not contain matching emulation/hardware expectancy

/// -fintelfpga with AOCX library
// Create the dummy archive
// RUN:  echo "Dummy AOCX image" > %t.aocx
// RUN:  echo "void foo() {}" > %t.c
// RUN:  %clang -c -o %t.o %t.c
// RUN:  clang-offload-wrapper -o %t-aocx.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=fpga_aocx-intel-unknown-sycldevice %t.aocx
// RUN:  llc -filetype=obj -o %t-aocx.o %t-aocx.bc
// RUN:  llvm-ar crv %t_aocx.a %t.o %t-aocx.o
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -Xshardware -fsycl -fintelfpga %t_aocx.a -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-PHASES %s
// RUN:  %clang_cl --target=x86_64-pc-windows-msvc -Xshardware -fsycl -fintelfpga %t_aocx.a -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-PHASES %s
// CHK-FPGA-AOCX-PHASES: 0: input, "{{.*}}", fpga_aocx, (host-sycl)
// CHK-FPGA-AOCX-PHASES: 1: linker, {0}, image, (host-sycl)
// CHK-FPGA-AOCX-PHASES: 2: clang-offload-unbundler, {0}, fpga_aocx
// CHK-FPGA-AOCX-PHASES: 3: file-table-tform, {2}, tempfilelist, (device-sycl)
// CHK-FPGA-AOCX-PHASES: 4: clang-offload-wrapper, {3}, object, (device-sycl)
// CHK-FPGA-AOCX-PHASES: 5: offload, "host-sycl ({{.*}}x86_64{{.*}})" {1}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {4}, image

// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -Xshardware %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX,CHK-FPGA-AOCX-LIN %s
// RUN:  %clang_cl -fsycl -fintelfpga -Xshardware %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX,CHK-FPGA-AOCX-WIN %s
// CHK-FPGA-AOCX: clang-offload-bundler{{.*}} "-type=aocx" "-targets=sycl-fpga_aocx-intel-unknown-sycldevice" "-inputs=[[LIBINPUT:.+\.a]]" "-outputs=[[BUNDLEOUT:.+\.aocx]]" "-unbundle"
// CHK-FPGA-AOCX: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[TABLEOUT:.+\.txt]]" "[[BUNDLEOUT]]"
// CHK-FPGA-AOCX: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "-batch" "[[TABLEOUT]]"
// CHK-FPGA-AOCX-LIN: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUT:.+\.o]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-WIN: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUT2:.+\.obj]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-NOT: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocx-intel-unknown-sycldevice"
// CHK-FPGA-AOCX-LIN: ld{{.*}} "[[LIBINPUT]]" "[[LLCOUT]]"
// CHK-FPGA-AOCX-WIN: link{{.*}} "[[LIBINPUT]]" "[[LLCOUT2]]"

/// AOCX with source
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -Xshardware -fno-sycl-device-lib=all %s %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-SRC,CHK-FPGA-AOCX-SRC-LIN %s
// RUN:  %clang_cl -fsycl -fno-sycl-device-lib=all -Xshardware -fintelfpga %s %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-SRC,CHK-FPGA-AOCX-SRC-WIN %s
// CHK-FPGA-AOCX-SRC: clang-offload-bundler{{.*}} "-type=aocx" "-targets=sycl-fpga_aocx-intel-unknown-sycldevice" "-inputs=[[LIBINPUT:.+\.a]]" "-outputs=[[BUNDLEOUT:.+\.aocx]]" "-unbundle"
// CHK-FPGA-AOCX-SRC: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[TABLEOUT:.+\.txt]]" "[[BUNDLEOUT]]"
// CHK-FPGA-AOCX-SRC: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "-batch" "[[TABLEOUT]]"
// CHK-FPGA-AOCX-SRC: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUT:.+\.(o|obj)]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-SRC: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device" {{.*}} "-o" "[[DEVICEBC:.+\.bc]]"
// CHK-FPGA-AOCX-SRC: llvm-link{{.*}} "[[DEVICEBC]]" "-o" "[[LLVMLINKOUT:.+\.bc]]" "--suppress-warnings"
// CHK-FPGA-AOCX-SRC: sycl-post-link{{.*}} "-O2" "-spec-const=default" "-o" "[[POSTLINKOUT:.+\.table]]" "[[LLVMLINKOUT]]
// CHK-FPGA-AOCX-SRC: file-table-tform{{.*}} "-o" "[[TABLEOUT:.+\.txt]]" "[[POSTLINKOUT]]"
// CHK-FPGA-AOCX-SRC: llvm-spirv{{.*}} "-o" "[[LLVMSPVOUT:.+\.txt]]" {{.*}} "[[TABLEOUT]]"
// CHK-FPGA-AOCX-SRC: aoc{{.*}} "-o" "[[AOCOUT:.+\.aocx]]" "[[LLVMSPVOUT]]" "-sycl"
// CHK-FPGA-AOCX-SRC: file-table-tform{{.*}} "-replace=Code,Code" "-o" "[[TABLEOUT2:.+\.table]]" "[[POSTLINKOUT]]" "[[AOCOUT]]"
// CHK-FPGA-AOCX-SRC: clang-offload-wrapper{{.*}} "-o=[[WRAPOUTSRC:.+.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "-batch" "[[TABLEOUT2]]"
// CHK-FPGA-AOCX-SRC: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUTSRC:.+\.(o|obj)]]" "[[WRAPOUTSRC]]"
// CHK-FPGA-AOCX-SRC: clang{{.*}} "-fsycl-is-host" {{.*}} "-o" "[[HOSTOBJ:.+\.(o|obj)]]"
// CHK-FPGA-AOCX-SRC-LIN: ld{{.*}} "[[HOSTOBJ]]" "[[LIBINPUT]]" "[[LLCOUT]]" "[[LLCOUTSRC]]"
// CHK-FPGA-AOCX-SRC-WIN: link{{.*}} "[[HOSTOBJ]]" "[[LIBINPUT]]" "[[LLCOUT]]" "[[LLCOUTSRC]]"

/// AOCX with object
// RUN: touch %t.o
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga -Xshardware %t.o %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-OBJ,CHK-FPGA-AOCX-OBJ-LIN %s
// RUN:  %clang_cl -fsycl -fno-sycl-device-lib=all -fintelfpga -Xshardware %t.o %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-OBJ,CHK-FPGA-AOCX-OBJ-WIN %s
// CHK-FPGA-AOCX-OBJ: clang-offload-bundler{{.*}} "-type=aocx" "-targets=sycl-fpga_aocx-intel-unknown-sycldevice" "-inputs=[[LIBINPUT:.+\.a]]" "-outputs=[[BUNDLEOUT:.+\.aocx]]" "-unbundle"
// CHK-FPGA-AOCX-OBJ: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[TABLEOUT:.+\.txt]]" "[[BUNDLEOUT]]"
// CHK-FPGA-AOCX-OBJ: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "-batch" "[[TABLEOUT]]"
// CHK-FPGA-AOCX-OBJ: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUT:.+\.(o|obj)]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-OBJ: clang-offload-bundler{{.*}} "-type=o" {{.*}} "-outputs=[[HOSTOBJ:.+\.(o|obj)]],[[DEVICEOBJ:.+\.(o|obj)]]" "-unbundle"
// CHK-FPGA-AOCX-OBJ: llvm-link{{.*}} "[[DEVICEOBJ]]" "-o" "[[LLVMLINKOUT:.+\.bc]]" "--suppress-warnings"
// CHK-FPGA-AOCX-OBJ: sycl-post-link{{.*}} "-O2" "-spec-const=default" "-o" "[[POSTLINKOUT:.+\.table]]" "[[LLVMLINKOUT]]
// CHK-FPGA-AOCX-OBJ: file-table-tform{{.*}} "-o" "[[TABLEOUT:.+\.txt]]" "[[POSTLINKOUT]]"
// CHK-FPGA-AOCX-OBJ: llvm-spirv{{.*}} "-o" "[[LLVMSPVOUT:.+\.txt]]" {{.*}} "[[TABLEOUT]]"
// CHK-FPGA-AOCX-OBJ: aoc{{.*}} "-o" "[[AOCOUT:.+\.aocx]]" "[[LLVMSPVOUT]]" "-sycl"
// CHK-FPGA-AOCX-OBJ: file-table-tform{{.*}} "-replace=Code,Code" "-o" "[[TABLEOUT2:.+\.table]]" "[[POSTLINKOUT]]" "[[AOCOUT]]"
// CHK-FPGA-AOCX-OBJ: clang-offload-wrapper{{.*}} "-o=[[WRAPOUTSRC:.+.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "-batch" "[[TABLEOUT2]]"
// CHK-FPGA-AOCX-OBJ: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUTSRC:.+\.(o|obj)]]" "[[WRAPOUTSRC]]"
// CHK-FPGA-AOCX-OBJ-LIN: ld{{.*}} "[[HOSTOBJ]]" "[[LIBINPUT]]" "[[LLCOUT]]" "[[LLCOUTSRC]]"
// CHK-FPGA-AOCX-OBJ-WIN: link{{.*}} "[[HOSTOBJ]]" "[[LIBINPUT]]" "[[LLCOUT]]" "[[LLCOUTSRC]]"

/// AOCX with object, pulling in additional triples
// RUN: touch %t.o
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64,spir64_fpga -Xshardware %t.o %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-OBJ2 %s
// RUN:  %clang_cl -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64,spir64_fpga -Xshardware %t.o %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-OBJ2 %s
// CHK-FPGA-AOCX-OBJ2: clang-offload-bundler{{.*}} "-type=o" {{.*}} "-outputs=[[HOSTOBJ:.+\.(o|obj)]],[[DEVICEOBJ:.+\.(o|obj)]],[[DEVICEOBJ2:.+\.(o|obj)]]" "-unbundle"
// CHK-FPGA-AOCX-OBJ2: llvm-link{{.*}} "[[DEVICEOBJ]]" "-o" "[[LLVMLINKOUT:.+\.bc]]" "--suppress-warnings"
// CHK-FPGA-AOCX-OBJ2: sycl-post-link{{.*}} "-O2" "-spec-const=rt" "-o" "[[POSTLINKOUT:.+\.table]]" "[[LLVMLINKOUT]]"
// CHK-FPGA-AOCX-OBJ2: file-table-tform{{.*}} "-o" "[[TABLEOUT:.+\.txt]]" "[[POSTLINKOUT]]"
// CHK-FPGA-AOCX-OBJ2: llvm-spirv{{.*}} "-o" "[[LLVMSPVOUT:.+\.txt]]" {{.*}} "[[TABLEOUT]]"
// CHK-FPGA-AOCX-OBJ2: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-target=spir64" "-kind=sycl" "-batch"
// CHK-FPGA-AOCX-OBJ2: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUT:.+\.(o|obj)]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-OBJ2: clang-offload-bundler{{.*}} "-type=aocx" "-targets=sycl-spir64-unknown-unknown-sycldevice,sycl-fpga_aocx-intel-unknown-sycldevice" "-inputs=[[LIBINPUT:.+\.a]]" "-outputs={{.*}},[[BUNDLEOUT:.+\.aocx]]" "-unbundle"
// CHK-FPGA-AOCX-OBJ2: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[TABLEOUT:.+\.txt]]" "[[BUNDLEOUT]]"
// CHK-FPGA-AOCX-OBJ2: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "-batch" "[[TABLEOUT]]"
// CHK-FPGA-AOCX-OBJ2: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUT2:.+\.(o|obj)]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-OBJ2: llvm-link{{.*}} "[[DEVICEOBJ2]]" "-o" "[[LLVMLINKOUT2:.+\.bc]]" "--suppress-warnings"
// CHK-FPGA-AOCX-OBJ2: sycl-post-link{{.*}} "-O2" "-spec-const=default" "-o" "[[POSTLINKOUT2:.+\.table]]" "[[LLVMLINKOUT2]]"
// CHK-FPGA-AOCX-OBJ2: file-table-tform{{.*}} "-o" "[[TABLEOUT2:.+\.txt]]" "[[POSTLINKOUT2]]"
// CHK-FPGA-AOCX-OBJ2: llvm-spirv{{.*}} "-o" "[[LLVMSPVOUT2:.+\.txt]]" {{.*}} "[[TABLEOUT2]]"
// CHK-FPGA-AOCX-OBJ2: aoc{{.*}} "-o" "[[AOCOUT:.+\.aocx]]" "[[LLVMSPVOUT2]]" "-sycl"
// CHK-FPGA-AOCX-OBJ2: file-table-tform{{.*}} "-replace=Code,Code" "-o" "[[TABLEOUT3:.+\.table]]" "[[POSTLINKOUT2]]" "[[AOCOUT]]"
// CHK-FPGA-AOCX-OBJ2: clang-offload-wrapper{{.*}} "-o=[[WRAPOUTSRC:.+.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "-batch" "[[TABLEOUT3]]"
// CHK-FPGA-AOCX-OBJ2: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUTSRC:.+\.(o|obj)]]" "[[WRAPOUTSRC]]"
// CHK-FPGA-AOCX-OBJ2: {{(ld|link).*}} "[[HOSTOBJ]]" "[[LIBINPUT]]" "[[LLCOUT]]" "[[LLCOUT2]]" "[[LLCOUTSRC]]"

/// -fintelfpga -fsycl-link from source
// RUN: touch %t.cpp
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga -fsycl-link=early %t.cpp -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-SRC %s
// RUN: %clang_cl --target=x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga -fsycl-link=early %t.cpp -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-SRC %s
// CHK-FPGA-LINK-SRC: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHK-FPGA-LINK-SRC: 1: append-footer, {0}, c++, (host-sycl)
// CHK-FPGA-LINK-SRC: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-FPGA-LINK-SRC: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHK-FPGA-LINK-SRC: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-FPGA-LINK-SRC: 5: compiler, {4}, ir, (device-sycl)
// CHK-FPGA-LINK-SRC: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// CHK-FPGA-LINK-SRC: 7: compiler, {6}, ir, (host-sycl)
// CHK-FPGA-LINK-SRC: 8: backend, {7}, assembler, (host-sycl)
// CHK-FPGA-LINK-SRC: 9: assembler, {8}, object, (host-sycl)
// CHK-FPGA-LINK-SRC: 10: clang-offload-wrapper, {9}, ir, (host-sycl)
// CHK-FPGA-LINK-SRC: 11: backend, {10}, assembler, (host-sycl)
// CHK-FPGA-LINK-SRC: 12: assembler, {11}, object, (host-sycl)
// CHK-FPGA-LINK-SRC: 13: linker, {12}, archive, (host-sycl)
// CHK-FPGA-LINK-SRC: 14: linker, {5}, ir, (device-sycl)
// CHK-FPGA-LINK-SRC: 15: sycl-post-link, {14}, tempfiletable, (device-sycl)
// CHK-FPGA-LINK-SRC: 16: file-table-tform, {15}, tempfilelist, (device-sycl)
// CHK-FPGA-LINK-SRC: 17: llvm-spirv, {16}, tempfilelist, (device-sycl)
// CHK-FPGA-LINK-SRC: 18: backend-compiler, {17}, fpga_aocr_emu, (device-sycl)
// CHK-FPGA-LINK-SRC: 19: file-table-tform, {15, 18}, tempfiletable, (device-sycl)
// CHK-FPGA-LINK-SRC: 20: clang-offload-wrapper, {19}, object, (device-sycl)
// CHK-FPGA-LINK-SRC: 21: offload, "host-sycl (x86_64-unknown-linux-gnu)" {13}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {20}, archive

/// -fintelfpga with -reuse-exe=
// RUN:  touch %t.cpp
// RUN:  %clangxx -### -reuse-exe=testing -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -Xshardware %t.cpp 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-REUSE-EXE %s
// RUN:  %clang_cl -### -reuse-exe=testing -fsycl -fintelfpga -Xshardware %t.cpp 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-REUSE-EXE %s
// CHK-FPGA-REUSE-EXE: aoc{{.*}} "-o" {{.*}} "-sycl" {{.*}} "-reuse-exe=testing"

/// -fintelfpga dependency file generation test
// RUN: touch %t-1.cpp
// RUN: touch %t-2.cpp
// RUN: %clangxx -### -fsycl -fintelfpga -Xshardware %t-1.cpp %t-2.cpp -o %t.out 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %t-1.cpp %t-2.cpp -o %t.out 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES %s
// RUN: %clangxx -### -fsycl -fintelfpga -Xshardware %t-1.cpp %t-2.cpp 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %t-1.cpp %t-2.cpp 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES %s
// CHK-FPGA-DEP-FILES: clang{{.*}} "-dependency-file" "[[INPUT1:.+\.d]]" "-MT" "{{.*}}.o"
// CHK-FPGA-DEP-FILES: clang{{.*}} "-dependency-file" "[[INPUT2:.+\.d]]" "-MT" "{{.*}}.o"
// CHK-FPGA-DEP-FILES: aoc{{.*}} "-dep-files={{.*}}[[INPUT1]],{{.*}}[[INPUT2]]"
// CHK-FPGA-DEP-FILES-NOT: clang{{.*}} "-fsycl-is-host"{{.*}} "-dependency-file"

/// -fintelfpga dependency file check with host .d enabled
// RUN: %clangxx -### -MMD -fsycl -fintelfpga -Xshardware %t-1.cpp %t-2.cpp 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES-HOST %s
// CHK-FPGA-DEP-FILES-HOST: clang{{.*}} "-triple" "spir64_fpga-unknown-unknown-sycldevice"{{.*}} "-dependency-file" "[[INPUT1:.+\.d]]" "-MT" "{{.*}}.o"
// CHK-FPGA-DEP-FILES-HOST: clang{{.*}} "-triple" "spir64_fpga-unknown-unknown-sycldevice"{{.*}} "-dependency-file" "[[INPUT2:.+\.d]]" "-MT" "{{.*}}.o"
// CHK-FPGA-DEP-FILES-HOST: aoc{{.*}} "-dep-files={{.*}}[[INPUT1]],{{.*}}[[INPUT2]]"
// CHK-FPGA-DEP-FILES-HOST: clang{{.*}} "-fsycl-is-host"{{.*}}
// CHK-FPGA-DEP-FILES-HOST: clang{{.*}} "-fsycl-is-host"{{.*}}

/// -fintelfpga dependency file generation test to object
// RUN: %clangxx -### -fsycl -fintelfpga -target x86_64-unknown-linux-gnu %t-1.cpp %t-2.cpp -c 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-DEP-FILES2,CHK-FPGA-DEP-FILES2-LIN %s
// RUN: %clang_cl -### -fsycl -fintelfpga --target=x86_64-pc-windows-msvc %t-1.cpp %t-2.cpp -c 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-DEP-FILES2,CHK-FPGA-DEP-FILES2-WIN %s
// CHK-FPGA-DEP-FILES2: clang{{.*}} "-dependency-file" "[[INPUT1:.+\.d]]"
// CHK-FPGA-DEP-FILES2-LIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice,host-x86_64-unknown-linux-gnu,sycl-fpga_dep" {{.*}} "-inputs={{.*}}.bc,{{.*}}.o,[[INPUT1]]"
// CHK-FPGA-DEP-FILES2-WIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice,host-x86_64-pc-windows-msvc,sycl-fpga_dep" {{.*}} "-inputs={{.*}}.bc,{{.*}}.obj,[[INPUT1]]"
// CHK-FPGA-DEP-FILES2: clang{{.*}} "-dependency-file" "[[INPUT2:.+\.d]]"
// CHK-FPGA-DEP-FILES2-LIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice,host-x86_64-unknown-linux-gnu,sycl-fpga_dep" {{.*}} "-inputs={{.*}}.bc,{{.*}}.o,[[INPUT2]]"
// CHK-FPGA-DEP-FILES2-WIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice,host-x86_64-pc-windows-msvc,sycl-fpga_dep" {{.*}} "-inputs={{.*}}.bc,{{.*}}.obj,[[INPUT2]]"

/// -fintelfpga dependency file test to object with output designator
// RUN: touch %t-1.cpp
// RUN: %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %t-1.cpp -c -o dummy.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-DEP-FILES3,CHK-FPGA-DEP-FILES3-LIN %s
// RUN: %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %t-1.cpp -c -MMD -MF"dummy.d" 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-DEP-FILES3,CHK-FPGA-DEP-FILES3-LIN %s
// RUN: %clang_cl -### --target=x86_64-pc-windows-msvc -fsycl -fintelfpga %t-1.cpp -c -Fodummy.obj 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-DEP-FILES3,CHK-FPGA-DEP-FILES3-WIN %s
// CHK-FPGA-DEP-FILES3: clang{{.*}} "-triple" "spir64_fpga-unknown-unknown-sycldevice"{{.*}} "-dependency-file" "[[OUTPUT:.+\.d]]"
// CHK-FPGA-DEP-FILES3-LIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice,host-x86_64-unknown-linux-gnu,sycl-fpga_dep" {{.*}} "-inputs={{.*}}.bc,{{.*}}.o,[[OUTPUT]]"
// CHK-FPGA-DEP-FILES3-WIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice,host-x86_64-pc-windows-msvc,sycl-fpga_dep" {{.*}} "-inputs={{.*}}.bc,{{.*}}.obj,[[OUTPUT]]"

/// -fintelfpga dependency obj use test
// RUN: touch %t-1.o
// RUN: touch %t-2.o
// RUN: %clangxx -### -fsycl -fintelfpga -Xshardware -target x86_64-unknown-linux-gnu %t-1.o %t-2.o 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES-OBJ %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %t-1.o %t-2.o 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES-OBJ %s
// CHK-FPGA-DEP-FILES-OBJ: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-fpga_dep" "-inputs={{.*}}-1.o" "-outputs=[[DEPFILE1:.+\.d]]" "-unbundle"
// CHK-FPGA-DEP-FILES-OBJ: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-fpga_dep" "-inputs={{.*}}-2.o" "-outputs=[[DEPFILE2:.+\.d]]" "-unbundle"
// CHK-FPGA-DEP-FILES-OBJ: aoc{{.*}} "-dep-files=[[DEPFILE1]],[[DEPFILE2]]

/// -fintelfpga dependency file use from object phases test
// RUN: touch %t-1.o
// RUN: %clangxx -fsycl -fno-sycl-device-lib=all -fintelfpga -ccc-print-phases %t-1.o 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES-OBJ-PHASES %s
// RUN: %clang_cl -fsycl -fno-sycl-device-lib=all -fintelfpga -ccc-print-phases %t-1.o 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES-OBJ-PHASES %s
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 0: input, "{{.*}}-1.o", object, (host-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 1: clang-offload-unbundler, {0}, object, (host-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 2: linker, {1}, image, (host-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 3: linker, {1}, ir, (device-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 4: sycl-post-link, {3}, tempfiletable, (device-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 5: file-table-tform, {4}, tempfilelist, (device-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 6: llvm-spirv, {5}, tempfilelist, (device-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 7: clang-offload-unbundler, {0}, fpga_dep
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 8: backend-compiler, {6, 7}, fpga_aocx, (device-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 9: file-table-tform, {4, 8}, tempfiletable, (device-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 10: clang-offload-wrapper, {9}, object, (device-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 11: offload, "host-sycl ({{.*}})" {2}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {10}, image


/// -fintelfpga output report file test
// RUN: mkdir -p %t_dir
// RUN: %clangxx -### -fsycl -fintelfpga -Xshardware %s -o %t_dir/file.out 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %s -o %t_dir/file.out 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %s -Fe%t_dir/file.out 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s

// RUN: %clangxx -### -fsycl -fintelfpga -Xshardware %s -o %t_dir/file.o 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %s -o %t_dir/file.o 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %s -Fe%t_dir/file.o 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s

// RUN: %clangxx -### -fsycl -fintelfpga -Xshardware %s -o %t_dir/file.a 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %s -o %t_dir/file.a 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %s -Fe%t_dir/file.a 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s

// RUN: %clangxx -### -fsycl -fintelfpga -Xshardware %s -o %t_dir/file.lib 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %s -o %t_dir/file.lib 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %s -Fe%t_dir/file.lib 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s

// RUN: %clangxx -### -fsycl -fintelfpga -Xshardware %s -o %t_dir/file.obj 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %s -o %t_dir/file.obj 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %s -Fe%t_dir/file.obj 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s

// RUN: %clangxx -### -fsycl -fintelfpga -Xshardware %s -o %t_dir/file.exe 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %s -o %t_dir/file.exe 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %s -Fe%t_dir/file.exe 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s

// RUN: %clangxx -### -fsycl -fintelfpga -Xshardware %s -o %t_dir/file.xxx 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT-KEEP-EXT %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %s -o %t_dir/file.xxx 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT-KEEP-EXT %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %s -Fe%t_dir/file.xxx 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT-KEEP-EXT %s
// CHK-FPGA-REPORT-OPT: aoc{{.*}} "-sycl" {{.*}} "-output-report-folder={{.*}}{{(/|\\\\)}}file.prj"
// CHK-FPGA-REPORT-OPT-KEEP-EXT: aoc{{.*}} "-sycl" {{.*}} "-output-report-folder={{.*}}{{(/|\\\\)}}file.xxx.prj"

/// -fintelfpga output report file from dir/source
/// check dependency file from dir/source
// RUN: mkdir -p %t_dir
// RUN: touch %t_dir/dummy.cpp
// RUN: %clangxx -### -fsycl -fintelfpga -Xshardware %t_dir/dummy.cpp  2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT2 %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %t_dir/dummy.cpp 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT2 %s
// CHK-FPGA-REPORT-OPT2: aoc{{.*}} "-sycl"{{.*}} "-dep-files={{.+}}dummy-{{.+}}.d" "-output-report-folder={{.*}}dummy.prj"
// CHK-FPGA-REPORT-OPT2-NOT: aoc{{.*}} "-sycl" {{.*}}_dir{{.*}}

/// -fintelfpga dependency files from multiple source
// RUN: touch dummy2.cpp
// RUN: %clangxx -### -fsycl -fintelfpga -Xshardware %t_dir/dummy.cpp dummy2.cpp 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-MULTI-DEPS %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %t_dir/dummy.cpp dummy2.cpp 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-MULTI-DEPS %s
// CHK-FPGA-MULTI-DEPS: aoc{{.*}} "-sycl"{{.*}} "-dep-files={{.+}}dummy-{{.+}}.d,{{.+}}dummy2-{{.+}}.d" "-output-report-folder={{.*}}dummy.prj"

/// -fintelfpga output report file should be based on first input (src/obj)
// RUN: mkdir -p %t_dir
// RUN: touch %t_dir/dummy1.cpp
// RUN: touch %t_dir/dummy2.cpp
// RUN: touch %t_dir/dummy1.o
// RUN: touch %t_dir/dummy2.o
// RUN: %clangxx -### -fsycl -fintelfpga -Xshardware %t_dir/dummy2.o %t_dir/dummy1.cpp  2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-REPORT-NAME %s
// RUN: %clangxx -### -fsycl -fintelfpga -Xshardware %t_dir/dummy2.cpp %t_dir/dummy1.o  2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-REPORT-NAME %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %t_dir/dummy2.o %t_dir/dummy1.cpp 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-REPORT-NAME %s
// RUN: %clang_cl -### -fsycl -fintelfpga -Xshardware %t_dir/dummy2.cpp %t_dir/dummy1.o 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-REPORT-NAME %s
// CHK-FPGA-REPORT-NAME: aoc{{.*}} "-sycl"{{.*}} "-output-report-folder={{.*}}dummy2.prj"

/// Check for implied options with -Xshardware (-g -O0)
/// Expectation is for -O0 to not be used with -Xshardware
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -g -O0 -Xs "-DFOO1 -DFOO2" -Xshardware %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS %s
// RUN:   %clang_cl -### -fsycl -fintelfpga -Zi -Od -Xs "-DFOO1 -DFOO2" -Xshardware %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS %s
// CHK-TOOLS-IMPLIED-OPTS-NOT: clang{{.*}} "-fsycl-is-device"{{.*}} "-O0"
// CHK-TOOLS-IMPLIED-OPTS: sycl-post-link{{.*}} "-O2"
// CHK-TOOLS-IMPLIED-OPTS: aoc{{.*}} "-g" "-DFOO1" "-DFOO2"
