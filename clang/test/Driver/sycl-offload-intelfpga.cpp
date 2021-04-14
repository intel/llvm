///
/// tests specific to -fintelfpga -fsycl
///
// REQUIRES: clang-driver

/// Check SYCL headers path
// RUN:   %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %s 2>&1 \
// RUN:   %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice %s 2>&1 \
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
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga -fsycl-link %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-EARLY %s
// RUN:  %clangxx -### -O2 -target x86_64-unknown-linux-gnu -fsycl  -fno-sycl-device-lib=all -fintelfpga -fsycl-link=early %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-EARLY %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga  -fsycl-link=image %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-IMAGE %s
// CHK-FPGA-LINK-NOT: clang-offload-bundler{{.*}} "-check-section"
// CHK-FPGA-LINK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice" "-inputs=[[INPUT:.+\.o]]" "-outputs=[[OUTPUT1:.+\.o]]" "-unbundle"
// CHK-FPGA-LINK-NOT: clang-offload-bundler{{.*}}
// CHK-FPGA-LINK: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2_1:.+\.bc]]"
// CHK-FPGA-LINK: sycl-post-link
// CHK-FPGA-LINK-NOT: -split-esimd
// CHK-FPGA-LINK: "-ir-output-only" "-O2" "-spec-const=default" "-o" "[[OUTPUT2:.+\.bc]]" "[[OUTPUT2_1]]"
// CHK-FPGA-LINK: llvm-spirv{{.*}} "-o" "[[OUTPUT3:.+\.spv]]" "-spirv-max-version=1.3" "-spirv-debug-info-version=legacy" "-spirv-allow-extra-diexpressions" "-spirv-allow-unknown-intrinsics=llvm.genx." "-spirv-ext=+all,-SPV_INTEL_usm_storage_classes,-SPV_INTEL_optnone,-SPV_KHR_linkonce_odr" "[[OUTPUT2]]"
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
// RUN:  %clang_cl -### -fsycl -fintelfpga -fno-sycl-device-lib=all -fsycl-link %t.obj -Folibfoo.lib 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-WIN %s
// RUN:  %clang_cl -### -fsycl -fintelfpga -fno-sycl-device-lib=all -fsycl-link %t.obj -o libfoo.lib 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-WIN %s
// CHK-FPGA-LINK-WIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice{{.*}}" "-inputs=[[INPUT:.+\.obj]]" "-outputs=[[OUTPUT1:.+\.obj]]" "-unbundle"
// CHK-FPGA-LINK-WIN-NOT: clang-offload-bundler{{.*}}
// CHK-FPGA-LINK-WIN: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2_1:.+\.bc]]"
// CHK-FPGA-LINK-WIN: sycl-post-link
// CHK-FPGA-LINK-WIN-NOT: -split-esimd
// CHK-FPGA-LINK-WIN: "-ir-output-only" "-O2" "-spec-const=default" "-o" "[[OUTPUT2:.+\.bc]]" "[[OUTPUT2_1]]"
// CHK-FPGA-LINK-WIN: llvm-spirv{{.*}} "-o" "[[OUTPUT3:.+\.spv]]" "-spirv-max-version=1.3" "-spirv-debug-info-version=legacy" "-spirv-allow-extra-diexpressions" "-spirv-allow-unknown-intrinsics=llvm.genx." "-spirv-ext=+all,-SPV_INTEL_usm_storage_classes,-SPV_INTEL_optnone,-SPV_KHR_linkonce_odr" "[[OUTPUT2]]"
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
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=image %t-aocr.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-LIB,CHK-FPGA-LINK-LIB-IMAGE %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=early %t-aocr.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-LIB,CHK-FPGA-LINK-LIB-EARLY %s
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocx-intel-unknown-sycldevice" "-inputs={{.*}}" "-check-section"
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocr-intel-unknown-sycldevice" "-inputs={{.*}}" "-check-section"
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=aocr" "-targets=sycl-fpga_aocr-intel-unknown-sycldevice" "-inputs=[[INPUT:.+\.a]]" "-outputs=[[OUTPUT2:.+\.aocr]]" "-unbundle"
// CHK-FPGA-LINK-LIB-IMAGE: llvm-foreach{{.*}} "--out-ext=aocx" "--in-file-list=[[OUTPUT2]]" "--in-replace=[[OUTPUT2]]" "--out-file-list=[[OUTPUT3:.+\.aocx]]" "--out-replace=[[OUTPUT3]]" "--" "{{.*}}aoc{{.*}}" "-o" "[[OUTPUT3]]" "[[OUTPUT2]]" "-sycl" "-output-report-folder={{.*}}-aocr.prj" "-g"
// CHK-FPGA-LINK-LIB-IMAGE: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[OUTPUT4:.+\.txt]]" "[[OUTPUT3]]"
// CHK-FPGA-LINK-LIB-IMAGE: clang-offload-wrapper{{.*}} "-host=x86_64-unknown-linux-gnu" "--emit-reg-funcs=0" "-target=fpga_aocx-intel-unknown-sycldevice" "-kind=sycl" "-batch" "[[OUTPUT4]]"
// CHK-FPGA-LINK-LIB-EARLY: llvm-foreach{{.*}} "--out-ext=aocr" "--in-file-list=[[OUTPUT2]]" "--in-replace=[[OUTPUT2]]" "--out-file-list=[[OUTPUT3:.+\.aocr]]" "--out-replace=[[OUTPUT3]]" "--" "{{.*}}aoc{{.*}}" "-o" "[[OUTPUT3]]" "[[OUTPUT2]]" "-sycl" "-rtl" "-output-report-folder={{.*}}-aocr.prj" "-g"
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
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fno-sycl-device-lib=all -fsycl -fintelfpga %t-aocr.a %t2.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA %s
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=aocr" "-targets=sycl-fpga_aocr-intel-unknown-sycldevice" "-inputs=[[INPUT:.+\.a]]" "-outputs=[[OUTPUT2:.+\.aocr]]" "-unbundle"
// CHK-FPGA: llvm-foreach{{.*}} "--out-ext=aocx" "--in-file-list=[[OUTPUT2]]" "--in-replace=[[OUTPUT2]]" "--out-file-list=[[OUTPUT3:.+\.aocx]]" "--out-replace=[[OUTPUT3]]" "--" "{{.*}}aoc{{.*}}" "-o" "[[OUTPUT3]]" "[[OUTPUT2]]" "-sycl" {{.*}} "-g"
// CHK-FPGA: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[OUTPUT4:.+\.txt]]" "[[OUTPUT3]]"
// CHK-FPGA: clang-offload-wrapper{{.*}} "-o=[[OUTPUT_AOCX_BC:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_fpga" "-kind=sycl" "-batch" "[[OUTPUT4]]"
// CHK-FPGA: llc{{.*}} "-filetype=obj" "-o" "[[FINALLINK:.+\.o]]" "[[OUTPUT_AOCX_BC]]"
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=o" "-targets=host-x86_64-unknown-linux-gnu,sycl-spir64_fpga-unknown-unknown-sycldevice" {{.*}} "-outputs=[[FINALLINK2:.+\.o]],[[OUTPUT1:.+\.o]]" "-unbundle"
// CHK-FPGA: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2_BC:.+\.bc]]"
// CHK-FPGA: sycl-post-link
// CHK-FPGA-NOT: -split-esimd
// CHK-FPGA: "-ir-output-only" "-O2" "-spec-const=default" "-o" "[[OUTPUT3_BC:.+\.bc]]" "[[OUTPUT2_BC]]"
// CHK-FPGA: llvm-spirv{{.*}} "-o" "[[OUTPUT5:.+\.spv]]" "-spirv-max-version=1.3" "-spirv-debug-info-version=legacy" "-spirv-allow-extra-diexpressions" "-spirv-allow-unknown-intrinsics=llvm.genx." "-spirv-ext=+all,-SPV_INTEL_usm_storage_classes,-SPV_INTEL_optnone,-SPV_KHR_linkonce_odr" "[[OUTPUT3_BC]]"
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-fpga_dep" {{.*}} "-outputs=[[DEPFILE:.+\.d]]" "-unbundle"
// CHK-FPGA: aoc{{.*}} "-o" "[[OUTPUT6:.+\.aocx]]" "[[OUTPUT5]]" "-sycl" "-dep-files=[[DEPFILE]]"
// CHK-FPGA: clang-offload-wrapper{{.*}} "-o=[[OUTPUT7:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_fpga" "-kind=sycl" "[[OUTPUT6]]"
// CHK-FPGA: llc{{.*}} "-filetype=obj" "-o" "[[FINALLINK3:.+\.o]]" "[[OUTPUT7]]"
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
// CHK-FPGA-AOCX-PHASES: 3: file-table-tform, {2}, tempfilelist, (device-sycl)
// CHK-FPGA-AOCX-PHASES: 4: clang-offload-wrapper, {3}, object, (device-sycl)
// CHK-FPGA-AOCX-PHASES: 5: offload, "host-sycl ({{.*}}x86_64{{.*}})" {1}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {4}, image

// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX,CHK-FPGA-AOCX-LIN %s
// RUN:  %clang_cl -fsycl -fintelfpga %t_aocx.a -### 2>&1 \
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
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fno-sycl-device-lib=all %s %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-SRC,CHK-FPGA-AOCX-SRC-LIN %s
// RUN:  %clang_cl -fsycl -fno-sycl-device-lib=all -fintelfpga %s %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-SRC,CHK-FPGA-AOCX-SRC-WIN %s
// CHK-FPGA-AOCX-SRC: clang-offload-bundler{{.*}} "-type=aocx" "-targets=sycl-fpga_aocx-intel-unknown-sycldevice" "-inputs=[[LIBINPUT:.+\.a]]" "-outputs=[[BUNDLEOUT:.+\.aocx]]" "-unbundle"
// CHK-FPGA-AOCX-SRC: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[TABLEOUT:.+\.txt]]" "[[BUNDLEOUT]]"
// CHK-FPGA-AOCX-SRC: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "-batch" "[[TABLEOUT]]"
// CHK-FPGA-AOCX-SRC: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUT:.+\.(o|obj)]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-SRC: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device" {{.*}} "-o" "[[DEVICEBC:.+\.bc]]"
// CHK-FPGA-AOCX-SRC: llvm-link{{.*}} "[[DEVICEBC]]" "-o" "[[LLVMLINKOUT:.+\.bc]]" "--suppress-warnings"
// CHK-FPGA-AOCX-SRC: sycl-post-link
// CHK-FPGA-AOCX-SRC-NOT: -split-esimd
// CHK-FPGA-AOCX-SRC: "-ir-output-only" "-O2" "-spec-const=default" "-o" "[[POSTLINKOUT:.+\.bc]]" "[[LLVMLINKOUT]]
// CHK-FPGA-AOCX-SRC: llvm-spirv{{.*}} "-o" "[[LLVMSPVOUT:.+\.spv]]" {{.*}} "[[POSTLINKOUT]]"
// CHK-FPGA-AOCX-SRC: aoc{{.*}} "-o" "[[AOCOUT:.+\.aocx]]" "[[LLVMSPVOUT]]" "-sycl"
// CHK-FPGA-AOCX-SRC: clang-offload-wrapper{{.*}} "-o=[[WRAPOUTSRC:.+.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "[[AOCOUT]]"
// CHK-FPGA-AOCX-SRC: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUTSRC:.+\.(o|obj)]]" "[[WRAPOUTSRC]]"
// CHK-FPGA-AOCX-SRC: clang{{.*}} "-fsycl-is-host" {{.*}} "-o" "[[HOSTOBJ:.+\.(o|obj)]]"
// CHK-FPGA-AOCX-SRC-LIN: ld{{.*}} "[[HOSTOBJ]]" "[[LIBINPUT]]" "[[LLCOUT]]" "[[LLCOUTSRC]]"
// CHK-FPGA-AOCX-SRC-WIN: link{{.*}} "[[HOSTOBJ]]" "[[LIBINPUT]]" "[[LLCOUT]]" "[[LLCOUTSRC]]"

/// AOCX with object
// RUN: touch %t.o
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga %t.o %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-OBJ,CHK-FPGA-AOCX-OBJ-LIN %s
// RUN:  %clang_cl -fsycl -fno-sycl-device-lib=all -fintelfpga %t.o %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-OBJ,CHK-FPGA-AOCX-OBJ-WIN %s
// CHK-FPGA-AOCX-OBJ: clang-offload-bundler{{.*}} "-type=aocx" "-targets=sycl-fpga_aocx-intel-unknown-sycldevice" "-inputs=[[LIBINPUT:.+\.a]]" "-outputs=[[BUNDLEOUT:.+\.aocx]]" "-unbundle"
// CHK-FPGA-AOCX-OBJ: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[TABLEOUT:.+\.txt]]" "[[BUNDLEOUT]]"
// CHK-FPGA-AOCX-OBJ: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "-batch" "[[TABLEOUT]]"
// CHK-FPGA-AOCX-OBJ: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUT:.+\.(o|obj)]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-OBJ: clang-offload-bundler{{.*}} "-type=o" {{.*}} "-outputs=[[HOSTOBJ:.+\.(o|obj)]],[[DEVICEOBJ:.+\.(o|obj)]]" "-unbundle"
// CHK-FPGA-AOCX-OBJ: llvm-link{{.*}} "[[DEVICEOBJ]]" "-o" "[[LLVMLINKOUT:.+\.bc]]" "--suppress-warnings"
// CHK-FPGA-AOCX-OBJ: sycl-post-link
// CHK-FPGA-AOCX-OBJ-NOT: -split-esimd
// CHK-FPGA-AOCX-OBJ: "-ir-output-only" "-O2" "-spec-const=default" "-o" "[[POSTLINKOUT:.+\.bc]]" "[[LLVMLINKOUT]]
// CHK-FPGA-AOCX-OBJ: llvm-spirv{{.*}} "-o" "[[LLVMSPVOUT:.+\.spv]]" {{.*}} "[[POSTLINKOUT]]"
// CHK-FPGA-AOCX-OBJ: aoc{{.*}} "-o" "[[AOCOUT:.+\.aocx]]" "[[LLVMSPVOUT]]" "-sycl"
// CHK-FPGA-AOCX-OBJ: clang-offload-wrapper{{.*}} "-o=[[WRAPOUTSRC:.+.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "[[AOCOUT]]"
// CHK-FPGA-AOCX-OBJ: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUTSRC:.+\.(o|obj)]]" "[[WRAPOUTSRC]]"
// CHK-FPGA-AOCX-OBJ-LIN: ld{{.*}} "[[HOSTOBJ]]" "[[LIBINPUT]]" "[[LLCOUT]]" "[[LLCOUTSRC]]"
// CHK-FPGA-AOCX-OBJ-WIN: link{{.*}} "[[HOSTOBJ]]" "[[LIBINPUT]]" "[[LLCOUT]]" "[[LLCOUTSRC]]"

/// -fintelfpga -fsycl-link from source
// RUN: touch %t.cpp
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga -fsycl-link=early %t.cpp -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-SRC %s
// RUN: %clang_cl --target=x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga -fsycl-link=early %t.cpp -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-SRC %s
// CHK-FPGA-LINK-SRC: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHK-FPGA-LINK-SRC: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-FPGA-LINK-SRC: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-FPGA-LINK-SRC: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-FPGA-LINK-SRC: 4: compiler, {3}, ir, (device-sycl)
// CHK-FPGA-LINK-SRC: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {4}, c++-cpp-output
// CHK-FPGA-LINK-SRC: 6: compiler, {5}, ir, (host-sycl)
// CHK-FPGA-LINK-SRC: 7: backend, {6}, assembler, (host-sycl)
// CHK-FPGA-LINK-SRC: 8: assembler, {7}, object, (host-sycl)
// CHK-FPGA-LINK-SRC: 9: clang-offload-wrapper, {8}, ir, (host-sycl)
// CHK-FPGA-LINK-SRC: 10: backend, {9}, assembler, (host-sycl)
// CHK-FPGA-LINK-SRC: 11: assembler, {10}, object, (host-sycl)
// CHK-FPGA-LINK-SRC: 12: linker, {11}, archive, (host-sycl)
// CHK-FPGA-LINK-SRC: 13: linker, {4}, ir, (device-sycl)
// CHK-FPGA-LINK-SRC: 14: sycl-post-link, {13}, ir, (device-sycl)
// CHK-FPGA-LINK-SRC: 15: llvm-spirv, {14}, spirv, (device-sycl)
// CHK-FPGA-LINK-SRC: 16: backend-compiler, {15}, fpga_aocr, (device-sycl)
// CHK-FPGA-LINK-SRC: 17: clang-offload-wrapper, {16}, object, (device-sycl)
// CHK-FPGA-LINK-SRC: 18: offload, "host-sycl (x86_64-unknown-linux-gnu)" {12}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {17}, archive

/// -fintelfpga with -reuse-exe=
// RUN:  touch %t.cpp
// RUN:  %clangxx -### -reuse-exe=testing -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %t.cpp 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-REUSE-EXE %s
// RUN:  %clang_cl -### -reuse-exe=testing -fsycl -fintelfpga %t.cpp 2>&1 \
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
// CHK-FPGA-DEP-FILES: clang{{.*}} "-dependency-file" "[[INPUT1:.+\.d]]" "-MT" "{{.*}}.o"
// CHK-FPGA-DEP-FILES: clang{{.*}} "-dependency-file" "[[INPUT2:.+\.d]]" "-MT" "{{.*}}.o"
// CHK-FPGA-DEP-FILES: aoc{{.*}} "-dep-files={{.*}}[[INPUT1]],{{.*}}[[INPUT2]]"
// CHK-FPGA-DEP-FILES-NOT: clang{{.*}} "-fsycl-is-host"{{.*}} "-dependency-file"

/// -fintelfpga dependency file check with host .d enabled
// RUN: %clangxx -### -MMD -fsycl -fintelfpga %t-1.cpp %t-2.cpp 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES-HOST %s
// CHK-FPGA-DEP-FILES-HOST: clang{{.*}} "-dependency-file" "[[INPUT1:.+\.d]]" "-MT" "{{.*}}.o"
// CHK-FPGA-DEP-FILES-HOST: clang{{.*}} "-dependency-file" "[[INPUT2:.+\.d]]" "-MT" "{{.*}}.o"
// CHK-FPGA-DEP-FILES-HOST: aoc{{.*}} "-dep-files={{.*}}[[INPUT1]],{{.*}}[[INPUT2]]"
// CHK-FPGA-DEP-FILES-HOST: clang{{.*}} "-fsycl-is-host"{{.*}} "-dependency-file"
// CHK-FPGA-DEP-FILES-HOST: clang{{.*}} "-fsycl-is-host"{{.*}} "-dependency-file"

/// -fintelfpga dependency file generation test to object
// RUN: %clangxx -### -fsycl -fintelfpga -target x86_64-unknown-linux-gnu %t-1.cpp %t-2.cpp -c 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-DEP-FILES2,CHK-FPGA-DEP-FILES2-LIN %s
// RUN: %clang_cl -### -fsycl -fintelfpga %t-1.cpp %t-2.cpp -c 2>&1 \
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
// RUN: %clang_cl -### -fsycl -fintelfpga %t-1.cpp -c -Fodummy.obj 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-DEP-FILES3,CHK-FPGA-DEP-FILES3-WIN %s
// CHK-FPGA-DEP-FILES3: clang{{.*}} "-dependency-file" "[[OUTPUT:.+\.d]]"
// CHK-FPGA-DEP-FILES3-LIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice,host-x86_64-unknown-linux-gnu,sycl-fpga_dep" {{.*}} "-inputs={{.*}}.bc,{{.*}}.o,[[OUTPUT]]"
// CHK-FPGA-DEP-FILES3-WIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice,host-x86_64-pc-windows-msvc,sycl-fpga_dep" {{.*}} "-inputs={{.*}}.bc,{{.*}}.obj,[[OUTPUT]]"

/// -fintelfpga dependency obj use test
// RUN: touch %t-1.o
// RUN: touch %t-2.o
// RUN: %clangxx -### -fsycl -fintelfpga -target x86_64-unknown-linux-gnu %t-1.o %t-2.o 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES-OBJ %s
// RUN: %clang_cl -### -fsycl -fintelfpga %t-1.o %t-2.o 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES-OBJ %s
// CHK-FPGA-DEP-FILES-OBJ: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-fpga_dep" "-inputs={{.*}}-1.o" "-outputs=[[DEPFILE1:.+\.d]]" "-unbundle"
// CHK-FPGA-DEP-FILES-OBJ: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-fpga_dep" "-inputs={{.*}}-2.o" "-outputs=[[DEPFILE2:.+\.d]]" "-unbundle"
// CHK-FPGA-DEP-FILES-OBJ: aoc{{.*}} "-dep-files=[[DEPFILE1]],[[DEPFILE2]]

/// -fintelfpga dependency file use from object phases test
// RUN: touch %t-1.o
// RUN: %clangxx -fsycl -fno-sycl-device-lib=all -fintelfpga -ccc-print-phases -### %t-1.o 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES-OBJ-PHASES %s
// RUN: %clang_cl -fsycl -fno-sycl-device-lib=all -fintelfpga -ccc-print-phases -### %t-1.o 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES-OBJ-PHASES %s
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 0: input, "{{.*}}-1.o", object, (host-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 1: clang-offload-unbundler, {0}, object, (host-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 2: linker, {1}, image, (host-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 3: linker, {1}, ir, (device-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 4: sycl-post-link, {3}, ir, (device-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 5: llvm-spirv, {4}, spirv, (device-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 6: clang-offload-unbundler, {0}, fpga_dependencies
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 7: backend-compiler, {5, 6}, fpga_aocx, (device-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 8: clang-offload-wrapper, {7}, object, (device-sycl)
// CHK-FPGA-DEP-FILES-OBJ-PHASES: 9: offload, "host-sycl (x86_64-{{unknown-linux-gnu|pc-windows-msvc}})" {2}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {8}, image

/// -fintelfpga output report file test
// RUN: mkdir -p %t_dir
// RUN: %clangxx -### -fsycl -fintelfpga %s -o %t_dir/file.out 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga %s -o %t_dir/file.out 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// RUN: %clang_cl -### -fsycl -fintelfpga %s -Fe%t_dir/file.out 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT %s
// CHK-FPGA-REPORT-OPT: aoc{{.*}} "-sycl" {{.*}} "-output-report-folder={{.*}}{{(/|\\\\)}}file.prj"

/// -fintelfpga output report file from dir/source
/// check dependency file from dir/source
// RUN: mkdir -p %t_dir
// RUN: touch %t_dir/dummy.cpp
// RUN: %clangxx -### -fsycl -fintelfpga %t_dir/dummy.cpp  2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT2 %s
// RUN: %clang_cl -### -fsycl -fintelfpga %t_dir/dummy.cpp 2>&1 \
// RUN:  | FileCheck -DOUTDIR=%t_dir -check-prefix=CHK-FPGA-REPORT-OPT2 %s
// CHK-FPGA-REPORT-OPT2: aoc{{.*}} "-sycl"{{.*}} "-dep-files={{.+}}dummy-{{.+}}.d" "-output-report-folder={{.*}}dummy.prj"
// CHK-FPGA-REPORT-OPT2-NOT: aoc{{.*}} "-sycl" {{.*}}_dir{{.*}}

/// -fintelfpga dependency files from multiple source
// RUN: touch dummy2.cpp
// RUN: %clangxx -### -fsycl -fintelfpga %t_dir/dummy.cpp dummy2.cpp 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-MULTI-DEPS %s
// RUN: %clang_cl -### -fsycl -fintelfpga %t_dir/dummy.cpp dummy2.cpp 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-MULTI-DEPS %s
// CHK-FPGA-MULTI-DEPS: aoc{{.*}} "-sycl"{{.*}} "-dep-files={{.+}}dummy-{{.+}}.d,{{.+}}dummy2-{{.+}}.d" "-output-report-folder={{.*}}dummy.prj"

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
// CHK-FPGA-REPORT-NAME: aoc{{.*}} "-sycl"{{.*}} "-output-report-folder={{.*}}dummy2.prj"

/// Check for implied options (-g -O0)
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -g -O0 -Xs "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS %s
// RUN:   %clang_cl -### -fsycl -fintelfpga -Zi -Od -Xs "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS %s
// CHK-TOOLS-IMPLIED-OPTS: aoc{{.*}} "-g" "-cl-opt-disable" "-DFOO1" "-DFOO2"
