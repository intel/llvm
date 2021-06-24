///
/// tests specific to -fintelfpga -fsycl for emulation
///
// REQUIRES: clang-driver

/// -fintelfpga -fsycl-link tests
// RUN:  touch %t.o
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga -fsycl-link %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-EARLY %s
// RUN:  %clangxx -### -O2 -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga -fsycl-link=early %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-EARLY %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga -fsycl-link=image %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-IMAGE %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice -g -fsycl-link=image %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-IMAGE %s
// CHK-FPGA-LINK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice" "-inputs=[[INPUT:.+\.o]]" "-outputs=[[OUTPUT1:.+\.o]]" "-unbundle"
// CHK-FPGA-LINK: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2_1:.+\.bc]]"
// CHK-FPGA-LINK: sycl-post-link{{.*}} "-O2" "-spec-const=default" "-o" "[[OUTPUT2:.+\.table]]" "[[OUTPUT2_1]]"
// CHK-FPGA-LINK: file-table-tform{{.*}} "-o" "[[TABLEOUT:.+\.txt]]" "[[OUTPUT2]]"
// CHK-FPGA-LINK: llvm-spirv{{.*}} "-o" "[[OUTPUT3:.+\.txt]]" "-spirv-max-version={{.*}}"{{.*}} "[[TABLEOUT]]"
// CHK-FPGA-EARLY: opencl-aot{{.*}} "-device=fpga_fast_emu" "-spv=[[OUTPUT3]]" "-ir=[[OUTPUT4:.+\.aocr]]" "--bo=-g"
// CHK-FPGA-IMAGE: opencl-aot{{.*}} "-device=fpga_fast_emu" "-spv=[[OUTPUT3]]" "-ir=[[OUTPUT4:.+\.aocx]]" "--bo=-g"
// CHK-FPGA-LINK: file-table-tform{{.*}} "-replace=Code,Code" "-o" "[[TABLEOUT2:.+\.table]]" "[[OUTPUT2]]" "[[OUTPUT4]]"
// CHK-FPGA-LINK: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" {{.*}} "-kind=sycl" "-batch" "[[TABLEOUT2]]"
// CHK-FPGA-LINK: llc{{.*}} "-o" "[[OBJOUTDEV:.+\.o]]" "[[WRAPOUT]]"
// CHK-FPGA-EARLY: clang-offload-wrapper{{.*}} "-host" "x86_64-unknown-linux-gnu" "-o" "[[WRAPOUTHOST:.+\.bc]]" "-kind=host"
// CHK-FPGA-EARLY-NOT: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-O2"
// CHK-FPGA-EARLY: "-o" "[[OBJOUT:.+\.o]]" {{.*}} "[[WRAPOUTHOST]]"
// CHK-FPGA-EARLY: llvm-ar{{.*}} "cr" "libfoo.a" "[[OBJOUT]]" "[[OBJOUTDEV]]"
// CHK-FPGA-IMAGE: llvm-ar{{.*}} "cr" "libfoo.a" "[[INPUT]]" "[[OBJOUTDEV]]"

/// -fintelfpga -fsycl-link clang-cl specific
// RUN:  touch %t.obj
// RUN:  %clang_cl -### -fsycl -fintelfpga -fno-sycl-device-lib=all -fsycl-link %t.obj -Folibfoo.lib 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-WIN %s
// RUN:  %clang_cl -### -fsycl -fintelfpga -fno-sycl-device-lib=all -fsycl-link %t.obj -o libfoo.lib 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-WIN %s
// CHK-FPGA-LINK-WIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown-sycldevice{{.*}}" "-inputs=[[INPUT:.+\.obj]]" "-outputs=[[OUTPUT1:.+\.obj]]" "-unbundle"
// CHK-FPGA-LINK-WIN: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2_1:.+\.bc]]"
// CHK-FPGA-LINK-WIN: sycl-post-link{{.*}} "-O2" "-spec-const=default" "-o" "[[OUTPUT2:.+\.table]]" "[[OUTPUT2_1]]"
// CHK-FPGA-LINK-WIN: file-table-tform{{.*}} "-o" "[[TABLEOUT:.+\.txt]]" "[[OUTPUT2]]"
// CHK-FPGA-LINK-WIN: llvm-spirv{{.*}} "-o" "[[OUTPUT3:.+\.txt]]" "-spirv-max-version={{.*}}"{{.*}} "[[TABLEOUT]]"
// CHK-FPGA-LINK-WIN: opencl-aot{{.*}} "-device=fpga_fast_emu" "-spv=[[OUTPUT3]]" "-ir=[[OUTPUT4:.+\.aocr]]" "--bo=-g"
// CHK-FPGA-LINK-WIN: file-table-tform{{.*}} "-replace=Code,Code" "-o" "[[TABLEOUT2:.+\.table]]" "[[OUTPUT2]]" "[[OUTPUT4]]"
// CHK-FPGA-LINK-WIN: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-kind=sycl" "-batch" "[[TABLEOUT2]]"
// CHK-FPGA-LINK-WIN: llc{{.*}} "-o" "[[OBJOUTDEV:.+\.obj]]" "[[WRAPOUT]]"
// CHK-FPGA-LINK-WIN: clang-offload-wrapper{{.*}} "-o" "[[WRAPOUTHOST:.+\.bc]]" "-kind=host"
// CHK-FPGA-LINK-WIN: clang{{.*}} "-o" "[[OBJOUT:.+\.obj]]" {{.*}} "[[WRAPOUTHOST]]"
// CHK-FPGA-LINK-WIN: lib.exe{{.*}} "[[OBJOUT]]" "[[OBJOUTDEV]]" {{.*}} "-OUT:libfoo.lib"

/// Check -fintelfpga -fsycl-link with an FPGA archive
// Create the dummy archive
// RUN:  echo "Dummy AOCR image" > %t.aocr
// RUN:  echo "void foo() {}" > %t.c
// RUN:  %clang -c -o %t.o %t.c
// RUN:  clang-offload-wrapper -o %t-aocr.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=fpga_aocr_emu-intel-unknown-sycldevice %t.aocr
// RUN:  llc -filetype=obj -o %t-aocr.o %t-aocr.bc
// RUN:  llvm-ar crv %t-aocr.a %t.o %t-aocr.o
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=image %t-aocr.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-LIB,CHK-FPGA-LINK-LIB-IMAGE %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=early %t-aocr.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-LIB,CHK-FPGA-LINK-LIB-EARLY %s
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocr_emu-intel-unknown-sycldevice" "-inputs={{.*}}" "-check-section"
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=aocr" "-targets=sycl-fpga_aocr_emu-intel-unknown-sycldevice" "-inputs=[[INPUT:.+\.a]]" "-outputs=[[OUTPUT2:.+\.aocr]]" "-unbundle"
// CHK-FPGA-LINK-LIB-IMAGE: llvm-foreach{{.*}} "--out-ext=aocx" "--in-file-list=[[OUTPUT2]]" "--in-replace=[[OUTPUT2]]" "--out-file-list=[[OUTPUT3:.+\.aocx]]" "--out-replace=[[OUTPUT3]]" "--" "{{.*}}opencl-aot{{.*}} "-device=fpga_fast_emu" "-spv=[[OUTPUT2]]" "-ir=[[OUTPUT3]]" "--bo=-g"
// CHK-FPGA-LINK-LIB-IMAGE: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[OUTPUT4:.+\.txt]]" "[[OUTPUT3]]"
// CHK-FPGA-LINK-LIB-IMAGE: clang-offload-wrapper{{.*}} "-host=x86_64-unknown-linux-gnu" "--emit-reg-funcs=0" "-target=fpga_aocx_emu-intel-unknown-sycldevice" "-kind=sycl" "-batch" "[[OUTPUT4]]"
// CHK-FPGA-LINK-LIB-EARLY: llvm-foreach{{.*}} "--out-ext=aocr" "--in-file-list=[[OUTPUT2]]" "--in-replace=[[OUTPUT2]]" "--out-file-list=[[OUTPUT3:.+\.aocr]]" "--out-replace=[[OUTPUT3]]" "--" "{{.*}}opencl-aot{{.*}} "-device=fpga_fast_emu" "-spv=[[OUTPUT2]]" "-ir=[[OUTPUT3]]" "--bo=-g"
// CHK-FPGA-LINK-LIB-EARLY: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[OUTPUT4:.+\.txt]]" "[[OUTPUT3]]"
// CHK-FPGA-LINK-LIB-EARLY: clang-offload-wrapper{{.*}} "-host=x86_64-unknown-linux-gnu" "-target=fpga_aocr_emu-intel-unknown-sycldevice" "-kind=sycl" "-batch" "[[OUTPUT4]]"
// CHK-FPGA-LINK-LIB: llc{{.*}} "-filetype=obj" "-o" "[[OUTPUT5:.+\.o]]"
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=aoo" "-targets=host-x86_64-unknown-linux-gnu" "-inputs=[[INPUT]]" "-outputs=[[OUTPUT1:.+\.txt]]" "-unbundle"
// CHK-FPGA-LINK-LIB-IMAGE: llvm-ar{{.*}} "cr" {{.*}} "@[[OUTPUT1]]"

/// Check the warning's emission for conflicting emulation/hardware
// RUN: touch %t-aocr.a
// RUN: %clangxx -fsycl -fintelfpga -fsycl-link=image -target x86_64-unknown-linux-gnu %t-aocr.a %s -Xshardware -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHK-FPGA-LINK-WARN
// RUN: %clangxx -fsycl -fintelfpga -fsycl-link=early -target x86_64-unknown-linux-gnu %t-aocr.a %s -Xshardware -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHK-FPGA-LINK-WARN
// CHK-FPGA-LINK-WARN: warning: FPGA archive '{{.*}}-aocr.a' does not contain matching emulation/hardware expectancy

/// -fintelfpga with AOCR library and additional object
// RUN:  touch %t2.o
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fno-sycl-device-lib=all -fsycl -fintelfpga %t-aocr.a %t2.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA %s
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=aocr" "-targets=sycl-fpga_aocr_emu-intel-unknown-sycldevice" "-inputs=[[INPUT:.+\.a]]" "-outputs=[[OUTPUT2:.+\.aocr]]" "-unbundle"
// CHK-FPGA: llvm-foreach{{.*}} "--out-ext=aocx" "--in-file-list=[[OUTPUT2]]" "--in-replace=[[OUTPUT2]]" "--out-file-list=[[OUTPUT3:.+\.aocx]]" "--out-replace=[[OUTPUT3]]" "--" "{{.*}}opencl-aot{{.*}} "-device=fpga_fast_emu" "-spv=[[OUTPUT2]]" "-ir=[[OUTPUT3]]" "--bo=-g"
// CHK-FPGA: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[OUTPUT4:.+\.txt]]" "[[OUTPUT3]]"
// CHK-FPGA: clang-offload-wrapper{{.*}} "-o=[[OUTPUT_AOCX_BC:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_fpga" "-kind=sycl" "-batch" "[[OUTPUT4]]"
// CHK-FPGA: llc{{.*}} "-filetype=obj" "-o" "[[FINALLINK:.+\.o]]" "[[OUTPUT_AOCX_BC]]"
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=o" "-targets=host-x86_64-unknown-linux-gnu,sycl-spir64_fpga-unknown-unknown-sycldevice" {{.*}} "-outputs=[[FINALLINK2:.+\.o]],[[OUTPUT1:.+\.o]]" "-unbundle"
// CHK-FPGA: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2_BC:.+\.bc]]"
// CHK-FPGA: sycl-post-link{{.*}} "-O2" "-spec-const=default" "-o" "[[OUTPUT2:.+\.table]]" "[[OUTPUT2_BC]]"
// CHK-FPGA: file-table-tform{{.*}} "-o" "[[TABLEOUT:.+\.txt]]" "[[OUTPUT2]]"
// CHK-FPGA: llvm-spirv{{.*}} "-o" "[[OUTPUT3:.+\.txt]]" "-spirv-max-version={{.*}}"{{.*}} "[[TABLEOUT]]"
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-fpga_dep" {{.*}} "-outputs=[[DEPFILE:.+\.d]]" "-unbundle"
// CHK-FPGA: opencl-aot{{.*}} "-device=fpga_fast_emu" "-spv=[[OUTPUT3]]" "-ir=[[OUTPUT4:.+\.aocx]]" "--bo=-g"
// CHK-FPGA: file-table-tform{{.*}} "-replace=Code,Code" "-o" "[[TABLEOUT2:.+\.table]]" "[[OUTPUT2]]" "[[OUTPUT4]]"
// CHK-FPGA: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-kind=sycl" "-batch" "[[TABLEOUT2]]"
// CHK-FPGA: llc{{.*}} "-filetype=obj" "-o" "[[FINALLINK3:.+\.o]]" "[[WRAPOUT]]"
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=aoo" "-targets=host-x86_64-unknown-linux-gnu" {{.*}} "-outputs=[[FINALLINK4:.+\.txt]]" "-unbundle"
// CHK-FPGA: {{link|ld}}{{.*}} "@[[FINALLINK4]]" "[[FINALLINK2]]" "[[FINALLINK]]" "[[FINALLINK3]]"

/// -fintelfpga with AOCX library
// Create the dummy archive
// RUN:  echo "Dummy AOCX image" > %t.aocx
// RUN:  echo "void foo() {}" > %t.c
// RUN:  %clang -c -o %t.o %t.c
// RUN:  clang-offload-wrapper -o %t-aocx.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=fpga_aocx_emu-intel-unknown-sycldevice %t.aocx
// RUN:  llc -filetype=obj -o %t-aocx.o %t-aocx.bc
// RUN:  llvm-ar crv %t_aocx.a %t.o %t-aocx.o
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %t_aocx.a -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-PHASES %s
// RUN:  %clang_cl -fsycl -fintelfpga %t_aocx.a -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-PHASES %s
// CHK-FPGA-AOCX-PHASES: 0: input, "{{.*}}", fpga_aocx_emu, (host-sycl)
// CHK-FPGA-AOCX-PHASES: 1: linker, {0}, image, (host-sycl)
// CHK-FPGA-AOCX-PHASES: 2: clang-offload-unbundler, {0}, fpga_aocx_emu
// CHK-FPGA-AOCX-PHASES: 3: file-table-tform, {2}, tempfilelist, (device-sycl)
// CHK-FPGA-AOCX-PHASES: 4: clang-offload-wrapper, {3}, object, (device-sycl)
// CHK-FPGA-AOCX-PHASES: 5: offload, "host-sycl ({{.*}}x86_64{{.*}})" {1}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {4}, image

// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX,CHK-FPGA-AOCX-LIN %s
// RUN:  %clang_cl -fsycl -fintelfpga %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX,CHK-FPGA-AOCX-WIN %s
// CHK-FPGA-AOCX: clang-offload-bundler{{.*}} "-type=aocx" "-targets=sycl-fpga_aocx_emu-intel-unknown-sycldevice" "-inputs=[[LIBINPUT:.+\.a]]" "-outputs=[[BUNDLEOUT:.+\.aocx]]" "-unbundle"
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
// CHK-FPGA-AOCX-SRC: clang-offload-bundler{{.*}} "-type=aocx" "-targets=sycl-fpga_aocx_emu-intel-unknown-sycldevice" "-inputs=[[LIBINPUT:.+\.a]]" "-outputs=[[BUNDLEOUT:.+\.aocx]]" "-unbundle"
// CHK-FPGA-AOCX-SRC: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[TABLEOUT:.+\.txt]]" "[[BUNDLEOUT]]"
// CHK-FPGA-AOCX-SRC: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "-batch" "[[TABLEOUT]]"
// CHK-FPGA-AOCX-SRC: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUT:.+\.(o|obj)]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-SRC: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device" {{.*}} "-o" "[[DEVICEBC:.+\.bc]]"
// CHK-FPGA-AOCX-SRC: llvm-link{{.*}} "[[DEVICEBC]]" "-o" "[[LLVMLINKOUT:.+\.bc]]" "--suppress-warnings"
// CHK-FPGA-AOCX-SRC: sycl-post-link{{.*}} "-O2" "-spec-const=default" "-o" "[[OUTPUT2:.+\.table]]" "[[LLVMLINKOUT]]"
// CHK-FPGA-AOCX-SRC: file-table-tform{{.*}} "-o" "[[TABLEOUT:.+\.txt]]" "[[OUTPUT2]]"
// CHK-FPGA-AOCX-SRC: llvm-spirv{{.*}} "-o" "[[OUTPUT3:.+\.txt]]" "-spirv-max-version={{.*}}"{{.*}} "[[TABLEOUT]]"
// CHK-FPGA-AOCX-SRC: opencl-aot{{.*}} "-device=fpga_fast_emu" "-spv=[[OUTPUT3]]" "-ir=[[OUTPUT4:.+\.aocx]]" "--bo=-g"
// CHK-FPGA-AOCX-SRC: file-table-tform{{.*}} "-replace=Code,Code" "-o" "[[TABLEOUT2:.+\.table]]" "[[OUTPUT2]]" "[[OUTPUT4]]"
// CHK-FPGA-AOCX-SRC: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-kind=sycl" "-batch" "[[TABLEOUT2]]"
// CHK-FPGA-AOCX-SRC: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUTSRC:.+\.(o|obj)]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-SRC: clang{{.*}} "-fsycl-is-host" {{.*}} "-o" "[[HOSTOBJ:.+\.(o|obj)]]"
// CHK-FPGA-AOCX-SRC-LIN: ld{{.*}} "[[HOSTOBJ]]" "[[LIBINPUT]]" "[[LLCOUT]]" "[[LLCOUTSRC]]"
// CHK-FPGA-AOCX-SRC-WIN: link{{.*}} "[[HOSTOBJ]]" "[[LIBINPUT]]" "[[LLCOUT]]" "[[LLCOUTSRC]]"

/// AOCX with object
// RUN: touch %t.o
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga %t.o %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-OBJ,CHK-FPGA-AOCX-OBJ-LIN %s
// RUN:  %clang_cl -fsycl -fno-sycl-device-lib=all -fintelfpga %t.o %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-OBJ,CHK-FPGA-AOCX-OBJ-WIN %s
// CHK-FPGA-AOCX-OBJ: clang-offload-bundler{{.*}} "-type=aocx" "-targets=sycl-fpga_aocx_emu-intel-unknown-sycldevice" "-inputs=[[LIBINPUT:.+\.a]]" "-outputs=[[BUNDLEOUT:.+\.aocx]]" "-unbundle"
// CHK-FPGA-AOCX-OBJ: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[TABLEOUT:.+\.txt]]" "[[BUNDLEOUT]]"
// CHK-FPGA-AOCX-OBJ: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "-batch" "[[TABLEOUT]]"
// CHK-FPGA-AOCX-OBJ: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUT:.+\.(o|obj)]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-OBJ: clang-offload-bundler{{.*}} "-type=o" {{.*}} "-outputs=[[HOSTOBJ:.+\.(o|obj)]],[[DEVICEOBJ:.+\.(o|obj)]]" "-unbundle"
// CHK-FPGA-AOCX-OBJ: llvm-link{{.*}} "[[DEVICEOBJ]]" "-o" "[[LLVMLINKOUT:.+\.bc]]" "--suppress-warnings"
// CHK-FPGA-AOCX-OBJ: sycl-post-link{{.*}} "-O2" "-spec-const=default" "-o" "[[OUTPUT2:.+\.table]]" "[[LLVMLINKOUT]]"
// CHK-FPGA-AOCX-OBJ: file-table-tform{{.*}} "-o" "[[TABLEOUT:.+\.txt]]" "[[OUTPUT2]]"
// CHK-FPGA-AOCX-OBJ: llvm-spirv{{.*}} "-o" "[[OUTPUT3:.+\.txt]]" "-spirv-max-version={{.*}}"{{.*}} "[[TABLEOUT]]"
// CHK-FPGA-AOCX-OBJ: opencl-aot{{.*}} "-device=fpga_fast_emu" "-spv=[[OUTPUT3]]" "-ir=[[OUTPUT4:.+\.aocx]]" "--bo=-g"
// CHK-FPGA-AOCX-OBJ: file-table-tform{{.*}} "-replace=Code,Code" "-o" "[[TABLEOUT2:.+\.table]]" "[[OUTPUT2]]" "[[OUTPUT4]]"
// CHK-FPGA-AOCX-OBJ: clang-offload-wrapper{{.*}} "-o=[[WRAPOUTSRC:.+\.bc]]" {{.*}} "-kind=sycl" "-batch" "[[TABLEOUT2]]"
// CHK-FPGA-AOCX-OBJ: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUTSRC:.+\.(o|obj)]]" "[[WRAPOUTSRC]]"
// CHK-FPGA-AOCX-OBJ-LIN: ld{{.*}} "[[HOSTOBJ]]" "[[LIBINPUT]]" "[[LLCOUT]]" "[[LLCOUTSRC]]"
// CHK-FPGA-AOCX-OBJ-WIN: link{{.*}} "[[HOSTOBJ]]" "[[LIBINPUT]]" "[[LLCOUT]]" "[[LLCOUTSRC]]"

/// -fintelfpga -fsycl-link from source
// RUN: touch %t.cpp
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl-use-footer -fsycl -fno-sycl-device-lib=all -fintelfpga -fsycl-link=early %t.cpp -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-SRC %s
// RUN: %clang_cl --target=x86_64-unknown-linux-gnu -fsycl-use-footer -fsycl -fno-sycl-device-lib=all -fintelfpga -fsycl-link=early %t.cpp -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-SRC %s
// CHK-FPGA-LINK-SRC: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHK-FPGA-LINK-SRC: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-FPGA-LINK-SRC: 2: append-footer, {1}, c++, (host-sycl)
// CHK-FPGA-LINK-SRC: 3: preprocessor, {2}, c++-cpp-output, (host-sycl)
// CHK-FPGA-LINK-SRC: 4: input, "[[INPUT]]", c++, (device-sycl)
// CHK-FPGA-LINK-SRC: 5: preprocessor, {4}, c++-cpp-output, (device-sycl)
// CHK-FPGA-LINK-SRC: 6: compiler, {5}, ir, (device-sycl)
// CHK-FPGA-LINK-SRC: 7: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {6}, c++-cpp-output
// CHK-FPGA-LINK-SRC: 8: compiler, {7}, ir, (host-sycl)
// CHK-FPGA-LINK-SRC: 9: backend, {8}, assembler, (host-sycl)
// CHK-FPGA-LINK-SRC: 10: assembler, {9}, object, (host-sycl)
// CHK-FPGA-LINK-SRC: 11: clang-offload-wrapper, {10}, ir, (host-sycl)
// CHK-FPGA-LINK-SRC: 12: backend, {11}, assembler, (host-sycl)
// CHK-FPGA-LINK-SRC: 13: assembler, {12}, object, (host-sycl)
// CHK-FPGA-LINK-SRC: 14: linker, {13}, archive, (host-sycl)
// CHK-FPGA-LINK-SRC: 15: linker, {6}, ir, (device-sycl)
// CHK-FPGA-LINK-SRC: 16: sycl-post-link, {15}, tempfiletable, (device-sycl)
// CHK-FPGA-LINK-SRC: 17: file-table-tform, {16}, tempfilelist, (device-sycl)
// CHK-FPGA-LINK-SRC: 18: llvm-spirv, {17}, tempfilelist, (device-sycl)
// CHK-FPGA-LINK-SRC: 19: backend-compiler, {18}, fpga_aocr_emu, (device-sycl)
// CHK-FPGA-LINK-SRC: 20: file-table-tform, {16, 19}, tempfiletable, (device-sycl)
// CHK-FPGA-LINK-SRC: 21: clang-offload-wrapper, {20}, object, (device-sycl)
// CHK-FPGA-LINK-SRC: 22: offload, "host-sycl (x86_64-unknown-linux-gnu)" {14}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {21}, archive

/// Check the warning's emission for conflicting emulation/hardware (AOCX)
// RUN: touch %t_aocx.a
// RUN: %clangxx -fsycl -fintelfpga -fsycl-link=image -target x86_64-unknown-linux-gnu %t_aocx.a %s -Xshardware -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHK-FPGA-LINK-WARN-AOCX
// CHK-FPGA-LINK-WARN-AOCX: warning: FPGA archive '{{.*}}_aocx.a' does not contain matching emulation/hardware expectancy

/// Check for implied options with emulation (-g -O0)
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -g -O0 -Xs "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS %s
// RUN:   %clang_cl -### -fsycl -fintelfpga -Zi -Od -Xs "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS %s
// CHK-TOOLS-IMPLIED-OPTS: clang{{.*}} "-fsycl-is-device"{{.*}} "-O0"
// CHK-TOOLS-IMPLIED-OPTS: sycl-post-link{{.*}} "-O0"
// CHK-TOOLS-IMPLIED-OPTS: opencl-aot{{.*}} "--bo=-g -cl-opt-disable" "-DFOO1" "-DFOO2"

