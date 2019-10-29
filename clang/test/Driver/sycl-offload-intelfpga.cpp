///
/// tests specific to -fintelfpga -fsycl
///
// REQUIRES: clang-driver

/// -fintelfpga implies -g and -MMD
// RUN:   %clang++ -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-INTELFPGA %s
// CHK-TOOLS-INTELFPGA: clang{{.*}} "-debug-info-kind=limited" {{.*}} "-dependency-file"
// CHK-TOOLS-INTELFPGA: aoc{{.*}} "-dep-files={{.*}}"

/// -fintelfpga implies -g but -g0 should override
// RUN:   %clang++ -### -target x86_64-unknown-linux-gnu -g0 -fsycl -fintelfpga %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-INTELFPGA-G0 %s
// CHK-TOOLS-INTELFPGA-G0-NOT: clang{{.*}} "-debug-info-kind=limited"

/// -fintelfpga -fsycl-link tests
// RUN:  touch %t.o
// RUN:  %clang++ -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link %t.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-EARLY %s
// RUN:  %clang++ -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=early %t.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-EARLY %s
// RUN:  %clang++ -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=image %t.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-IMAGE %s
// CHK-FPGA-LINK-NOT: clang-offload-bundler{{.*}} "-check-section"
// CHK-FPGA-LINK: clang-offload-bundler{{.*}} "-type=o" "-targets=host-x86_64-unknown-linux-gnu,sycl-spir64_fpga-unknown-{{.*}}-sycldevice" "-inputs=[[INPUT:.+\.o]]" "-outputs=[[OUTPUT1:.+\.o]],[[OUTPUT2:.+\.o]]" "-unbundle"
// CHK-FPGA-LINK: llvm-link{{.*}} "[[OUTPUT2]]" "-o" "[[OUTPUT3:.+\.bc]]"
// CHK-FPGA-LINK: llvm-spirv{{.*}} "-spirv-max-version=1.1" "-spirv-ext=+all" "-o" "[[OUTPUT4:.+\.spv]]" "[[OUTPUT3]]"
// CHK-FPGA-EARLY: aoc{{.*}} "-o" "[[OUTPUT5:.+\.aocr]]" "[[OUTPUT4]]" "-sycl" "-rtl"
// CHK-FPGA-IMAGE: aoc{{.*}} "-o" "[[OUTPUT5:.+\.aocx]]" "[[OUTPUT4]]" "-sycl"
// CHK-FPGA-LINK: {{lib|llvm-ar}}{{.*}}

/// -fintelfpga -fsycl-link clang-cl specific
// RUN:  touch %t.obj
// RUN:  %clang_cl -### -fsycl -fintelfpga -fsycl-link %t.obj 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-WIN %s
// CHK-FPGA-LINK-WIN: clang-offload-bundler{{.*}} "-type=o" "-targets=host-x86_64-pc-windows-msvc,sycl-spir64_fpga-unknown-{{.*}}-sycldevice{{.*}}" "-inputs=[[INPUT:.+\.obj]]" "-outputs=[[OUTPUT1:.+\.obj]],[[OUTPUT2:.+\.obj]]" "-unbundle"
// CHK-FPGA-LINK-WIN: llvm-link{{.*}} "[[OUTPUT2]]" "-o" "[[OUTPUT3:.+\.bc]]"
// CHK-FPGA-LINK-WIN: llvm-spirv{{.*}} "-spirv-max-version=1.1" "-spirv-ext=+all" "-o" "[[OUTPUT4:.+\.spv]]" "[[OUTPUT3]]"
// CHK-FPGA-LINK-WIN: aoc{{.*}} "-o" "[[OUTPUT5:.+\.aocr]]" "[[OUTPUT4]]" "-sycl" "-rtl"
// CHK-FPGA-LINK-WIN: lib.exe{{.*}}


/// Check -fintelfpga -fsycl-link with an FPGA archive
// Create the dummy archive
// RUN:  echo "Dummy AOCR image" > %t.aocr
// RUN:  echo "void foo() {}" > %t.c
// RUN:  %clang -c %t.c
// RUN:  clang-offload-wrapper -o %t-aocr.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=fpga_aocr-intel-linux-sycldevice %t.aocr
// RUN:  llc -filetype=obj -o %t-aocr.o %t-aocr.bc
// RUN:  llvm-ar crv %t.a %t.o %t-aocr.o
// RUN:  %clang++ -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=image %t.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-LIB,CHK-FPGA-LINK-LIB-IMAGE %s
// RUN:  %clang++ -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=early %t.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-LIB,CHK-FPGA-LINK-LIB-EARLY %s
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocx-intel-{{linux|windows}}-sycldevice" "-inputs=[[INPUT:.+\.a]]" "-check-section"
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocr-intel-{{linux|windows}}-sycldevice" "-inputs=[[INPUT]]" "-check-section"
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocr-intel-{{linux|windows}}-sycldevice" "-inputs=[[INPUT]]" "-outputs=[[OUTPUT2:.+\.aocr]]" "-unbundle"
// CHK-FPGA-LINK-LIB-IMAGE: aoc{{.*}} "-o" "[[OUTPUT3:.+\.aocx]]" "[[OUTPUT2]]" "-sycl"
// CHK-FPGA-LINK-LIB-EARLY: aoc{{.*}} "-o" "[[OUTPUT4:.+\.aocr]]" "[[OUTPUT2]]" "-sycl" "-rtl"
// CHK-FPGA-LINK-LIB-IMAGE: clang-offload-wrapper{{.*}} "-host=x86_64-unknown-linux-gnu" "-target=fpga_aocx-intel-{{linux|windows}}-sycldevice" "-kind=sycl" "[[OUTPUT3]]"
// CHK-FPGA-LINK-LIB-EARLY: clang-offload-wrapper{{.*}} "-host=x86_64-unknown-linux-gnu" "-target=fpga_aocr-intel-{{linux|windows}}-sycldevice" "-kind=sycl" "[[OUTPUT4]]"
// CHK-FPGA-LINK-LIB: llc{{.*}} "-filetype=obj" "-o" "[[OUTPUT5:.+\.o]]"
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=aoo" "-targets=host-x86_64-unknown-linux-gnu" "-inputs=[[INPUT]]" "-outputs=[[OUTPUT1:.+\.txt]]" "-unbundle"
// CHK-FPGA-LINK-LIB: llvm-ar{{.*}} "cr" {{.*}} "@[[OUTPUT1]]"

/// -fintelfpga with AOCR library and additional object
// RUN:  touch %t2.o
// RUN:  %clang++ -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %t.a %t2.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA %s
// CHK-FPGA: aoc{{.*}} "-o" {{.*}} "-sycl"
// CHK-FPGA: llc{{.*}} "-filetype=obj" "-o" "[[FINALLINK:.*\.o]]"
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=o" "-targets=host-x86_64-unknown-linux-gnu,sycl-spir64_fpga-unknown-{{linux|windows}}-sycldevice" {{.*}} "-outputs=[[FINALLINK2:.+\.o]],[[OUTPUT1:.+\.o]]" "-unbundle"
// CHK-FPGA: llvm-no-spir-kernel{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT3:.+\.o]]"
// CHK-FPGA: llvm-link{{.*}} "[[OUTPUT3]]" "-o" "[[OUTPUT4:.+\.bc]]"
// CHK-FPGA: llvm-spirv{{.*}} "-o" "[[OUTPUT5:.+\.spv]]" "[[OUTPUT4]]"
// CHK-FPGA: clang-offload-wrapper{{.*}} "-o=[[OUTPUT6:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_fpga" "-kind=sycl" "[[OUTPUT5]]"
// CHK-FPGA: llc{{.*}} "-filetype=obj" "-o" "[[FINALLINK3:.+\.o]]" "[[OUTPUT6]]"
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=aoo" "-targets=host-x86_64-unknown-linux-gnu" {{.*}} "-outputs=[[FINALLINK4:.+\.txt]]" "-unbundle"
// CHK-FPGA: {{link|ld}}{{.*}} "@[[FINALLINK4]]" "[[FINALLINK2]]" "[[FINALLINK]]" "[[FINALLINK3]]"

/// -fintelfpga -fsycl-link from source
// RUN: touch %t.cpp
// RUN: %clang++ -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=early %t.cpp -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-SRC %s
// CHK-FPGA-LINK-SRC: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHK-FPGA-LINK-SRC: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-FPGA-LINK-SRC: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-FPGA-LINK-SRC: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-FPGA-LINK-SRC: 4: compiler, {3}, sycl-header, (device-sycl)
// CHK-FPGA-LINK-SRC: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_fpga-unknown-{{linux|windows}}-sycldevice)" {4}, c++-cpp-output
// CHK-FPGA-LINK-SRC: 6: compiler, {5}, ir, (host-sycl)
// CHK-FPGA-LINK-SRC: 7: backend, {6}, assembler, (host-sycl)
// CHK-FPGA-LINK-SRC: 8: assembler, {7}, object, (host-sycl)
// CHK-FPGA-LINK-SRC: 9: linker, {8}, archive, (host-sycl)
// CHK-FPGA-LINK-SRC: 10: compiler, {3}, ir, (device-sycl)
// CHK-FPGA-LINK-SRC: 11: backend, {10}, assembler, (device-sycl)
// CHK-FPGA-LINK-SRC: 12: assembler, {11}, object, (device-sycl)
// CHK-FPGA-LINK-SRC: 13: linker, {12}, spirv, (device-sycl)
// CHK-FPGA-LINK-SRC: 14: backend-compiler, {13}, fpga-aocr, (device-sycl)
// CHK-FPGA-LINK-SRC: 15: clang-offload-wrapper, {14}, object, (device-sycl)
// CHK-FPGA-LINK-SRC: 16: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64_fpga-unknown-{{linux|windows}}-sycldevice)" {15}, archive

/// -fintelfpga with -reuse-exe=
// RUN:  touch %t.cpp
// RUN:  %clang++ -### -reuse-exe=testing -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %t.cpp 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-REUSE-EXE %s
// CHK-FPGA-REUSE-EXE: aoc{{.*}} "-o" {{.*}} "-sycl" {{.*}} "-reuse-exe=testing"

/// -fintelfpga dependency file generation test
// RUN: touch %t-1.cpp
// RUN: touch %t-2.cpp
// RUN: %clang++ -### -fsycl -fintelfpga %t-1.cpp %t-2.cpp -o %t.out 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES %s
// RUN: %clang++ -### -fsycl -fintelfpga %t-1.cpp %t-2.cpp 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-DEP-FILES %s
// CHK-FPGA-DEP-FILES: clang{{.*}} "-dependency-file" "[[INPUT1:.+\.d]]"
// CHK-FPGA-DEP-FILES: clang{{.*}} "-dependency-file" "[[INPUT2:.+\.d]]"
// CHK-FPGA-DEP-FILES: aoc{{.*}} "-dep-files={{.*}}[[INPUT1]],{{.*}}[[INPUT2]]"

// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows-msvc
