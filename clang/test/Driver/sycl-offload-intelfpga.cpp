///
/// tests specific to -fintelfpga -fsycl
///
// REQUIRES: clang-driver

/// -fintelfpga implies -g and -MMD
// RUN:   %clang++ -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-INTELFPGA %s
// CHK-TOOLS-INTELFPGA: clang{{.*}} "-dependency-file"
// CHK-TOOLS-INTELFPGA: clang{{.*}} "-debug-info-kind=limited"

/// -fintelfpga -fsycl-link tests
// RUN:  touch %t.o
// RUN:  %clang++ -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link %t.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-EARLY %s
// RUN:  %clang++ -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=early %t.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-EARLY %s
// RUN:  %clang++ -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=image %t.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-IMAGE %s
// CHK-FPGA-LINK: clang-offload-bundler{{.*}} "-type=o" "-targets=fpga-fpga_aocx-intel-{{.*}}-sycldevice" "-inputs=[[INPUT:.+\.o]]" "-check-section"
// CHK-FPGA-LINK: clang-offload-bundler{{.*}} "-type=o" "-targets=fpga-fpga_aocr-intel-{{.*}}-sycldevice" "-inputs=[[INPUT]]" "-check-section"
// CHK-FPGA-LINK: clang-offload-bundler{{.*}} "-type=o" "-targets=host-x86_64-unknown-linux-gnu,sycl-spir64_fpga-unknown-{{.*}}-sycldevice" "-inputs=[[INPUT]]" "-outputs=[[OUTPUT1:.+\.o]],[[OUTPUT2:.+\.o]]" "-unbundle"
// CHK-FPGA-LINK: llvm-link{{.*}} "[[OUTPUT2]]" "-o" "[[OUTPUT3:.+\.bc]]"
// CHK-FPGA-LINK: llvm-spirv{{.*}} "-spirv-max-version=1.1" "-spirv-ext=+all" "-o" "[[OUTPUT4:.+\.spv]]" "[[OUTPUT3]]"
// CHK-FPGA-EARLY: aoc{{.*}} "-o" "[[OUTPUT5:.+\.aocr]]" "[[OUTPUT4]]" "-sycl" "-rtl"
// CHK-FPGA-IMAGE: aoc{{.*}} "-o" "[[OUTPUT5:.+\.aocx]]" "[[OUTPUT4]]" "-sycl"
// CHK-FPGA-LINK: ld{{.*}} "-r" "[[INPUT]]" "-o" "[[OUTPUT6:.+\.o]]"
// CHK-FPGA-EARLY: clang-offload-bundler{{.*}} "-type=o" "-targets=fpga-fpga_aocr-intel-{{.*}}-sycldevice,host-x86_64-unknown-linux-gnu" "-outputs={{.*}}" "-inputs=[[OUTPUT5]],[[OUTPUT6]]"
// CHK-FPGA-IMAGE: clang-offload-bundler{{.*}} "-type=o" "-targets=fpga-fpga_aocx-intel-{{.*}}-sycldevice,host-x86_64-unknown-linux-gnu" "-outputs={{.*}}" "-inputs=[[OUTPUT5]],[[OUTPUT6]]"

/// Check Phases with -fintelfpga -fsycl-link
// RUN:  touch %t.o
// RUN:  %clang++ -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=image %t.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-PHASES,CHK-FPGA-LINK-PHASES-IMAGE %s
// RUN:  %clang++ -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -fsycl-link=early %t.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-PHASES,CHK-FPGA-LINK-PHASES-EARLY %s
// CHK-FPGA-LINK-PHASES: 0: input, "[[INPUT:.+\.o]]", object
// CHK-FPGA-LINK-PHASES: 1: clang-offload-unbundler, {0}, object
// CHK-FPGA-LINK-PHASES: 2: linker, {1}, spirv, (device-sycl)
// CHK-FPGA-LINK-PHASES-IMAGE: 3: backend-compiler, {2}, fpga-aocx, (device-sycl)
// CHK-FPGA-LINK-PHASES-EARLY: 3: backend-compiler, {2}, fpga-aocr, (device-sycl)
// CHK-FPGA-LINK-PHASES: 4: input, "[[INPUT]]", object, (device-sycl)
// CHK-FPGA-LINK-PHASES: 5: linker, {4}, object, (device-sycl)
// CHK-FPGA-LINK-PHASES: 6: clang-offload-bundler, {3, 5}, object, (device-sycl)
// CHK-FPGA-LINK-PHASES: 7: offload, "device-sycl (spir64_fpga-unknown-{{.*}}-sycldevice)" {6}, object

// -fintelfpga -reuse-exe tests
// RUN: %clang++ -### -fsycl -fintelfpga %s -reuse-exe=does_not_exist 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-REUSE-EXE %s
// CHK-FPGA-REUSE-EXE: warning: -reuse-exe file 'does_not_exist' not found; ignored
//

// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows-msvc
