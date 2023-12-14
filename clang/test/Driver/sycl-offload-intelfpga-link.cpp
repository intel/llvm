/// -fintelfpga -fsycl-link tests
// RUN:  touch %t.o
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga -fsycl-link -Xshardware %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-EARLY %s
// RUN:  %clangxx -### -O2 -target x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga -fsycl-link=early -Xshardware %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-EARLY %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga  -fsycl-link=image -Xshardware %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-IMAGE %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64_fpga-unknown-unknown -fsycl-link=image -Xshardware %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK,CHK-FPGA-IMAGE %s
// CHK-FPGA-LINK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown" "-input=[[INPUT:.+\.o]]" "-output=[[OUTPUT1:.+\.o]]" "-unbundle"
// CHK-FPGA-LINK-NOT: clang-offload-bundler{{.*}}
// CHK-FPGA-LINK: spirv-to-ir-wrapper{{.*}} "[[OUTPUT1]]" "-o" "[[IROUTPUT1:.+\.bc]]"
// CHK-FPGA-LINK: llvm-link{{.*}} "[[IROUTPUT1]]" "-o" "[[OUTPUT2_1:.+\.bc]]"
// CHK-FPGA-LINK: sycl-post-link{{.*}} "-O2" "-spec-const=emulation" "-device-globals" "-o" "[[OUTPUT2:.+\.table]]" "[[OUTPUT2_1]]"
// CHK-FPGA-LINK: file-table-tform{{.*}} "-o" "[[TABLEOUT:.+\.txt]]" "[[OUTPUT2]]"
// CHK-FPGA-LINK: llvm-spirv{{.*}} "-o" "[[OUTPUT3:.+\.txt]]" "-spirv-max-version={{.*}}"{{.*}} "[[TABLEOUT]]"
// CHK-FPGA-EARLY: aoc{{.*}} "-o" "[[OUTPUT4:.+\.aocr]]" "[[OUTPUT3]]" "-sycl" "-rtl"
// CHK-FPGA-IMAGE: aoc{{.*}} "-o" "[[OUTPUT5:.+\.aocx]]" "[[OUTPUT3]]" "-sycl"
// CHK-FPGA-LINK: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" {{.*}} "-kind=sycl"
// CHK-FPGA-LINK: llc{{.*}} "-o" "[[OBJOUTDEV:.+\.o]]" "[[WRAPOUT]]"
// CHK-FPGA-EARLY: clang-offload-wrapper{{.*}} "-host" "x86_64-unknown-linux-gnu" "-o" "[[WRAPOUTHOST:.+\.bc]]" "-kind=host"
// CHK-FPGA-EARLY-NOT: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-O2"
// CHK-FPGA-EARLY: "-o" "[[OBJOUT:.+\.o]]" {{.*}} "[[WRAPOUTHOST]]"
// CHK-FPGA-EARLY: llvm-ar{{.*}} "cqL" "libfoo.a" "[[OBJOUT]]" "[[OBJOUTDEV]]"
// CHK-FPGA-IMAGE: llvm-ar{{.*}} "cqL" "libfoo.a" "[[INPUT]]" "[[OBJOUTDEV]]"

// Output designation should not be used for unbundling step
// RUN:  touch %t.o
// RUN:  touch %t.obj
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fintelfpga -fsycl-link %t.o -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK-FPGA-LINK-OUT %s
// RUN:  %clang_cl -### -fintelfpga -fsycl-link %t.obj -Folibfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-OUT %s
// RUN:  %clang_cl -### -fintelfpga -fsycl-link %t.obj -o libfoo.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-OUT %s
// CHK-FPGA-LINK-OUT-NOT: clang-offload-bundler{{.*}} "-output=libfoo.a" "-unbundle"

/// -fintelfpga -fsycl-link clang-cl specific
// RUN:  touch %t.obj
// RUN:  %clang_cl -### -fintelfpga -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-link -Xshardware %t.obj -Folibfoo.lib 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-WIN %s
// RUN:  %clang_cl -### -fintelfpga -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-link -Xshardware %t.obj -o libfoo.lib 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-WIN %s
// CHK-FPGA-LINK-WIN: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_fpga-unknown-unknown{{.*}}" "-input=[[INPUT:.+\.obj]]" "-output=[[OUTPUT1:.+\.obj]]" "-unbundle"
// CHK-FPGA-LINK-WIN-NOT: clang-offload-bundler{{.*}}
// CHK-FPGA-LINK-WIN: spirv-to-ir-wrapper{{.*}} "[[OUTPUT1]]" "-o" "[[IROUTPUT1:.+\.bc]]"
// CHK-FPGA-LINK-WIN: llvm-link{{.*}} "[[IROUTPUT1]]" "-o" "[[OUTPUT2_1:.+\.bc]]"
// CHK-FPGA-LINK-WIN: sycl-post-link{{.*}} "-O2" "-spec-const=emulation" "-device-globals" "-o" "[[OUTPUT2:.+\.table]]" "[[OUTPUT2_1]]"
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
// RUN:  clang-offload-wrapper -o %t-aocr.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=fpga_aocr-intel-unknown %t.aocr
// RUN:  llc -filetype=obj -o %t-aocr.o %t-aocr.bc
// RUN:  llvm-ar crv %t-aocr.a %t.o %t-aocr.o
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fintelfpga -fsycl-link=image -Xshardware %t-aocr.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-LIB,CHK-FPGA-LINK-LIB-IMAGE %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fintelfpga -fsycl-link=early -Xshardware %t-aocr.a 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-LIB,CHK-FPGA-LINK-LIB-EARLY %s
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocx-intel-unknown" "-input={{.*}}" "-check-section"
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocr-intel-unknown" "-input={{.*}}" "-check-section"
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=aocr" "-targets=sycl-fpga_aocr-intel-unknown" "-input=[[INPUT:.+\.a]]" "-output=[[OUTPUT2:.+\.aocr]]" "-unbundle"
// CHK-FPGA-LINK-LIB-IMAGE: llvm-foreach{{.*}} "--out-ext=aocx" "--in-file-list=[[OUTPUT2]]" "--in-replace=[[OUTPUT2]]" "--out-file-list=[[OUTPUT3:.+\.aocx]]" "--out-replace=[[OUTPUT3]]" "--out-increment=a.prj" "--" "{{.*}}aoc{{.*}}" "-o" "[[OUTPUT3]]" "[[OUTPUT2]]" "-sycl" "-output-report-folder=a.prj" "-g"
// CHK-FPGA-LINK-LIB-IMAGE: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[OUTPUT4:.+\.txt]]" "[[OUTPUT3]]"
// CHK-FPGA-LINK-LIB-IMAGE: clang-offload-bundler{{.*}} "-type=aoo" "-targets=host-fpga_aocr-intel-unknown" "-input=[[INPUT]]" "-output=[[OUTPUT_BUNDLE_BC:.+\.txt]]" "-unbundle"
// CHK-FPGA-LINK-LIB-IMAGE: file-table-tform{{.*}} "-rename=0,SymAndProps" "-o" "[[OUTPUT_BC2:.+\.txt]]" "[[OUTPUT_BUNDLE_BC]]"
// CHK-FPGA-LINK-LIB-IMAGE: clang-offload-wrapper{{.*}} "-o=[[WRAPPED_SYM_PROP:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "--emit-reg-funcs=0" "-target=fpga_aocx-intel-unknown" "-kind=sycl" "--sym-prop-bc-files=[[OUTPUT_BC2]]" "-batch" "[[OUTPUT4]]"
// CHK-FPGA-LINK-LIB-IMAGE: llc{{.*}} "-filetype=obj"{{.*}} "[[WRAPPED_SYM_PROP]]"
// CHK-FPGA-LINK-LIB-IMAGE: clang-offload-wrapper{{.*}} "-o=[[WRAPPED_SYM_PROP2:.+\.o]]" "-host=x86_64-unknown-linux-gnu" "--emit-reg-funcs=0" "-target=fpga_aocx-intel-unknown" "-kind=sycl" "--sym-prop-bc-files=[[OUTPUT_BC2]]" "-batch" "[[OUTPUT4]]"
// CHK-FPGA-LINK-LIB-IMAGE: clang-offload-wrapper{{.*}} "-o=[[WRAPWRAP_SYM_PROP:.+\.bc]]" "-host=x86_64-unknown-linux-gnu"{{.*}} "-target=fpga_aocx-intel-unknown" "-kind=host" "[[WRAPPED_SYM_PROP2]]"
// CHK-FPGA-LINK-LIB-IMAGE: llc{{.*}} "-filetype=obj"{{.*}} "[[WRAPWRAP_SYM_PROP]]"
// CHK-FPGA-LINK-LIB-EARLY: llvm-foreach{{.*}} "--out-ext=aocr" "--in-file-list=[[OUTPUT2]]" "--in-replace=[[OUTPUT2]]" "--out-file-list=[[OUTPUT3:.+\.aocr]]" "--out-replace=[[OUTPUT3]]" "--out-increment=a.prj" "--" "{{.*}}aoc{{.*}}" "-o" "[[OUTPUT3]]" "[[OUTPUT2]]" "-sycl" "-rtl" "-output-report-folder=a.prj" "-g"
// CHK-FPGA-LINK-LIB-EARLY: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[OUTPUT4:.+\.txt]]" "[[OUTPUT3]]"
// CHK-FPGA-LINK-LIB-EARLY: clang-offload-bundler{{.*}} "-type=aoo" "-targets=host-fpga_aocr-intel-unknown" "-input=[[INPUT]]" "-output=[[OUTPUT_BUNDLE_BC:.+\.txt]]" "-unbundle"
// CHK-FPGA-LINK-LIB-EARLY: file-table-tform{{.*}} "-rename=0,SymAndProps" "-o" "[[OUTPUT_BC2:.+\.txt]]" "[[OUTPUT_BUNDLE_BC]]"
// CHK-FPGA-LINK-LIB-EARLY: clang-offload-wrapper{{.*}} "-o=[[WRAPPED_SYM_PROP:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=fpga_aocr-intel-unknown" "-kind=sycl" "--sym-prop-bc-files=[[OUTPUT_BC2]]" "-batch" "[[OUTPUT4]]"
// CHK-FPGA-LINK-LIB-EARLY: llc{{.*}} "-filetype=obj"{{.*}} "[[WRAPPED_SYM_PROP]]"
// CHK-FPGA-LINK-LIB-EARLY: clang-offload-wrapper{{.*}} "-o=[[WRAPPED_SYM_PROP2:.+\.o]]" "-host=x86_64-unknown-linux-gnu" "-target=fpga_aocr-intel-unknown" "-kind=sycl" "--sym-prop-bc-files=[[OUTPUT_BC2]]" "-batch" "[[OUTPUT4]]"
// CHK-FPGA-LINK-LIB-EARLY: clang-offload-wrapper{{.*}} "-o=[[WRAPWRAP_SYM_PROP:.+\.bc]]" "-host=x86_64-unknown-linux-gnu"{{.*}} "-target=fpga_aocr-intel-unknown" "-kind=host" "[[WRAPPED_SYM_PROP2]]"
// CHK-FPGA-LINK-LIB-EARLY: llc{{.*}} "-filetype=obj"{{.*}} "[[WRAPWRAP_SYM_PROP]]"
// CHK-FPGA-LINK-LIB: clang-offload-bundler{{.*}} "-type=aoo" "-targets=host-x86_64-unknown-linux-gnu" "-input=[[INPUT]]" "-output=[[OUTPUT1:.+\.txt]]" "-unbundle"
// CHK-FPGA-LINK-LIB-EARLY: clang-offload-wrapper{{.*}} "-host" "x86_64-unknown-linux-gnu" "-o" "[[WRAPPED_AOCR_LIST_BC:.+\.bc]]" "-kind=host" "-target=x86_64-unknown-linux-gnu" "[[OUTPUT1]]"
// CHK-FPGA-LINK-LIB-EARLY: clang{{.*}} "-o" "[[OUTPUT_O:.+\.o]]" "-x" "ir" "[[WRAPPED_AOCR_LIST_BC]]"
// CHK-FPGA-LINK-LIB-EARLY: llvm-ar{{.*}} "cqL" {{.*}} "[[OUTPUT_O]]"
// CHK-FPGA-LINK-LIB-IMAGE: llvm-ar{{.*}} "cqL" {{.*}} "@[[OUTPUT1]]"

/// Check the warning's emission for -fsycl-link's appending behavior
// RUN: touch dummy.a
// RUN: %clangxx -fintelfpga -fsycl-link=image -target x86_64-unknown-linux-gnu %s -o dummy.a -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHK-FPGA-LINK-WARN
// RUN: %clangxx -fintelfpga -fsycl-link=early -target x86_64-unknown-linux-gnu %s -o dummy.a -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHK-FPGA-LINK-WARN
// CHK-FPGA-LINK-WARN: warning: appending to an existing archive 'dummy.a'

/// -fintelfpga -fsycl-link name creation without output file specified
// RUN: mkdir -p %t_dir
// RUN: touch %t_dir/dummy_file.cpp
// RUN: %clangxx -### -fintelfpga -fsycl-link -target x86_64-unknown-linux-gnu %t_dir/dummy_file.cpp 2>&1 \
// RUN: | FileCheck -check-prefixes=CHK-SYCL-LINK-LIN -DINPUTSRC=a %s
// RUN: %clang_cl -### -fintelfpga -fsycl-link %t_dir/dummy_file.cpp 2>&1 \
// RUN: | FileCheck -check-prefixes=CHK-SYCL-LINK-WIN -DINPUTSRC=a %s
// CHK-SYCL-LINK-LIN: llvm-ar{{.*}} "cqL" "[[INPUTSRC]].a"
// CHK-SYCL-LINK-WIN: lib.exe{{.*}} "-OUT:[[INPUTSRC]].a"

/// Check the warning's emission for conflicting emulation/hardware (AOCR)
// RUN: touch %t-aocr.a
// RUN: %clangxx -fintelfpga -fsycl-link=image -target x86_64-unknown-linux-gnu %t-aocr.a %s -### 2>&1 \
// RUN:  | FileCheck %s --check-prefix=CHK-FPGA-LINK-WARN-AOCR
// CHK-FPGA-LINK-WARN-AOCR: warning: FPGA archive '{{.*}}-aocr.a' does not contain matching emulation/hardware expectancy

/// Check deps behaviors with input fat archive and creating aocx archive
// RUN: %clangxx -fintelfpga -fsycl-link=image \
// RUN:          -target x86_64-unknown-linux-gnu %S/Inputs/SYCL/liblin64.a \
// RUN:          %s -### 2>&1 \
// RUN:  | FileCheck %s --check-prefix=CHK-FPGA-LINK-UNDEFS
// CHK-FPGA-LINK-UNDEFS: ld{{.*}} "--unresolved-symbols=ignore-all"
// CHK-FPGA-LINK-UNDEFS: clang-offload-deps{{.*}}

/// -fintelfpga -fsycl-link from source
// RUN: touch %t.cpp
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga -fsycl-link=early %t.cpp -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-SRC %s
// RUN: %clang_cl --target=x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga -fsycl-link=early %t.cpp -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-LINK-SRC %s
// CHK-FPGA-LINK-SRC: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHK-FPGA-LINK-SRC: 1: append-footer, {0}, c++, (host-sycl)
// CHK-FPGA-LINK-SRC: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-FPGA-LINK-SRC: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHK-FPGA-LINK-SRC: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-FPGA-LINK-SRC: 5: compiler, {4}, ir, (device-sycl)
// CHK-FPGA-LINK-SRC: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64_fpga-unknown-unknown)" {5}, c++-cpp-output
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
// CHK-FPGA-LINK-SRC: 21: clang-offload-wrapper, {19}, object, (device-sycl)
// CHK-FPGA-LINK-SRC: 22: clang-offload-wrapper, {21}, object, (device-sycl)
// CHK-FPGA-LINK-SRC: 23: offload, "host-sycl (x86_64-unknown-linux-gnu)" {13}, "device-sycl (spir64_fpga-unknown-unknown)" {20}, "device-sycl (spir64_fpga-unknown-unknown)" {22}, archive

/// -fintelfpga with AOCR library and additional object
// RUN:  touch %t2.o
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga -Xshardware %t-aocr.a %t2.o 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA %s

// CHK-FPGA: clang-offload-bundler{{.*}} "-type=aocr" "-targets=sycl-fpga_aocr-intel-unknown" "-input=[[INPUT:.+\.a]]" "-output=[[OUTPUT2:.+\.aocr]]" "-unbundle"
// CHK-FPGA: llvm-foreach{{.*}} "--out-ext=aocx" "--in-file-list=[[OUTPUT2]]" "--in-replace=[[OUTPUT2]]" "--out-file-list=[[OUTPUT3:.+\.aocx]]" "--out-replace=[[OUTPUT3]]" "--out-increment=a.prj" "--" "{{.*}}aoc{{.*}}" "-o" "[[OUTPUT3]]" "[[OUTPUT2]]" "-sycl" "-output-report-folder=a.prj" "-g"
// CHK-FPGA: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[OUTPUT4:.+\.txt]]" "[[OUTPUT3]]"
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=aoo" "-targets=host-fpga_aocr-intel-unknown" "-input=[[INPUT]]" "-output=[[OUTPUT_BUNDLE_BC:.+\.txt]]" "-unbundle"
// CHK-FPGA: file-table-tform{{.*}} "-rename=0,SymAndProps" "-o" "[[OUTPUT_BC2:.+\.txt]]" "[[OUTPUT_BUNDLE_BC]]"
// CHK-FPGA: clang-offload-wrapper{{.*}} "-o=[[WRAPPED_SYM_PROP:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_fpga" "-kind=sycl" "--sym-prop-bc-files=[[OUTPUT_BC2]]" "-batch" "[[OUTPUT4]]"
// CHK-FPGA: llc{{.*}} "-filetype=obj" "-o" "[[FINALLINK:.+\.o]]" "[[WRAPPED_SYM_PROP]]"
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=o" "-targets=host-x86_64-unknown-linux-gnu,sycl-spir64_fpga-unknown-unknown" {{.*}} "-output=[[FINALLINK2:.+\.o]]" "-output=[[OUTPUT1:.+\.o]]" "-unbundle"
// CHK-FPGA: spirv-to-ir-wrapper{{.*}} "[[OUTPUT1]]" "-o" "[[IROUTPUT1:.+\.bc]]"
// CHK-FPGA: llvm-link{{.*}} "[[IROUTPUT1]]" "-o" "[[OUTPUT2_BC:.+\.bc]]"
// CHK-FPGA: sycl-post-link{{.*}} "-O2" "-spec-const=emulation" "-device-globals" "-o" "[[OUTPUT3_TABLE:.+\.table]]" "[[OUTPUT2_BC]]"
// CHK-FPGA: file-table-tform{{.*}} "-o" "[[TABLEOUT:.+\.txt]]" "[[OUTPUT3_TABLE]]"
// CHK-FPGA: llvm-spirv{{.*}} "-o" "[[OUTPUT5:.+\.txt]]" "-spirv-max-version={{.*}}"{{.*}} "[[TABLEOUT]]"
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-fpga_dep" {{.*}} "-output=[[DEPFILE:.+\.d]]" "-unbundle"
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=aoo" "-targets=sycl-fpga_dep" {{.*}} "-output=[[DEPFILE_AOCR:.+\.txt]]" "-unbundle"
// CHK-FPGA: aoc{{.*}} "-o" "[[OUTPUT6:.+\.aocx]]" "[[OUTPUT5]]" "-sycl" "-dep-files=[[DEPFILE]],@[[DEPFILE_AOCR]]"
// CHK-FPGA: file-table-tform{{.*}} "-replace=Code,Code" "-o" "[[TABLEOUT2:.+\.table]]" "[[OUTPUT3_TABLE]]" "[[OUTPUT6]]"
// CHK-FPGA: clang-offload-wrapper{{.*}} "-o=[[OUTPUT7:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_fpga" "-kind=sycl" "-batch" "[[TABLEOUT2]]"
// CHK-FPGA: llc{{.*}} "-filetype=obj" "-o" "[[FINALLINK3:.+\.o]]" "[[OUTPUT7]]"
// CHK-FPGA: clang-offload-bundler{{.*}} "-type=aoo" "-targets=host-x86_64-unknown-linux-gnu" {{.*}} "-output=[[FINALLINK4:.+\.txt]]" "-unbundle"
// CHK-FPGA: {{link|ld}}{{.*}} "@[[FINALLINK4]]" "[[FINALLINK2]]" "[[FINALLINK]]" "[[FINALLINK3]]"

/// -fintelfpga with AOCX library
// Create the dummy archive
// RUN:  echo "Dummy AOCX image" > %t.aocx
// RUN:  echo "void foo() {}" > %t.c
// RUN:  %clang -c -o %t.o %t.c
// RUN:  clang-offload-wrapper -o %t-aocx.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=fpga_aocx-intel-unknown %t.aocx
// RUN:  llc -filetype=obj -o %t-aocx.o %t-aocx.bc
// RUN:  llvm-ar crv %t_aocx.a %t.o %t-aocx.o
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -Xshardware -fintelfpga %t_aocx.a -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-PHASES %s
// RUN:  %clang_cl --target=x86_64-pc-windows-msvc -Xshardware -fintelfpga %t_aocx.a -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-PHASES %s
// CHK-FPGA-AOCX-PHASES: 0: input, "{{.*}}", fpga_aocx, (host-sycl)
// CHK-FPGA-AOCX-PHASES: 1: linker, {0}, image, (host-sycl)
// CHK-FPGA-AOCX-PHASES: 2: clang-offload-unbundler, {0}, fpga_aocx
// CHK-FPGA-AOCX-PHASES: 3: file-table-tform, {2}, tempfilelist, (device-sycl)
// CHK-FPGA-AOCX-PHASES: 4: input, "{{.*}}", fpga_aocx
// CHK-FPGA-AOCX-PHASES: 5: clang-offload-unbundler, {4}, tempfilelist
// CHK-FPGA-AOCX-PHASES: 6: file-table-tform, {5}, tempfilelist, (device-sycl)
// CHK-FPGA-AOCX-PHASES: 7: clang-offload-wrapper, {3, 6}, object, (device-sycl)
// CHK-FPGA-AOCX-PHASES: 8: offload, "host-sycl ({{.*}}x86_64{{.*}})" {1}, "device-sycl (spir64_fpga-unknown-unknown)" {7}, image

// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fintelfpga -Xshardware %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX,CHK-FPGA-AOCX-LIN %s
// RUN:  %clang_cl -fintelfpga -Xshardware %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX,CHK-FPGA-AOCX-WIN %s
// CHK-FPGA-AOCX: clang-offload-bundler{{.*}} "-type=aocx" "-targets=sycl-fpga_aocx-intel-unknown" "-input=[[LIBINPUT:.+\.a]]" "-output=[[BUNDLEOUT:.+\.aocx]]" "-unbundle"
// CHK-FPGA-AOCX: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[TABLEOUT:.+\.txt]]" "[[BUNDLEOUT]]"
// CHK-FPGA-AOCX: clang-offload-bundler{{.*}} "-type=aoo" "-targets=host-fpga_aocx-intel-unknown" "-input=[[LIBINPUT]]" "-output=[[BUNDLEBCOUT:.+\.txt]]" "-unbundle"
// CHK-FPGA-AOCX: file-table-tform{{.*}} "-rename=0,SymAndProps" "-o" "[[SYM_AND_PROP:.+\.txt]]" "[[BUNDLEBCOUT]]"
// CHK-FPGA-AOCX: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "--sym-prop-bc-files=[[SYM_AND_PROP]]" "-batch" "[[TABLEOUT]]"
// CHK-FPGA-AOCX-LIN: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUT:.+\.o]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-WIN: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUT2:.+\.obj]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-NOT: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocx-intel-unknown"
// CHK-FPGA-AOCX-LIN: ld{{.*}} "[[LIBINPUT]]" "[[LLCOUT]]"
// CHK-FPGA-AOCX-WIN: link{{.*}} "[[LIBINPUT]]" "[[LLCOUT2]]"

/// AOCX with source
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fintelfpga -Xshardware -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-SRC,CHK-FPGA-AOCX-SRC-LIN %s
// RUN:  %clang_cl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -Xshardware -fintelfpga %s %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-SRC,CHK-FPGA-AOCX-SRC-WIN %s
// CHK-FPGA-AOCX-SRC: clang-offload-bundler{{.*}} "-type=aocx" "-targets=sycl-fpga_aocx-intel-unknown" "-input=[[LIBINPUT:.+\.a]]" "-output=[[BUNDLEOUT:.+\.aocx]]" "-unbundle"
// CHK-FPGA-AOCX-SRC: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[TABLEOUT:.+\.txt]]" "[[BUNDLEOUT]]"
// CHK-FPGA-AOCX-SRC: clang-offload-bundler{{.*}} "-type=aoo" "-targets=host-fpga_aocx-intel-unknown" "-input=[[LIBINPUT]]" "-output=[[BUNDLEBCOUT:.+\.txt]]" "-unbundle"
// CHK-FPGA-AOCX-SRC: file-table-tform{{.*}} "-rename=0,SymAndProps" "-o" "[[SYM_AND_PROP:.+\.txt]]" "[[BUNDLEBCOUT]]"
// CHK-FPGA-AOCX-SRC: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "--sym-prop-bc-files=[[SYM_AND_PROP]]" "-batch" "[[TABLEOUT]]"
// CHK-FPGA-AOCX-SRC: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUT:.+\.(o|obj)]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-SRC: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device" {{.*}} "-o" "[[DEVICEBC:.+\.bc]]"
// CHK-FPGA-AOCX-SRC: llvm-link{{.*}} "[[DEVICEBC]]" "-o" "[[LLVMLINKOUT:.+\.bc]]" "--suppress-warnings"
// CHK-FPGA-AOCX-SRC: sycl-post-link{{.*}} "-O2" "-spec-const=emulation" "-device-globals" "-o" "[[POSTLINKOUT:.+\.table]]" "[[LLVMLINKOUT]]
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
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga -Xshardware %t.o %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-OBJ,CHK-FPGA-AOCX-OBJ-LIN %s
// RUN:  %clang_cl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga -Xshardware %t.o %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-OBJ,CHK-FPGA-AOCX-OBJ-WIN %s
// CHK-FPGA-AOCX-OBJ: clang-offload-bundler{{.*}} "-type=aocx" "-targets=sycl-fpga_aocx-intel-unknown" "-input=[[LIBINPUT:.+\.a]]" "-output=[[BUNDLEOUT:.+\.aocx]]" "-unbundle"
// CHK-FPGA-AOCX-OBJ: file-table-tform{{.*}} "-rename=0,Code" "-o" "[[TABLEOUT:.+\.txt]]" "[[BUNDLEOUT]]"
// CHK-FPGA-AOCX-OBJ: clang-offload-bundler{{.*}} "-type=aoo" "-targets=host-fpga_aocx-intel-unknown" "-input=[[LIBINPUT]]" "-output=[[BUNDLEBCOUT:.+\.txt]]" "-unbundle"
// CHK-FPGA-AOCX-OBJ: file-table-tform{{.*}} "-rename=0,SymAndProps" "-o" "[[SYM_AND_PROP:.+\.txt]]" "[[BUNDLEBCOUT]]"
// CHK-FPGA-AOCX-OBJ: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "--sym-prop-bc-files=[[SYM_AND_PROP]]" "-batch" "[[TABLEOUT]]"
// CHK-FPGA-AOCX-OBJ: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUT:.+\.(o|obj)]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-OBJ: clang-offload-bundler{{.*}} "-type=o" {{.*}} "-output=[[HOSTOBJ:.+\.(o|obj)]]" "-output=[[DEVICEOBJ:.+\.(o|obj)]]" "-unbundle"
// CHK-FPGA-AOCX-OBJ: spirv-to-ir-wrapper{{.*}} "[[DEVICEOBJ]]" "-o" "[[IROUTPUT:.+\.bc]]"
// CHK-FPGA-AOCX-OBJ: llvm-link{{.*}} "[[IROUTPUT]]" "-o" "[[LLVMLINKOUT:.+\.bc]]" "--suppress-warnings"
// CHK-FPGA-AOCX-OBJ: sycl-post-link{{.*}} "-O2" "-spec-const=emulation" "-device-globals" "-o" "[[POSTLINKOUT:.+\.table]]" "[[LLVMLINKOUT]]
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
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64,spir64_fpga -Xshardware %t.o %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-OBJ2 %s
// RUN:  %clang_cl -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64,spir64_fpga -Xshardware %t.o %t_aocx.a -### 2>&1 \
// RUN:  | FileCheck -check-prefixes=CHK-FPGA-AOCX-OBJ2 %s
// CHK-FPGA-AOCX-OBJ2: clang-offload-bundler{{.*}} "-type=o" {{.*}} "-output=[[HOSTOBJ:.+\.(o|obj)]]" "-output=[[DEVICEOBJ:.+\.(o|obj)]]" "-output=[[DEVICEOBJ2:.+\.(o|obj)]]" "-unbundle"
// CHK-FPGA-AOCX-OBJ2: spirv-to-ir-wrapper{{.*}} "[[DEVICEOBJ]]" "-o" "[[IROUTPUT:.+\.bc]]"
// CHK-FPGA-AOCX-OBJ2: llvm-link{{.*}} "[[IROUTPUT]]" "-o" "[[LLVMLINKOUT:.+\.bc]]" "--suppress-warnings"
// CHK-FPGA-AOCX-OBJ2: sycl-post-link{{.*}} "-O2" "-spec-const=native" "-device-globals" "-o" "[[POSTLINKOUT:.+\.table]]" "[[LLVMLINKOUT]]"
// CHK-FPGA-AOCX-OBJ2: file-table-tform{{.*}} "-o" "[[TABLEOUT:.+\.txt]]" "[[POSTLINKOUT]]"
// CHK-FPGA-AOCX-OBJ2: llvm-spirv{{.*}} "-o" "[[LLVMSPVOUT:.+\.txt]]" {{.*}} "[[TABLEOUT]]"
// CHK-FPGA-AOCX-OBJ2: file-table-tform{{.*}} "-replace=Code,Code" "-o" "[[TFORM_OUT:.+\.table]]" "[[POSTLINKOUT]]" "[[LLVMSPVOUT]]"
// CHK-FPGA-AOCX-OBJ2: clang-offload-wrapper{{.*}} "-o=[[WRAPOUT:.+\.bc]]" {{.*}} "-target=spir64" "-kind=sycl" "-batch" "[[TFORM_OUT]]"
// CHK-FPGA-AOCX-OBJ2: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUT:.+\.(o|obj)]]" "[[WRAPOUT]]"
// CHK-FPGA-AOCX-OBJ2: spirv-to-ir-wrapper{{.*}} "[[DEVICEOBJ2]]" "-o" "[[IROUTPUT2:.+\.bc]]"
// CHK-FPGA-AOCX-OBJ2: llvm-link{{.*}} "[[IROUTPUT2]]" "-o" "[[LLVMLINKOUT2:.+\.bc]]" "--suppress-warnings"
// CHK-FPGA-AOCX-OBJ2: sycl-post-link{{.*}} "-O2" "-spec-const=emulation" "-device-globals" "-o" "[[POSTLINKOUT2:.+\.table]]" "[[LLVMLINKOUT2]]"
// CHK-FPGA-AOCX-OBJ2: file-table-tform{{.*}} "-o" "[[TABLEOUT2:.+\.txt]]" "[[POSTLINKOUT2]]"
// CHK-FPGA-AOCX-OBJ2: llvm-spirv{{.*}} "-o" "[[LLVMSPVOUT2:.+\.txt]]" {{.*}} "[[TABLEOUT2]]"
// CHK-FPGA-AOCX-OBJ2: aoc{{.*}} "-o" "[[AOCOUT:.+\.aocx]]" "[[LLVMSPVOUT2]]" "-sycl"
// CHK-FPGA-AOCX-OBJ2: file-table-tform{{.*}} "-replace=Code,Code" "-o" "[[TABLEOUT3:.+\.table]]" "[[POSTLINKOUT2]]" "[[AOCOUT]]"
// CHK-FPGA-AOCX-OBJ2: clang-offload-wrapper{{.*}} "-o=[[WRAPOUTSRC:.+.bc]]" {{.*}} "-target=spir64_fpga" "-kind=sycl" "-batch" "[[TABLEOUT3]]"
// CHK-FPGA-AOCX-OBJ2: llc{{.*}} "-filetype=obj" "-o" "[[LLCOUTSRC:.+\.(o|obj)]]" "[[WRAPOUTSRC]]"
// CHK-FPGA-AOCX-OBJ2: {{(ld|link).*}} "[[HOSTOBJ]]"{{.*}} "[[LLCOUT]]" "[[LLCOUTSRC]]"
