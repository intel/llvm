// REQUIRES: x86-registered-target

//
// Check help message.
//

// RUN: clang-offload-wrapper --help | FileCheck %s --check-prefix CHECK-HELP
// CHECK-HELP: OVERVIEW: A tool to create a wrapper bitcode for offload target binaries.
// CHECK-HELP: Takes offload target binaries and optional manifest files as input
// CHECK-HELP: and produces bitcode file containing target binaries packaged as data
// CHECK-HELP: and initialization code which registers target binaries in the offload
// CHECK-HELP: runtime. Manifest files format and contents are not restricted and are
// CHECK-HELP: a subject of agreement between the device compiler and the native
// CHECK-HELP: runtime for that device. When present, manifest file name should
// CHECK-HELP: immediately follow the corresponding device image filename on the
// CHECK-HELP: command line. Options annotating a device binary have effect on all
// CHECK-HELP: subsequent input, until redefined.
// CHECK-HELP: For example:
// CHECK-HELP:   clang-offload-wrapper                   \
// CHECK-HELP:       -host x86_64-pc-linux-gnu           \
// CHECK-HELP:       -kind=sycl                          \
// CHECK-HELP:         -target=spir64                    \
// CHECK-HELP:           -format=spirv                   \
// CHECK-HELP:           -compile-opts=-g                \
// CHECK-HELP:           -link-opts=-cl-denorms-are-zero \
// CHECK-HELP:           -entries=sym.txt                \
// CHECK-HELP:           -properties=props.txt           \
// CHECK-HELP:           a.spv                           \
// CHECK-HELP:           a_mf.txt                        \
// CHECK-HELP:         -target=xxx                       \
// CHECK-HELP:           -format=native                  \
// CHECK-HELP:           -compile-opts=""                \
// CHECK-HELP:           -link-opts=""                   \
// CHECK-HELP:           -entries=""                     \
// CHECK-HELP:           -properties=""                  \
// CHECK-HELP:           b.bin                           \
// CHECK-HELP:           b_mf.txt                        \
// CHECK-HELP:       -kind=openmp                        \
// CHECK-HELP:           c.bin\n
// CHECK-HELP: This command generates an x86 wrapper object (.bc) enclosing the
// CHECK-HELP: following tuples describing a single device binary each:
// CHECK-HELP: |offload|target|data  |data |manifest|compile|entries|properties|...|
// CHECK-HELP: |  kind |      |format|     |        |options|       |          |...|
// CHECK-HELP: |-------|------|------|-----|--------|-------|-------|----------|---|
// CHECK-HELP: |sycl   |spir64|spirv |a.spv|a_mf.txt|  -g   |sym.txt|props.txt |...|
// CHECK-HELP: |sycl   |xxx   |native|b.bin|b_mf.txt|       |       |          |...|
// CHECK-HELP: |openmp |xxx   |native|c.bin|        |       |       |          |...|
// CHECK-HELP: |...|    link            |
// CHECK-HELP: |...|    options         |
// CHECK-HELP: |---|--------------------|
// CHECK-HELP: |...|-cl-denorms-are-zero|
// CHECK-HELP: |...|                    |
// CHECK-HELP: |...|                    |
// CHECK-HELP: USAGE: clang-offload-wrapper [options] <input files>
// CHECK-HELP: OPTIONS:
// CHECK-HELP: Generic Options:
// CHECK-HELP:   --help                  - Display available options (--help-hidden for more)
// CHECK-HELP:   --help-list             - Display list of available options (--help-list-hidden for more)
// CHECK-HELP:   --version               - Display the version of this program
// CHECK-HELP: clang-offload-wrapper options:
// CHECK-HELP:   --batch                 - All input files are treated as a table file.  One table file per target.
// CHECK-HELP:                             Table files consist of a table of filenames that provide
// CHECK-HELP:                             Code, Symbols, Properties, etc.
// CHECK-HELP:                             Example input table file in batch mode:
// CHECK-HELP:                               [Code|Symbols|Properties|Manifest]
// CHECK-HELP:                               a_0.bc|a_0.sym|a_0.props|a_0.mnf
// CHECK-HELP:                               a_1.bin|||
// CHECK-HELP:                             Example usage:
// CHECK-HELP:                               clang-offload-wrapper -batch -host=x86_64-unknown-linux-gnu
// CHECK-HELP:                                 -kind=openmp -target=spir64_gen table1.txt
// CHECK-HELP:                                 -kind=openmp -target=spir64     table2.txt
// CHECK-HELP:   --compile-opts=<string> - compile options passed to the offload runtime
// CHECK-HELP:   --desc-name=<name>      - Specifies offload descriptor symbol name: '.<offload kind>.<name>',
// CHECK-HELP:                             and makes it globally visible
// CHECK-HELP:   --emit-reg-funcs        - Emit [un-]registration functions
// CHECK-HELP:   --entries=<filename>    - File listing all offload function entries, SYCL offload only
// CHECK-HELP:   --format=<value>        - device binary image formats:
// CHECK-HELP:     =none                 -   not set
// CHECK-HELP:     =native               -   unknown or native
// CHECK-HELP:     =spirv                -   SPIRV binary
// CHECK-HELP:     =llvmbc               -   LLVMIR bitcode
// CHECK-HELP:   --host=<triple>         - Target triple for the output module. If omitted, the host
// CHECK-HELP:                             triple is used.
// CHECK-HELP:   --kind=<value>          - offload kind:
// CHECK-HELP:     =unknown              -   unknown
// CHECK-HELP:     =host                 -   host
// CHECK-HELP:     =openmp               -   OpenMP
// CHECK-HELP:     =hip                  -   HIP
// CHECK-HELP:     =sycl                 -   SYCL
// CHECK-HELP:   --link-opts=<string>    - link options passed to the offload runtime
// CHECK-HELP:   -o <filename>           - Output filename
// CHECK-HELP:   --properties=<filename> - File listing device binary image properties, SYCL offload only
// CHECK-HELP:   --target=<string>       - offload target triple
// CHECK-HELP:   -v                      - verbose output

// -------
// Generate files to wrap.
//
// RUN: echo 'Content of device file1' > %t1.tgt
// RUN: echo 'Content of device file2' > %t2.tgt
// RUN: echo 'Content of device file3' > %t3.tgt
// RUN: echo 'Content of manifest file1' > %t1_mf.txt
//
// -------
// Check bitcode produced by the wrapper tool.
//
// RUN: clang-offload-wrapper -add-omp-offload-notes                                  \
// RUN:   -host=x86_64-pc-linux-gnu                                                   \
// RUN:     -kind=openmp -target=tg2                -format=native %t3.tgt %t1_mf.txt \
// RUN:     -kind=sycl   -target=tg1 -compile-opts=-g -link-opts=-cl-denorms-are-zero \
// RUN:                  -format spirv  %t1.tgt                                       \
// RUN:                  -target=tg2 -compile-opts= -link-opts=                       \
// RUN:                  -format native %t2.tgt                                       \
// RUN:   -o %t.wrapper.bc 2>&1 | FileCheck %s --check-prefix ELF-WARNING
// RUN: llvm-dis %t.wrapper.bc -o - | FileCheck %s --check-prefix CHECK-IR

// ELF-WARNING: is not an ELF image, so notes cannot be added to it.
// CHECK-IR: target triple = "x86_64-pc-linux-gnu"

// --- OpenMP device binary image descriptor structure
// CHECK-IR-DAG: [[ENTTY:%.+]] = type { ptr, ptr, i{{32|64}}, i32, i32 }
// CHECK-IR-DAG: [[IMAGETY:%.+]] = type { ptr, ptr, ptr, ptr }
// CHECK-IR-DAG: [[DESCTY:%.+]] = type { i32, ptr, ptr, ptr }

// --- SYCL device binary image descriptor structure
// CHECK-IR-DAG: [[SYCL_IMAGETY:%.+]] = type { i16, i8, i8, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
// CHECK-IR-DAG: [[SYCL_DESCTY:%.+]] = type { i16, i16, ptr, ptr, ptr }

// CHECK-IR: [[ENTBEGIN:@.+]] = external hidden constant [[ENTTY]]
// CHECK-IR: [[ENTEND:@.+]] = external hidden constant [[ENTTY]]

// CHECK-IR: [[DUMMY:@.+]] = hidden constant [0 x [[ENTTY]]] zeroinitializer, section "omp_offloading_entries"

// CHECK-IR: [[OMP_BIN:@.+]] = internal unnamed_addr constant [[OMP_BINTY:\[[0-9]+ x i8\]]] c"Content of device file3{{.+}}"
// CHECK-IR: [[OMP_INFO:@.+]] = internal local_unnamed_addr constant [2 x i64] [i64 ptrtoint (ptr [[OMP_BIN]] to i64), i64 24], section ".tgtimg", align 16

// CHECK-IR: [[OMP_IMAGES:@.+]] = internal unnamed_addr constant [1 x [[IMAGETY]]] [{{.+}} { ptr [[OMP_BIN]], ptr getelementptr inbounds ([[OMP_BINTY]], ptr [[OMP_BIN]], i64 1, i64 0), ptr [[ENTBEGIN]], ptr [[ENTEND]] }]

// CHECK-IR: [[OMP_DESC:@.+]] = internal constant [[DESCTY]] { i32 1, ptr [[OMP_IMAGES]], ptr [[ENTBEGIN]], ptr [[ENTEND]] }

// CHECK-IR: [[SYCL_TGT0:@.+]] = internal unnamed_addr constant [4 x i8] c"tg1\00"
// CHECK-IR: [[SYCL_COMPILE_OPTS0:@.+]] = internal unnamed_addr constant [3 x i8] c"-g\00"
// CHECK-IR: [[SYCL_LINK_OPTS0:@.+]] = internal unnamed_addr constant [21 x i8] c"-cl-denorms-are-zero\00"
// CHECK-IR: [[SYCL_BIN0:@.+]] = internal unnamed_addr constant [[SYCL_BIN0TY:\[[0-9]+ x i8\]]] c"Content of device file1{{.+}}"
// CHECK-IR: [[SYCL_INFO:@.+]] = internal local_unnamed_addr constant [2 x i64] [i64 ptrtoint (ptr [[SYCL_BIN0]] to i64), i64 24], section ".tgtimg", align 16

// CHECK-IR: [[SYCL_TGT1:@.+]] = internal unnamed_addr constant [4 x i8] c"tg2\00"
// CHECK-IR: [[SYCL_COMPILE_OPTS1:@.+]] = internal unnamed_addr constant [1 x i8] zeroinitializer
// CHECK-IR: [[SYCL_LINK_OPTS1:@.+]] = internal unnamed_addr constant [1 x i8] zeroinitializer
// CHECK-IR: [[SYCL_BIN1:@.+]] = internal unnamed_addr constant [[SYCL_BIN1TY:\[[0-9]+ x i8\]]] c"Content of device file2{{.+}}"
// CHECK-IR: [[SYCL_INFO1:@.+]] = internal local_unnamed_addr constant [2 x i64] [i64 ptrtoint (ptr [[SYCL_BIN1]] to i64), i64 24], section ".tgtimg", align 16

// CHECK-IR: [[SYCL_IMAGES:@.+]] = internal unnamed_addr constant [2 x [[SYCL_IMAGETY]]] [{{.*}} { i16 2, i8 4, i8 2, ptr [[SYCL_TGT0]], ptr [[SYCL_COMPILE_OPTS0]], ptr [[SYCL_LINK_OPTS0]], ptr null, ptr null, ptr [[SYCL_BIN0]], ptr getelementptr inbounds ([[SYCL_BIN0TY]], ptr [[SYCL_BIN0]], i64 1, i64 0), ptr null, ptr null, ptr null, ptr null }, [[SYCL_IMAGETY]] { i16 2, i8 4, i8 1, ptr [[SYCL_TGT1]], ptr [[SYCL_COMPILE_OPTS1]], ptr [[SYCL_LINK_OPTS1]], ptr null, ptr null, ptr [[SYCL_BIN1]], ptr getelementptr inbounds ([[SYCL_BIN1TY]], ptr [[SYCL_BIN1]], i64 1, i64 0), ptr null, ptr null, ptr null, ptr null }]

// CHECK-IR: [[SYCL_DESC:@.+]] = internal constant [[SYCL_DESCTY]] { i16 1, i16 2, ptr [[SYCL_IMAGES]], ptr null, ptr null }

// CHECK-IR: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr [[OMP_REGFN:@.+]], ptr null }, { i32, ptr, ptr } { i32 1, ptr [[SYCL_REGFN:@.+]], ptr null }]

// CHECK-IR: @llvm.global_dtors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr [[OMP_UNREGFN:@.+]], ptr null }, { i32, ptr, ptr } { i32 1, ptr [[SYCL_UNREGFN:@.+]], ptr null }]

// CHECK-IR: define internal void [[OMP_REGFN]]()
// CHECK-IR:   call void @__tgt_register_lib(ptr [[OMP_DESC]])
// CHECK-IR:   ret void

// CHECK-IR: declare void @__tgt_register_lib(ptr)

// CHECK-IR: define internal void [[OMP_UNREGFN]]()
// CHECK-IR:   call void @__tgt_unregister_lib(ptr [[OMP_DESC]])
// CHECK-IR:   ret void

// CHECK-IR: declare void @__tgt_unregister_lib(ptr)

// CHECK-IR: define internal void [[SYCL_REGFN]]()
// CHECK-IR:   call void @__sycl_register_lib(ptr [[SYCL_DESC]])
// CHECK-IR:   ret void

// CHECK-IR: declare void @__sycl_register_lib(ptr)

// CHECK-IR: define internal void [[SYCL_UNREGFN]]()
// CHECK-IR:   call void @__sycl_unregister_lib(ptr [[SYCL_DESC]])
// CHECK-IR:   ret void

// CHECK-IR: declare void @__sycl_unregister_lib(ptr)

// -------
// Check options' effects: -emit-reg-funcs, -desc-name
//
// RUN: echo 'Content of device file' > %t.tgt
//
// RUN: clang-offload-wrapper -kind sycl -host=x86_64-pc-linux-gnu -emit-reg-funcs=0 -desc-name=lalala -o - %t.tgt | llvm-dis | FileCheck %s --check-prefix CHECK-IR1
// CHECK-IR1: source_filename = "offload.wrapper.object"
// CHECK-IR1: [[IMAGETY:%.+]] = type { i16, i8, i8, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
// CHECK-IR1: [[DESCTY:%.+]] = type { i16, i16, ptr, ptr, ptr }
// CHECK-IR1-NOT: @llvm.global_ctors
// CHECK-IR1-NOT: @llvm.global_dtors
// CHECK-IR1-NOT: section ".tgtimg"
// CHECK-IR1: @.sycl_offloading.lalala = constant [[DESCTY]] { i16 {{[0-9]+}}, i16 1, ptr @.sycl_offloading.device_images, ptr null, ptr null }

// -------
// Check option's effects: -entries
//
// RUN: echo 'Content of device file' > %t.tgt
// RUN: echo -e 'entryA\nentryB' > %t.txt
// RUN: clang-offload-wrapper -host=x86_64-pc-linux-gnu -kind=sycl -entries=%t.txt %t.tgt -o - | llvm-dis | FileCheck %s --check-prefix CHECK-IR3
// CHECK-IR3: source_filename = "offload.wrapper.object"
// CHECK-IR3: @__sycl_offload_entry_name = internal unnamed_addr constant [7 x i8] c"entryA\00"
// CHECK-IR3: @__sycl_offload_entry_name.1 = internal unnamed_addr constant [7 x i8] c"entryB\00"
// CHECK-IR3: @__sycl_offload_entries_arr = internal constant [2 x %__tgt_offload_entry] [%__tgt_offload_entry { ptr null, ptr @__sycl_offload_entry_name, i64 0, i32 0, i32 0 }, %__tgt_offload_entry { ptr null, ptr @__sycl_offload_entry_name.1, i64 0, i32 0, i32 0 }]
// CHECK-IR3: @.sycl_offloading.device_images = internal unnamed_addr constant [1 x %__tgt_device_image] [%__tgt_device_image { {{.*}}, ptr @__sycl_offload_entries_arr, ptr getelementptr inbounds ([2 x %__tgt_offload_entry], ptr @__sycl_offload_entries_arr, i64 1, i64 0), ptr null, ptr null }]

// -------
// Check that device image can be extracted from the wrapper object by the clang-offload-bundler tool.
//
// RUN: clang-offload-wrapper -o %t.wrapper.bc -host=x86_64-pc-linux-gnu -kind=sycl -target=spir64-unknown-linux %t1.tgt
// RUN: %clang -target x86_64-pc-linux-gnu -c %t.wrapper.bc -o %t.wrapper.o
// RUN: clang-offload-bundler --type=o -input=%t.wrapper.o --targets=sycl-spir64-unknown-linux -output=%t1.out --unbundle
// RUN: diff %t1.out %t1.tgt

// Check that clang-offload-wrapper adds LLVMOMPOFFLOAD notes
// into the ELF offload images:
// RUN: yaml2obj %S/Inputs/empty-elf-template.yaml -o %t.64le -DBITS=64 -DENCODING=LSB
// RUN: clang-offload-wrapper -add-omp-offload-notes -kind=openmp -target=x86_64-pc-linux-gnu -o %t.wrapper.elf64le.bc %t.64le
// RUN: llvm-dis %t.wrapper.elf64le.bc -o - | FileCheck %s --check-prefix OMPNOTES
// RUN: yaml2obj %S/Inputs/empty-elf-template.yaml -o %t.64be -DBITS=64 -DENCODING=MSB
// RUN: clang-offload-wrapper -add-omp-offload-notes -kind=openmp -target=x86_64-pc-linux-gnu -o %t.wrapper.elf64be.bc %t.64be
// RUN: llvm-dis %t.wrapper.elf64be.bc -o - | FileCheck %s --check-prefix OMPNOTES
// RUN: yaml2obj %S/Inputs/empty-elf-template.yaml -o %t.32le -DBITS=32 -DENCODING=LSB
// RUN: clang-offload-wrapper -add-omp-offload-notes -kind=openmp -target=x86_64-pc-linux-gnu -o %t.wrapper.elf32le.bc %t.32le
// RUN: llvm-dis %t.wrapper.elf32le.bc -o - | FileCheck %s --check-prefix OMPNOTES
// RUN: yaml2obj %S/Inputs/empty-elf-template.yaml -o %t.32be -DBITS=32 -DENCODING=MSB
// RUN: clang-offload-wrapper -add-omp-offload-notes -kind=openmp -target=x86_64-pc-linux-gnu -o %t.wrapper.elf32be.bc %t.32be
// RUN: llvm-dis %t.wrapper.elf32be.bc -o - | FileCheck %s --check-prefix OMPNOTES

// There is no clean way for extracting the offload image
// from the object file currently, so try to find
// the inserted ELF notes in the device image variable's
// initializer:
// OMPNOTES: @{{.+}} = internal unnamed_addr constant [{{[0-9]+}} x i8] c"{{.*}}LLVMOMPOFFLOAD{{.*}}LLVMOMPOFFLOAD{{.*}}LLVMOMPOFFLOAD{{.*}}"
