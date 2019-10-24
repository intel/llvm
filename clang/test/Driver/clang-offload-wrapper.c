// REQUIRES: x86-registered-target

//
// Check help message.
//

// RUN: clang-offload-wrapper --help | FileCheck %s --check-prefix CHECK-HELP
// CHECK-HELP: {{.*}}OVERVIEW: A tool to create a wrapper bitcode for offload target binaries.
// CHECK-HELP: {{.*}}Takes offload target binaries and optional manifest files as input
// CHECK-HELP: {{.*}}and produces bitcode file containing target binaries packaged as data
// CHECK-HELP: {{.*}}and initialization code which registers target binaries in the offload
// CHECK-HELP: {{.*}}runtime. Manifest files format and contents are not restricted and are
// CHECK-HELP: {{.*}}a subject of agreement between the device compiler and the native
// CHECK-HELP: {{.*}}runtime for that device. When present, manifest file name should
// CHECK-HELP: {{.*}}immediately follow the corresponding device image filename on the
// CHECK-HELP: {{.*}}command line. Options annotating a device binary have effect on all
// CHECK-HELP: {{.*}}subsequent input, until redefined. For example:
// CHECK-HELP: {{.*}}$clang-offload-wrapper -host x86_64-pc-linux-gnu \
// CHECK-HELP: {{.*}}  -kind=sycl -target=spir64 -format=spirv -build-opts=-g \
// CHECK-HELP: {{.*}}  a.spv a_mf.txt \
// CHECK-HELP: {{.*}}             -target=xxx -format=native -build-opts=""  \
// CHECK-HELP: {{.*}}  b.bin b_mf.txt \
// CHECK-HELP: {{.*}}  -kind=openmp \
// CHECK-HELP: {{.*}}  c.bin
// CHECK-HELP: {{.*}}will generate an x86 wrapper object (.bc) enclosing the following
// CHECK-HELP: {{.*}}tuples describing a single device binary each ('-' means 'none')
// CHECK-HELP: {{.*}}offload kind | target | data format | data | manifest | build options:
// CHECK-HELP: {{.*}}----------------------------------------------------------------------
// CHECK-HELP: {{.*}}    sycl     | spir64 | spirv       | a.spv| a_mf.txt | -g
// CHECK-HELP: {{.*}}    sycl     | xxx    | native      | b.bin| b_mf.txt | -
// CHECK-HELP: {{.*}}    openmp   | xxx    | native      | c.bin| -        | -
// CHECK-HELP: {{.*}}USAGE: clang-offload-wrapper [options] <input  files>
// CHECK-HELP: {{.*}}OPTIONS:
// CHECK-HELP: {{.*}}clang-offload-wrapper options:
// CHECK-HELP: {{.*}}  --build-opts=<string>    - build options passed to the offload runtime
// CHECK-HELP: {{.*}}  --desc-name=<name>       - Specifies offload descriptor symbol name: '.<offload kind>.<name>', and makes it globally visible
// CHECK-HELP: {{.*}}  --emit-reg-funcs         - Emit [un-]registration functions
// CHECK-HELP: {{.*}}  --format=<value>         - device binary image formats:
// CHECK-HELP: {{.*}}    =none                  -   not set
// CHECK-HELP: {{.*}}    =native                -   unknown or native
// CHECK-HELP: {{.*}}    =spirv                 -   SPIRV binary
// CHECK-HELP: {{.*}}    =llvmbc                -   LLVMIR bitcode
// CHECK-HELP: {{.*}}  --host=<triple>          - Target triple for the output module
// CHECK-HELP: {{.*}}  --kind=<value>           - offload kind:
// CHECK-HELP: {{.*}}    =unknown               -   unknown
// CHECK-HELP: {{.*}}    =host                  -   host
// CHECK-HELP: {{.*}}    =openmp                -   OpenMP
// CHECK-HELP: {{.*}}    =hip                   -   HIP
// CHECK-HELP: {{.*}}    =sycl                  -   SYCL
// CHECK-HELP: {{.*}}  -o=<filename>            - Output filename
// CHECK-HELP: {{.*}}  --reg-func-name=<name>   - Offload descriptor registration function name
// CHECK-HELP: {{.*}}  --target=<string>        - offload target triple
// CHECK-HELP: {{.*}}  --unreg-func-name=<name> - Offload descriptor un-registration function name
// CHECK-HELP: {{.*}}  -v                       - verbose output

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
// RUN: clang-offload-wrapper                                                         \
// RUN:   -host=x86_64-pc-linux-gnu                                                   \
// RUN:     -kind=openmp -target=tg2                -format=native %t3.tgt %t1_mf.txt \
// RUN:     -kind=sycl   -target=tg1 -build-opts=-g -format spirv  %t1.tgt            \
// RUN:                  -target=tg2 -build-opts=   -format native %t2.tgt            \
// RUN:   -o %t.wrapper.bc
// RUN: llvm-dis %t.wrapper.bc -o - | FileCheck %s --check-prefix CHECK-IR

// CHECK-IR: target triple = "x86_64-pc-linux-gnu"

// CHECK-IR-DAG: [[ENTTY:%.+]] = type { i8*, i8*, i{{32|64}}, i32, i32 }
// CHECK-IR-DAG: [[IMAGETY:%.+]] = type { i8*, i8*, [[ENTTY]]*, [[ENTTY]]* }
// CHECK-IR-DAG: [[DESCTY:%.+]] = type { i32, [[IMAGETY]]*, [[ENTTY]]*, [[ENTTY]]* }

// CHECK-IR-DAG: [[SYCL_IMAGETY:%.+]] = type { i16, i8, i8, i8*, i8*, i8*, i8*, i8*, i8*, [[ENTTY]]*, [[ENTTY]]* }
// CHECK-IR-DAG: [[SYCL_DESCTY:%.+]] = type { i16, i16, [[SYCL_IMAGETY]]*, [[ENTTY]]*, [[ENTTY]]* }

// CHECK-IR: [[ENTBEGIN:@.+]] = external hidden constant [[ENTTY]]
// CHECK-IR: [[ENTEND:@.+]] = external hidden constant [[ENTTY]]

// CHECK-IR: [[DUMMY:@.+]] = hidden constant [0 x [[ENTTY]]] zeroinitializer, section "omp_offloading_entries"

// CHECK-IR: [[OMP_BIN:@.+]] = internal unnamed_addr constant [[OMP_BINTY:\[[0-9]+ x i8\]]] c"Content of device file3{{.+}}"

// CHECK-IR: [[OMP_IMAGES:@.+]] = internal unnamed_addr constant [1 x [[IMAGETY]]] [{{.+}} { i8* getelementptr inbounds ([[OMP_BINTY]], [[OMP_BINTY]]* [[OMP_BIN]], i64 0, i64 0), i8* getelementptr inbounds ([[OMP_BINTY]], [[OMP_BINTY]]* [[OMP_BIN]], i64 1, i64 0), [[ENTTY]]* [[ENTBEGIN]], [[ENTTY]]* [[ENTEND]] }]

// CHECK-IR: [[OMP_DESC:@.+]] = internal constant [[DESCTY]] { i32 1, [[IMAGETY]]* getelementptr inbounds ([1 x [[IMAGETY]]], [1 x [[IMAGETY]]]* [[OMP_IMAGES]], i64 0, i64 0), [[ENTTY]]* [[ENTBEGIN]], [[ENTTY]]* [[ENTEND]] }

// CHECK-IR: [[SYCL_TGT0:@.+]] = internal unnamed_addr constant [4 x i8] c"tg1\00"
// CHECK-IR: [[SYCL_OPTS0:@.+]] = internal unnamed_addr constant [3 x i8] c"-g\00"
// CHECK-IR: [[SYCL_BIN0:@.+]] = internal unnamed_addr constant [[SYCL_BIN0TY:\[[0-9]+ x i8\]]] c"Content of device file1{{.+}}"

// CHECK-IR: [[SYCL_TGT1:@.+]] = internal unnamed_addr constant [4 x i8] c"tg2\00"
// CHECK-IR: [[SYCL_OPTS1:@.+]] = internal unnamed_addr constant [1 x i8] zeroinitializer
// CHECK-IR: [[SYCL_BIN1:@.+]] = internal unnamed_addr constant  [[SYCL_BIN1TY:\[[0-9]+ x i8\]]] c"Content of device file2{{.+}}"

// CHECK-IR: [[SYCL_IMAGES:@.+]] = internal unnamed_addr constant [2 x [[SYCL_IMAGETY]]] [{{.+}} { i16 1, i8 4, i8 2, i8* getelementptr inbounds ([4 x i8], [4 x i8]* [[SYCL_TGT0]], i64 0, i64 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[SYCL_OPTS0]], i64 0, i64 0), i8* null, i8* null, i8* getelementptr inbounds ([[SYCL_BIN0TY]], [[SYCL_BIN0TY]]* [[SYCL_BIN0]], i64 0, i64 0), i8* getelementptr inbounds ([[SYCL_BIN0TY]], [[SYCL_BIN0TY]]* [[SYCL_BIN0]], i64 1, i64 0), [[ENTTY]]* null, [[ENTTY]]* null }, [[SYCL_IMAGETY]] { i16 1, i8 4, i8 1, i8* getelementptr inbounds ([4 x i8], [4 x i8]* [[SYCL_TGT1]], i64 0, i64 0), i8* getelementptr inbounds ([1 x i8], [1 x i8]* [[SYCL_OPTS1]], i64 0, i64 0), i8* null, i8* null, i8* getelementptr inbounds ([[SYCL_BIN1TY]], [[SYCL_BIN1TY]]* [[SYCL_BIN1]], i64 0, i64 0), i8* getelementptr inbounds ([[SYCL_BIN1TY]], [[SYCL_BIN1TY]]* [[SYCL_BIN1]], i64 1, i64 0), [[ENTTY]]* null, [[ENTTY]]* null }]

// CHECK-IR: [[SYCL_DESC:@.+]] = internal constant [[SYCL_DESCTY]] { i16 1, i16 2, [[SYCL_IMAGETY]]* getelementptr inbounds ([2 x [[SYCL_IMAGETY]]], [2 x [[SYCL_IMAGETY]]]* [[SYCL_IMAGES]], i64 0, i64 0), [[ENTTY]]* null, [[ENTTY]]* null }

// CHECK-IR: @llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 0, void ()* [[OMP_REGFN:@.+]], i8* null }, { i32, void ()*, i8* } { i32 0, void ()* [[SYCL_REGFN:@.+]], i8* null }]

// CHECK-IR: @llvm.global_dtors = appending global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 0, void ()* [[OMP_UNREGFN:@.+]], i8* null }, { i32, void ()*, i8* } { i32 0, void ()* [[SYCL_UNREGFN:@.+]], i8* null }]

// CHECK-IR: define internal void [[OMP_REGFN]]()
// CHECK-IR:   call void @__tgt_register_lib([[DESCTY]]* [[OMP_DESC]])
// CHECK-IR:   ret void

// CHECK-IR: declare void @__tgt_register_lib([[DESCTY]]*)

// CHECK-IR: define internal void [[OMP_UNREGFN]]()
// CHECK-IR:   call void @__tgt_unregister_lib([[DESCTY]]* [[OMP_DESC]])
// CHECK-IR:   ret void

// CHECK-IR: declare void @__tgt_unregister_lib([[DESCTY]]*)

// CHECK-IR: define internal void [[SYCL_REGFN]]()
// CHECK-IR:   call void @__tgt_register_lib([[DESCTY]]* bitcast ([[SYCL_DESCTY]]* [[SYCL_DESC]] to [[DESCTY]]*))
// CHECK-IR:   ret void

// CHECK-IR: define internal void [[SYCL_UNREGFN]]()
// CHECK-IR:   call void @__tgt_unregister_lib([[DESCTY]]* bitcast ([[SYCL_DESCTY]]* [[SYCL_DESC]] to [[DESCTY]]*))
// CHECK-IR:   ret void

// -------
// Check options' effects: -emit-reg-funcs, -desc-name
//
// RUN: echo 'Content of device file' > %t.tgt
//
// RUN: clang-offload-wrapper -kind sycl -host=x86_64-pc-linux-gnu -emit-reg-funcs=0 -desc-name=lalala -o - %t.tgt | llvm-dis | FileCheck %s --check-prefix CHECK-IR1
// CHECK-IR1: source_filename = "offload.wrapper.object"
// CHECK-IR1: [[IMAGETY:%.+]] = type { i16, i8, i8, i8*, i8*, i8*, i8*, i8*, i8*, %__tgt_offload_entry*, %__tgt_offload_entry* }
// CHECK-IR1: [[ENTTY:%.+]] = type { i8*, i8*, i64, i32, i32 }
// CHECK-IR1: [[DESCTY:%.+]] = type { i16, i16, [[IMAGETY]]*, [[ENTTY]]*, [[ENTTY]]* }
// CHECK-IR1-NOT: @llvm.global_ctors
// CHECK-IR1-NOT: @llvm.global_dtors
// CHECK-IR1: @.sycl_offloading.lalala = constant [[DESCTY]] { i16 1, i16 1, [[IMAGETY]]* getelementptr inbounds ([1 x [[IMAGETY]]], [1 x [[IMAGETY]]]* @.sycl_offloading.device_images, i64 0, i64 0), [[ENTTY]]* null, [[ENTTY]]* null }

// -------
// Check options' effects: -reg-func-name, -unreg-func-name
//
// RUN: clang-offload-wrapper -kind sycl -host=x86_64-pc-linux-gnu -reg-func-name=__REGFUNC__ -unreg-func-name=__UNREGFUNC__ -o - %t.tgt | llvm-dis | FileCheck %s --check-prefix CHECK-IR2
// CHECK-IR2: source_filename = "offload.wrapper.object"
// CHECK-IR2: define internal void {{.+}}()
// CHECK-IR2:   call void @__REGFUNC__
// CHECK-IR2:   ret void

// CHECK-IR2: declare void @__REGFUNC__

// CHECK-IR2: define internal void {{.+}}()
// CHECK-IR2:   call void @__UNREGFUNC__
// CHECK-IR2:   ret void

// CHECK-IR2: declare void @__UNREGFUNC__

// -------
// Check that device image can be extracted from the wrapper object by the clang-offload-bundler tool.
//
// RUN: clang-offload-wrapper -o %t.wrapper.bc -host=x86_64-pc-linux-gnu -kind=sycl -target=spir64-unknown-linux-sycldevice %t1.tgt
// RUN: %clang -target x86_64-pc-linux-gnu -c %t.wrapper.bc -o %t.wrapper.o
// RUN: clang-offload-bundler --type=o --inputs=%t.wrapper.o --targets=sycl-spir64-unknown-linux-sycldevice --outputs=%t1.out --unbundle
// RUN: diff %t1.out %t1.tgt
