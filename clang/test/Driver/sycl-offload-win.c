///
/// Perform several driver tests for SYCL offloading on Windows.
///

// REQUIRES: clang-driver
// REQUIRES: x86-registered-target

/// Test behaviors of -foffload-static-lib=<lib> with single object.
// RUN: touch %t.lib
// RUN: touch %t.obj
// RUN: %clang --target=x86_64-pc-windows-msvc -fsycl -foffload-static-lib=%t.lib %t.obj -### 2>&1 \
// RUN:   | FileCheck -DOBJ=%t.obj -DLIB=%t.lib %s -check-prefixes=FOFFLOAD_STATIC_LIB,FOFFLOAD_STATIC_LIB_DEFAULT
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl -foffload-static-lib=%t.lib %t.obj -### 2>&1 \
// RUN:   | FileCheck -DOBJ=%t.obj -DLIB=%t.lib %s -check-prefixes=FOFFLOAD_STATIC_LIB,FOFFLOAD_STATIC_LIB_CL
// FOFFLOAD_STATIC_LIB: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=o" "-targets=host-x86_64-pc-windows-msvc,sycl-spir64-unknown-unknown-sycldevice{{(-coff)?}}" "-inputs=[[OBJ]]" "-outputs={{.+}}.{{(o|obj)}},{{.+}}.{{(o|obj)}}" "-unbundle"
// FOFFLOAD_STATIC_LIB_DEFAULT: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=aoo" "-targets=sycl-spir64-{{.+}}-sycldevice" "-inputs=[[LIB]]"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_LIB_CL: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=aoo" "-targets=sycl-spir64-{{.+}}-sycldevice-coff" "-inputs=[[LIB]]"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_LIB: llvm-link{{(.exe)?}}{{.*}} "@{{.*}}"
// FOFFLOAD_STATIC_LIB: link{{(.exe)?}}{{.+}} "-defaultlib:[[LIB]]"

/// ###########################################################################

/// Test behaviors of -foffload-static-lib=<lib> with multiple objects.
// RUN: touch %t.lib
// RUN: touch %t-1.obj
// RUN: touch %t-2.obj
// RUN: touch %t-3.obj
// RUN: %clang --target=x86_64-pc-windows-msvc -fsycl -foffload-static-lib=%t.lib %t-1.obj %t-2.obj %t-3.obj -### 2>&1 \
// RUN:   | FileCheck -DOBJ1=%t-1.obj -DOBJ2=%t-2.obj -DOBJ3=%t-3.obj -DLIB=%t.lib %s -check-prefixes=FOFFLOAD_STATIC_LIB_MULTI_O,FOFFLOAD_STATIC_LIB_MULTI_O_DEFAULT
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl -foffload-static-lib=%t.lib %t-1.obj %t-2.obj %t-3.obj -### 2>&1 \
// RUN:   | FileCheck -DOBJ1=%t-1.obj -DOBJ2=%t-2.obj -DOBJ3=%t-3.obj -DLIB=%t.lib %s -check-prefixes=FOFFLOAD_STATIC_LIB_MULTI_O,FOFFLOAD_STATIC_LIB_MULTI_O_CL
// FOFFLOAD_STATIC_LIB_MULTI_O: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=o"{{.+}} "-inputs=[[OBJ1]]"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_LIB_MULTI_O: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=o"{{.+}} "-inputs=[[OBJ2]]"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_LIB_MULTI_O: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=o"{{.+}} "-inputs=[[OBJ3]]"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_LIB_MULTI_O_DEFAULT: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=aoo" "-targets=sycl-spir64-{{.+}}-sycldevice" "-inputs=[[LIB]]"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_LIB_MULTI_O_CL: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=aoo" "-targets=sycl-spir64-{{.+}}-sycldevice-coff" "-inputs=[[LIB]]"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_LIB_MULTI_O: llvm-link{{(.exe)?}}{{.*}} "@{{.*}}"
// FOFFLOAD_STATIC_LIB_MULTI_O: link{{(.exe)?}}{{.+}} "-defaultlib:[[LIB]]"

/// ###########################################################################

/// Test behaviors with multiple -foffload-static-lib=<lib> options.
// RUN: touch %t1.lib
// RUN: touch %t2.lib
// RUN: touch %t.obj
// RUN: %clang --target=x86_64-pc-windows-msvc -fsycl -foffload-static-lib=%t1.lib -foffload-static-lib=%t2.lib %t.obj -### 2>&1 \
// RUN:   | FileCheck -DOBJ=%t.obj -DLIB1=%t1.lib -DLIB2=%t2.lib %s -check-prefixes=FOFFLOAD_STATIC_MULTI_LIB,FOFFLOAD_STATIC_MULTI_LIB_DEFAULT
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl -foffload-static-lib=%t1.lib -foffload-static-lib=%t2.lib %t.obj -### 2>&1 \
// RUN:   | FileCheck -DOBJ=%t.obj -DLIB1=%t1.lib -DLIB2=%t2.lib %s -check-prefixes=FOFFLOAD_STATIC_MULTI_LIB,FOFFLOAD_STATIC_MULTI_LIB_CL
// FOFFLOAD_STATIC_MULTI_LIB: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=o"{{.+}} "-inputs=[[OBJ]]"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_MULTI_LIB_DEFAULT: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=aoo" "-targets=sycl-spir64-{{.+}}-sycldevice" "-inputs=[[LIB1]]"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_MULTI_LIB_DEFAULT: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=aoo" "-targets=sycl-spir64-{{.+}}-sycldevice" "-inputs=[[LIB2]]"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_MULTI_LIB_CL: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=aoo" "-targets=sycl-spir64-{{.+}}-sycldevice-coff" "-inputs=[[LIB1]]"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_MULTI_LIB_CL: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=aoo" "-targets=sycl-spir64-{{.+}}-sycldevice-coff" "-inputs=[[LIB2]]"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_MULTI_LIB: llvm-link{{(.exe)?}}{{.*}} "@{{.*}}" "@{{.*}}"
// FOFFLOAD_STATIC_MULTI_LIB: link{{(.exe)?}}{{.+}} "-defaultlib:[[LIB1]]" "-defaultlib:[[LIB2]]"

/// ###########################################################################

/// Test behaviors of -foffload-static-lib=<lib> from source.
// RUN: touch %t.lib
// RUN: %clang --target=x86_64-pc-windows-msvc -fsycl -foffload-static-lib=%t.lib -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -DLIB=%t.lib %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl -foffload-static-lib=%t.lib -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -DLIB=%t.lib %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC

// FOFFLOAD_STATIC_LIB_SRC: 0: input, "[[INPUTLIB:.+\.lib]]", object, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 1: input, "[[INPUTC:.+\.c]]", c, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 2: preprocessor, {1}, cpp-output, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 3: input, "[[INPUTC]]", c, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 4: preprocessor, {3}, cpp-output, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 5: compiler, {4}, sycl-header, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 6: offload, "host-sycl (x86_64-pc-windows-msvc)" {2}, "device-sycl (spir64-unknown-unknown-sycldevice{{(-coff)?}})" {5}, cpp-output
// FOFFLOAD_STATIC_LIB_SRC: 7: compiler, {6}, ir, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 8: backend, {7}, assembler, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 9: assembler, {8}, object, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 10: linker, {0, 9}, image, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 11: compiler, {4}, ir, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 12: backend, {11}, assembler, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 13: assembler, {12}, object, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 14: input, "[[INPUTLIB]]", archive
// FOFFLOAD_STATIC_LIB_SRC: 15: clang-offload-unbundler, {14}, archive
// FOFFLOAD_STATIC_LIB_SRC: 16: linker, {13, 15}, ir, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 17: llvm-spirv, {16}, spirv, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 18: clang-offload-wrapper, {17}, object, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 19: offload, "host-sycl (x86_64-pc-windows-msvc)" {10}, "device-sycl (spir64-unknown-unknown-sycldevice{{(-coff)?}})" {18}, image

/// ###########################################################################

// RUN: touch %t.lib
// RUN: %clang --target=x86_64-pc-windows-msvc -fsycl -foffload-static-lib=%t.lib %s -### 2>&1 \
// RUN:   | FileCheck -DLIB=%t.lib %s -check-prefixes=FOFFLOAD_STATIC_LIB_SRC2,FOFFLOAD_STATIC_LIB_SRC2_DEFAULT
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl -foffload-static-lib=%t.lib %s -### 2>&1 \
// RUN:   | FileCheck -DLIB=%t.lib %s -check-prefixes=FOFFLOAD_STATIC_LIB_SRC2,FOFFLOAD_STATIC_LIB_SRC2_CL
// FOFFLOAD_STATIC_LIB_SRC2_DEFAULT: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=aoo" "-targets=sycl-spir64-{{.+}}-sycldevice" "-inputs=[[LIB]]"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_LIB_SRC2_DEFAULT_CL: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=aoo" "-targets=sycl-spir64-{{.+}}-sycldevice-coff" "-inputs=[[LIB]]"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_LIB_SRC2: llvm-link{{(.exe)?}}{{.*}} "@{{.*}}"
// FOFFLOAD_STATIC_LIB_SRC2: link{{(.exe)?}}{{.+}} "-defaultlib:[[LIB]]"

// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows-msvc
