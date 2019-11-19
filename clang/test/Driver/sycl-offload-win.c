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
// FOFFLOAD_STATIC_LIB: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=o"{{.+}} "-inputs=[[OBJ]]"{{.+}} "-unbundle"
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
// RUN:   | FileCheck -DLIB=%t.lib %s -check-prefixes=FOFFLOAD_STATIC_LIB_SRC,FOFFLOAD_STATIC_LIB_SRC_DEFAULT
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl -foffload-static-lib=%t.lib -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -DLIB=%t.lib %s -check-prefixes=FOFFLOAD_STATIC_LIB_SRC,FOFFLOAD_STATIC_LIB_SRC_CL
// FOFFLOAD_STATIC_LIB_SRC: 0: input, "[[INPUT:.+\.c]]", c, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 1: preprocessor, {0}, cpp-output, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 2: input, "[[INPUT]]", c, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 3: preprocessor, {2}, cpp-output, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 4: compiler, {3}, sycl-header, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC_DEFAULT: 5: offload, "host-sycl (x86_64-pc-windows-msvc)" {1}, "device-sycl (spir64-unknown-unknown-sycldevice)" {4}, cpp-output
// FOFFLOAD_STATIC_LIB_SRC_CL: 5: offload, "host-sycl (x86_64-pc-windows-msvc)" {1}, "device-sycl (spir64-unknown-unknown-sycldevice-coff)" {4}, cpp-output
// FOFFLOAD_STATIC_LIB_SRC: 6: compiler, {5}, ir, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 7: backend, {6}, assembler, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 8: assembler, {7}, object, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 9: linker, {8}, image, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 10: compiler, {3}, ir, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 11: backend, {10}, assembler, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 12: assembler, {11}, object, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 13: input, "[[LIB]]", archive
// FOFFLOAD_STATIC_LIB_SRC: 14: clang-offload-unbundler, {13}, archive
// FOFFLOAD_STATIC_LIB_SRC: 15: linker, {12, 14}, spirv, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 16: clang-offload-wrapper, {15}, object, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC_DEFAULT: 17: offload, "host-sycl (x86_64-pc-windows-msvc)" {9}, "device-sycl (spir64-unknown-unknown-sycldevice)" {16}, image
// FOFFLOAD_STATIC_LIB_SRC_CL: 17: offload, "host-sycl (x86_64-pc-windows-msvc)" {9}, "device-sycl (spir64-unknown-unknown-sycldevice-coff)" {16}, image

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
