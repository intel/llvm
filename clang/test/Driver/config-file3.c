// Needs symlinks
// UNSUPPORTED: system-windows
// REQUIRES: x86-registered-target

// RUN: rm -rf %t && mkdir %t

//--- If config file is specified by relative path (workdir/cfg-s2), it is searched for by that path.

// RUN: mkdir -p %t/workdir/subdir
// RUN: echo "@subdir/cfg-s2" > %t/workdir/cfg-1
// RUN: echo "-Wundefined-var-template" > %t/workdir/subdir/cfg-s2
//
// RUN: pushd %t
// RUN: %clang --config=workdir/cfg-1 -c -### %s 2>&1 | FileCheck %s -check-prefix CHECK-REL
// RUN: popd
//
// CHECK-REL: Configuration file: {{.*}}/workdir/cfg-1
// CHECK-REL: -Wundefined-var-template

//--- Config files are searched for in binary directory as well.
//
// RUN: mkdir %t/testbin
// RUN: ln -s %clang %t/testbin/dpclang
// RUN: echo "-Werror" > %t/testbin/aaa.cfg
// RUN: %t/testbin/dpclang --config-system-dir= --config-user-dir= --config=aaa.cfg -c -no-canonical-prefixes -### %s 2>&1 | FileCheck %s -check-prefix CHECK-BIN
//
// CHECK-BIN: Configuration file: {{.*}}/testbin/aaa.cfg
// CHECK-BIN: -Werror

//--- Invocation x86_64-unknown-linux-gnu-dpclang-g++ tries x86_64-unknown-linux-gnu-dpclang++.cfg first.
//
// RUN: mkdir %t/testdmode
// RUN: ln -s %clang %t/testdmode/cheribsd-riscv64-hybrid-dpclang++
// RUN: ln -s %clang %t/testdmode/qqq-dpclang-g++
// RUN: ln -s %clang %t/testdmode/x86_64-dpclang
// RUN: ln -s %clang %t/testdmode/i386-unknown-linux-gnu-dpclang-g++
// RUN: ln -s %clang %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++
// RUN: ln -s %clang %t/testdmode/x86_64-unknown-linux-gnu-dpclang
// RUN: touch %t/testdmode/cheribsd-riscv64-hybrid-dpclang++.cfg
// RUN: touch %t/testdmode/cheribsd-riscv64-hybrid.cfg
// RUN: touch %t/testdmode/qqq-dpclang-g++.cfg
// RUN: touch %t/testdmode/qqq.cfg
// RUN: touch %t/testdmode/x86_64-dpclang.cfg
// RUN: touch %t/testdmode/x86_64.cfg
// RUN: touch %t/testdmode/x86_64-unknown-linux-gnu-dpclang++.cfg
// RUN: touch %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++.cfg
// RUN: touch %t/testdmode/x86_64-unknown-linux-gnu-dpclang.cfg
// RUN: touch %t/testdmode/x86_64-unknown-linux-gnu.cfg
// RUN: touch %t/testdmode/i386-unknown-linux-gnu-dpclang++.cfg
// RUN: touch %t/testdmode/i386-unknown-linux-gnu-dpclang-g++.cfg
// RUN: touch %t/testdmode/i386-unknown-linux-gnu-dpclang.cfg
// RUN: touch %t/testdmode/i386-unknown-linux-gnu.cfg
// RUN: touch %t/testdmode/dpclang++.cfg
// RUN: touch %t/testdmode/dpclang-g++.cfg
// RUN: touch %t/testdmode/dpclang.cfg
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1 --implicit-check-not 'Configuration file:'
//
// FULL1: Configuration file: {{.*}}/testdmode/x86_64-unknown-linux-gnu-dpclang++.cfg

//--- -m32 overrides triple.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++ -m32 --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1-I386 --implicit-check-not 'Configuration file:'
//
// FULL1-I386: Configuration file: {{.*}}/testdmode/i386-unknown-linux-gnu-dpclang++.cfg

//--- --target= also works for overriding triple.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++ --target=i386-unknown-linux-gnu --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1-I386 --implicit-check-not 'Configuration file:'

//--- With --target= + -m64, -m64 takes precedence.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++ --target=i386-unknown-linux-gnu -m64 --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1 --implicit-check-not 'Configuration file:'

//--- i386 prefix also works for 32-bit.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/i386-unknown-linux-gnu-dpclang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1-I386 --implicit-check-not 'Configuration file:'

//--- i386 prefix + -m64 also works for 64-bit.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/i386-unknown-linux-gnu-dpclang-g++ -m64 --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1 --implicit-check-not 'Configuration file:'

//--- File specified by --config= is loaded after the one inferred from the executable.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++ --config-system-dir=%S/Inputs/config --config-user-dir= --config=i386-qqq.cfg -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix EXPLICIT --implicit-check-not 'Configuration file:'
//
// EXPLICIT: Configuration file: {{.*}}/testdmode/x86_64-unknown-linux-gnu-dpclang++.cfg
// EXPLICIT-NEXT: Configuration file: {{.*}}/Inputs/config/i386-qqq.cfg

//--- --no-default-config --config= loads only specified file.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++ --config-system-dir=%S/Inputs/config --config-user-dir= --no-default-config --config=i386-qqq.cfg -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix EXPLICIT-ONLY --implicit-check-not 'Configuration file:'
//
// EXPLICIT-ONLY: Configuration file: {{.*}}/Inputs/config/i386-qqq.cfg

//--- --no-default-config disables default filenames.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++ --config-system-dir=%S/Inputs/config --config-user-dir= --no-default-config -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix NO-CONFIG
//
// NO-CONFIG-NOT: Configuration file:

//--- --driver-mode= is respected.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++ --driver-mode=gcc --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1-GCC --implicit-check-not 'Configuration file:'
//
// FULL1-GCC: Configuration file: {{.*}}/testdmode/x86_64-unknown-linux-gnu-dpclang.cfg

//--- "dpclang" driver symlink should yield the "*-dpclang" configuration file.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1-GCC --implicit-check-not 'Configuration file:'

//--- "dpclang" + --driver-mode= should yield "*-dpclang++".
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang --driver-mode=g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1 --implicit-check-not 'Configuration file:'

//--- Clang started via name prefix that is not valid is forcing that prefix instead of target triple.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/qqq-dpclang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix QQQ --implicit-check-not 'Configuration file:'
//
// QQQ: Configuration file: {{.*}}/testdmode/qqq-dpclang-g++.cfg

//--- Explicit --target= overrides the triple even with non-standard name prefix.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/qqq-dpclang-g++ --target=x86_64-unknown-linux-gnu --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1 --implicit-check-not 'Configuration file:'

//--- "x86_64" prefix does not form a valid triple either.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-dpclang --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix X86_64 --implicit-check-not 'Configuration file:'
//
// X86_64: Configuration file: {{.*}}/testdmode/x86_64-dpclang.cfg

//--- Try cheribsd prefix using misordered triple components.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/cheribsd-riscv64-hybrid-dpclang++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix CHERIBSD --implicit-check-not 'Configuration file:'
//
// CHERIBSD: Configuration file: {{.*}}/testdmode/cheribsd-riscv64-hybrid-dpclang++.cfg

//--- Test fallback to x86_64-unknown-linux-gnu-dpclang-g++.cfg.
//
// RUN: rm %t/testdmode/x86_64-unknown-linux-gnu-dpclang++.cfg
// RUN: rm %t/testdmode/i386-unknown-linux-gnu-dpclang++.cfg
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL2 --implicit-check-not 'Configuration file:'
//
// FULL2: Configuration file: {{.*}}/testdmode/x86_64-unknown-linux-gnu-dpclang-g++.cfg

//--- FULL2 + -m32.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++ -m32 --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL2-I386 --implicit-check-not 'Configuration file:'
//
// FULL2-I386: Configuration file: {{.*}}/testdmode/i386-unknown-linux-gnu-dpclang-g++.cfg

//--- Test fallback to x86_64-unknown-linux-gnu-dpclang.cfg + dpclang++.cfg.
//
// RUN: rm %t/testdmode/cheribsd-riscv64-hybrid-dpclang++.cfg
// RUN: rm %t/testdmode/qqq-dpclang-g++.cfg
// RUN: rm %t/testdmode/x86_64-dpclang.cfg
// RUN: rm %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++.cfg
// RUN: rm %t/testdmode/i386-unknown-linux-gnu-dpclang-g++.cfg
// RUN: rm %t/testdmode/x86_64-unknown-linux-gnu-dpclang.cfg
// RUN: rm %t/testdmode/i386-unknown-linux-gnu-dpclang.cfg
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL3 --implicit-check-not 'Configuration file:'
//
// FULL3: Configuration file: {{.*}}/testdmode/dpclang++.cfg
// FULL3: Configuration file: {{.*}}/testdmode/x86_64-unknown-linux-gnu.cfg

//--- FULL3 + -m32.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++ -m32 --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL3-I386 --implicit-check-not 'Configuration file:'
//
// FULL3-I386: Configuration file: {{.*}}/testdmode/dpclang++.cfg
// FULL3-I386: Configuration file: {{.*}}/testdmode/i386-unknown-linux-gnu.cfg

//--- FULL3 + --driver-mode=.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++ --driver-mode=gcc --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL3-GCC --implicit-check-not 'Configuration file:'
//
// FULL3-GCC: Configuration file: {{.*}}/testdmode/dpclang.cfg
// FULL3-GCC: Configuration file: {{.*}}/testdmode/x86_64-unknown-linux-gnu.cfg

//--- QQQ fallback.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/qqq-dpclang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix QQQ-FALLBACK --implicit-check-not 'Configuration file:'
//
// QQQ-FALLBACK: Configuration file: {{.*}}/testdmode/dpclang++.cfg
// QQQ-FALLBACK: Configuration file: {{.*}}/testdmode/qqq.cfg

//--- "x86_64" falback.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-dpclang --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix X86_64-FALLBACK --implicit-check-not 'Configuration file:'
//
// X86_64-FALLBACK: Configuration file: {{.*}}/testdmode/dpclang.cfg
// X86_64-FALLBACK: Configuration file: {{.*}}/testdmode/x86_64.cfg

//--- cheribsd fallback.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/cheribsd-riscv64-hybrid-dpclang++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix CHERIBSD-FALLBACK --implicit-check-not 'Configuration file:'
//
// CHERIBSD-FALLBACK: Configuration file: {{.*}}/testdmode/dpclang++.cfg
// CHERIBSD-FALLBACK: Configuration file: {{.*}}/testdmode/cheribsd-riscv64-hybrid.cfg

//--- Test fallback to x86_64-unknown-linux-gnu.cfg + dpclang-g++.cfg.
//
// RUN: rm %t/testdmode/dpclang++.cfg
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL4 --implicit-check-not 'Configuration file:'
//
// FULL4: Configuration file: {{.*}}/testdmode/dpclang-g++.cfg
// FULL4: Configuration file: {{.*}}/testdmode/x86_64-unknown-linux-gnu.cfg

//--- Test fallback to dpclang-g++.cfg if x86_64-unknown-linux-gnu-dpclang.cfg does not exist.
//
// RUN: rm %t/testdmode/x86_64-unknown-linux-gnu.cfg
// RUN: rm %t/testdmode/i386-unknown-linux-gnu.cfg
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL5 --implicit-check-not 'Configuration file:'
//
// FULL5: Configuration file: {{.*}}/testdmode/dpclang-g++.cfg

//--- FULL5 + -m32.
//
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++ -m32 --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL5-I386 --implicit-check-not 'Configuration file:'
//
// FULL5-I386: Configuration file: {{.*}}/testdmode/dpclang-g++.cfg

//--- Test that incorrect driver mode config file is not used.
//
// RUN: rm %t/testdmode/dpclang-g++.cfg
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testdmode/x86_64-unknown-linux-gnu-dpclang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix NO-CONFIG

//--- Tilde expansion in user configuration file directory
//
// RUN: env HOME=%S/Inputs/config %clang -### --config-user-dir=~ -v 2>&1 | FileCheck %s --check-prefix=CHECK-TILDE
// CHECK-TILDE: User configuration file directory: {{.*}}/Inputs/config

//--- Fallback to stripping OS versions
//
// RUN: touch %t/testdmode/x86_64-apple-darwin23.6.0-dpclang.cfg
// RUN: touch %t/testdmode/x86_64-apple-darwin23-dpclang.cfg
// RUN: touch %t/testdmode/x86_64-apple-darwin-dpclang.cfg
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %clang -target x86_64-apple-darwin23.6.0 --config-system-dir=%t/testdmode --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix DARWIN --implicit-check-not 'Configuration file:'
//
// DARWIN: Configuration file: {{.*}}/testdmode/x86_64-apple-darwin23.6.0-dpclang.cfg

//--- DARWIN + no full version
//
// RUN: rm %t/testdmode/x86_64-apple-darwin23.6.0-dpclang.cfg
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %clang -target x86_64-apple-darwin23.6.0 --config-system-dir=%t/testdmode --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix DARWIN-MAJOR --implicit-check-not 'Configuration file:'
//
// DARWIN-MAJOR: Configuration file: {{.*}}/testdmode/x86_64-apple-darwin23-dpclang.cfg

//--- DARWIN + no version
//
// RUN: rm %t/testdmode/x86_64-apple-darwin23-dpclang.cfg
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %clang -target x86_64-apple-darwin23.6.0 --config-system-dir=%t/testdmode --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix DARWIN-VERSIONLESS --implicit-check-not 'Configuration file:'
//
// DARWIN-VERSIONLESS: Configuration file: {{.*}}/testdmode/x86_64-apple-darwin-dpclang.cfg
