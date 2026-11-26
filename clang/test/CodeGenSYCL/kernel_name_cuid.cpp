// The MD5-derived hash of `-cuid=abc` (see ASTContext::getCUIDHash):
// DEFINE: %{hash} = b04fd23c98500190
//
// RUN: %clang_cc1 -fsycl-is-device -fsycl-unnamed-lambda \
// RUN:   -internal-isystem %S/Inputs -triple spir64-unknown-unknown -cuid=abc \
// RUN:   -fsycl-int-header=%t.h -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=DEVICE -DHASH=%{hash}
// RUN: FileCheck %s --input-file=%t.h --check-prefix=INT-HEADER -DHASH=%{hash}
// RUN: %clang_cc1 -fsycl-is-host -fsycl-unnamed-lambda \
// RUN:   -internal-isystem %S/Inputs -triple x86_64-unknown-linux-gnu \
// RUN:   -cuid=abc -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=HOST -DHASH=%{hash}
// RUN: %clang_cc1 -fsycl-is-device -fsycl-unnamed-lambda \
// RUN:   -internal-isystem %S/Inputs -triple spir64-unknown-unknown \
// RUN:   -fsycl-int-header=%t.nocuid.h -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=NOCUID --implicit-check-not=cuid_
// RUN: FileCheck %s --input-file=%t.nocuid.h --check-prefix=NOCUID \
// RUN:   --implicit-check-not=cuid_
// RUN: %clang_cc1 -fsycl-is-host -fsycl-unnamed-lambda \
// RUN:   -internal-isystem %S/Inputs -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=HOST-NOCUID --implicit-check-not=cuid_
//
// The driver never selects triples asking for Microsoft C++ ABI, but such
// triples are still accepted when given explicitly. In this case, CUID is added
// as a synthetic outermost scope instead, instead of leaving TU-local kernel
// names to collide.
// RUN: %clang_cc1 -fsycl-is-device -fsycl-unnamed-lambda \
// RUN:   -internal-isystem %S/Inputs -triple spir64-pc-windows-msvc \
// RUN:   -aux-triple x86_64-pc-windows-msvc -cuid=abc \
// RUN:   -fsycl-int-header=%t.msvc.h -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=MSVC -DHASH=%{hash}
// RUN: FileCheck %s --input-file=%t.msvc.h --check-prefix=MSVC-INT-HEADER \
// RUN:   -DHASH=%{hash}
// RUN: %clang_cc1 -fsycl-is-device -fsycl-unnamed-lambda \
// RUN:   -internal-isystem %S/Inputs -triple spir64-pc-windows-msvc \
// RUN:   -aux-triple x86_64-pc-windows-msvc -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=MSVC-NOCUID --implicit-check-not=cuid_

// Verify that with `-cuid`, mangled names of TU-local kernels have a per-TU
// compilation unit ID inserted into them, to prevent TU-local kernels with
// identical mangled names from silently merging at device link time.

#include "sycl.hpp"

static void staticUnnamed(sycl::queue &q) {
  // Unnamed kernel in a function with internal linkage:
  auto k = [] {};
  q.submit([&](sycl::handler &h) { h.single_task(k); });
  // Compute k's kernel name as SYCL headers would do on host, in order to
  // find the kernel by name later:
  (void) __builtin_sycl_unique_stable_name(decltype(k));
  // Host, device, and integration header all need to agree:
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]ZL13staticUnnamedRN4sycl3_V15queueEEUlvE_(
  // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]ZL13staticUnnamedRN4sycl3_V15queueEEUlvE_"; }
  // INT-HEADER-DAG: struct KernelInfoData<'_', 'Z', 'T', 'S', 'U', '2', '1', 'c', 'u', 'i', 'd', '_',
  // HOST-DAG: c"_ZTSU21cuid_[[HASH]]ZL13staticUnnamedRN4sycl3_V15queueEEUlvE_\00"
  // HOST-NOCUID-DAG: c"_ZTSZL13staticUnnamedRN4sycl3_V15queueEEUlvE_\00"
}

void extUnnamed(sycl::queue &q) {
  // Unnamed kernel in an externally visible function, don't add CUID:
  auto k = [] {};
  q.submit([&](sycl::handler &h) { h.single_task(k); });
  (void) __builtin_sycl_unique_stable_name(decltype(k));
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSZ10extUnnamedRN4sycl3_V15queueEEUlvE_(
  // INT-HEADER-DAG: getName() { return "_ZTSZ10extUnnamedRN4sycl3_V15queueEEUlvE_"; }
  // HOST-DAG: c"_ZTSZ10extUnnamedRN4sycl3_V15queueEEUlvE_\00"

  // Unnamed kernel nested in a command group lambda: even if it's nested
  // deeper, since extUnnamed is still externally visible, we don't need a CUID
  // here either:
  q.submit([&](sycl::handler &h) { h.single_task([] {}); });
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSZZ10extUnnamedRN4sycl3_V15queueEENKUlRNS0_7handlerEE0_clES4_EUlvE_(
  // INT-HEADER-DAG: getName() { return "_ZTSZZ10extUnnamedRN4sycl3_V15queueEENKUlRNS0_7handlerEE0_clES4_EUlvE_"; }
  // No HOST check here: the nested lambda's type can only be named inside the
  // command group, which the mock queue::submit never invokes, so nothing
  // inside it is emitted on host.

}

inline void inlineFn(sycl::queue &q) {
  // Unnamed kernel in an inline function, which should be the same kernel in
  // every TU that includes it, so don't add CUID:
  auto k = [] {};
  q.submit([&](sycl::handler &h) { h.single_task(k); });
  (void) __builtin_sycl_unique_stable_name(decltype(k));
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSZ8inlineFnRN4sycl3_V15queueEEUlvE_(
  // INT-HEADER-DAG: getName() { return "_ZTSZ8inlineFnRN4sycl3_V15queueEEUlvE_"; }
  // HOST-DAG: c"_ZTSZ8inlineFnRN4sycl3_V15queueEEUlvE_\00"

  // Likewise, local types of an inline function are the same types in every TU,
  // so don't add CUID:
  q.submit([&](sycl::handler &h) { h.single_task<class KInline>([] {}); });
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSZZ8inlineFnRN4sycl3_V15queueEENKUlRNS0_7handlerEE0_clES4_E7KInline(
  // INT-HEADER-DAG: getName() { return "_ZTSZZ8inlineFnRN4sycl3_V15queueEENKUlRNS0_7handlerEE0_clES4_E7KInline"; }
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AVKInline@?0???R<lambda_3>@?0??inlineFn@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z@"(
  // MSVC-INT-HEADER-DAG: getName() { return "?AVKInline@?0???R<lambda_3>@?0??inlineFn@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z@"; }

  struct LocalFunc { void operator()() const {} };
  q.submit([&](sycl::handler &h) { h.single_task(LocalFunc{}); });
  (void) __builtin_sycl_unique_stable_name(LocalFunc);
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSZ8inlineFnRN4sycl3_V15queueEE9LocalFunc(
  // INT-HEADER-DAG: getName() { return "_ZTSZ8inlineFnRN4sycl3_V15queueEE9LocalFunc"; }
  // HOST-DAG: c"_ZTSZ8inlineFnRN4sycl3_V15queueEE9LocalFunc\00"
}

static inline void staticInlineFn(sycl::queue &q) {
  // Static inline is still internal linkage, so add CUID:
  auto k = [] {};
  q.submit([&](sycl::handler &h) { h.single_task(k); });
  (void) __builtin_sycl_unique_stable_name(decltype(k));
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]ZL14staticInlineFnRN4sycl3_V15queueEEUlvE_(
  // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]ZL14staticInlineFnRN4sycl3_V15queueEEUlvE_"; }
  // HOST-DAG: c"_ZTSU21cuid_[[HASH]]ZL14staticInlineFnRN4sycl3_V15queueEEUlvE_\00"
  // NOCUID-DAG: _ZTSZL14staticInlineFnRN4sycl3_V15queueEEUlvE_
  // HOST-NOCUID-DAG: c"_ZTSZL14staticInlineFnRN4sycl3_V15queueEEUlvE_\00"
}

struct InClassDef {
  // Implicitly inline, still externally visible, so don't add CUID:
  void run(sycl::queue &q) {
    auto k = [] {};
    q.submit([&](sycl::handler &h) { h.single_task(k); });
    (void) __builtin_sycl_unique_stable_name(decltype(k));
    // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSZN10InClassDef3runERN4sycl3_V15queueEEUlvE_(
    // INT-HEADER-DAG: getName() { return "_ZTSZN10InClassDef3runERN4sycl3_V15queueEEUlvE_"; }
    // HOST-DAG: c"_ZTSZN10InClassDef3runERN4sycl3_V15queueEEUlvE_\00"
  }
};

inline auto inlineVar = [] {};
static inline auto staticInlineVar = [] {};
struct StaticMember { static inline auto lambda = [] {}; };

void varLambdas(sycl::queue &q) {
  // Inline variable, don't add CUID:
  q.submit([&](sycl::handler &h) { h.single_task(inlineVar); });
  (void) __builtin_sycl_unique_stable_name(decltype(inlineVar));
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSN9inlineVarMUlvE_E(
  // INT-HEADER-DAG: getName() { return "_ZTSN9inlineVarMUlvE_E"; }
  // HOST-DAG: c"_ZTSN9inlineVarMUlvE_E\00"

  // Static member of an externally visible struct, don't add CUID:
  q.submit([&](sycl::handler &h) { h.single_task(StaticMember::lambda); });
  (void) __builtin_sycl_unique_stable_name(decltype(StaticMember::lambda));
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSN12StaticMember6lambdaMUlvE_E(
  // INT-HEADER-DAG: getName() { return "_ZTSN12StaticMember6lambdaMUlvE_E"; }
  // HOST-DAG: c"_ZTSN12StaticMember6lambdaMUlvE_E\00"

  // Namespace-scope static variable has internal linkage, so add CUID:
  q.submit([&](sycl::handler &h) { h.single_task(staticInlineVar); });
  (void) __builtin_sycl_unique_stable_name(decltype(staticInlineVar));
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]NL15staticInlineVarMUlvE_E(
  // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]NL15staticInlineVarMUlvE_E"; }
  // HOST-DAG: c"_ZTSU21cuid_[[HASH]]NL15staticInlineVarMUlvE_E\00"
  // NOCUID-DAG: _ZTSNL15staticInlineVarMUlvE_E
  // HOST-NOCUID-DAG: c"_ZTSNL15staticInlineVarMUlvE_E\00"
  // MSVC-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]NL15staticInlineVarMUlvE_E(
  // MSVC-NOCUID-DAG: define {{.*}}spir_kernel void @_ZTSNL15staticInlineVarMUlvE_E(
}

namespace {
struct AnonFunc { void operator()() const {} };
} // namespace

struct ExtFunc { void operator()() const {} };

void funcObjects(sycl::queue &q) {
  // Unnamed kernel given as a function object whose type is in an unnamed
  // namespace:
  q.submit([&](sycl::handler &h) { h.single_task(AnonFunc{}); });
  (void) __builtin_sycl_unique_stable_name(AnonFunc);
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]N12_GLOBAL__N_18AnonFuncE(
  // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]N12_GLOBAL__N_18AnonFuncE"; }
  // HOST-DAG: c"_ZTSU21cuid_[[HASH]]N12_GLOBAL__N_18AnonFuncE\00"
  // NOCUID-DAG: _ZTSN12_GLOBAL__N_18AnonFuncE

  // This time the function object is externally visible, so don't add CUID:
  q.submit([&](sycl::handler &h) { h.single_task(ExtFunc{}); });
  (void) __builtin_sycl_unique_stable_name(ExtFunc);
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTS7ExtFunc(
  // INT-HEADER-DAG: getName() { return "_ZTS7ExtFunc"; }
  // HOST-DAG: c"_ZTS7ExtFunc\00"
}

// Kernel names declared in place are local to the function that submits the
// kernel, so they are only TU-local when their parent functions are.
static void staticNamed(sycl::queue &q) {
  // Enclosing function has internal linkage:
  q.submit([&](sycl::handler &h) { h.single_task<class K>([] {}); });
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]ZZL11staticNamedRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E1K(
  // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]ZZL11staticNamedRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E1K"; }
  // NOCUID-DAG: _ZTSZZL11staticNamedRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E1K
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AVK@?1???R<lambda_1>@?0??staticNamed@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z<cuid_[[HASH]]>@@"(
  // MSVC-INT-HEADER-DAG: getName() { return "?AVK@?1???R<lambda_1>@?0??staticNamed@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z<cuid_[[HASH]]>@@"; }
  // MSVC-NOCUID-DAG: define {{.*}}spir_kernel void @"?AVK@?1???R<lambda_1>@?0??staticNamed@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z@"(
}

// Two distinct kernel names with the same name, local to the same function
// with internal linkage, are distinguished by a discriminator assigned by the
// mangler. The CUID must be added to these mangled names rather than falling
// back to the stable names, which lack the discriminator and would collide:
static void sameNamedLocals(sycl::queue &q) {
  q.submit([&](sycl::handler &h) {
    { h.single_task<class KLocal>([] {}); }
    { h.single_task<class KLocal>([] {}); }
  });
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]ZZL15sameNamedLocalsRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E6KLocal(
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]ZZL15sameNamedLocalsRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E6KLocal_0(
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AVKLocal@?1???R<lambda_1>@?0??sameNamedLocals@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z<cuid_[[HASH]]>@@"(
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AVKLocal@?2???R<lambda_1>@?0??sameNamedLocals@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z<cuid_[[HASH]]>@@"(
}

void extNamed(sycl::queue &q) {
  // Enclosing function is externally visible, so don't add CUID:
  q.submit([&](sycl::handler &h) { h.single_task<class K>([] {}); });
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSZZ8extNamedRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E1K(
  // INT-HEADER-DAG: getName() { return "_ZTSZZ8extNamedRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E1K"; }
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AVK@?1???R<lambda_1>@?0??extNamed@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z@"(
  // MSVC-INT-HEADER-DAG: getName() { return "?AVK@?1???R<lambda_1>@?0??extNamed@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z@"; }
}

namespace {
struct Anonymous {};
} // namespace
struct External {};

template <typename T> class KTmpl;

template <typename T> void templateFn(sycl::queue &q) {
  auto k = [] {};
  q.submit([&](sycl::handler &h) { h.single_task(k); });
  (void) __builtin_sycl_unique_stable_name(decltype(k));

  // If instantiated with a type in an unnamed namespace
  // (templateFn<Anonymous>), add CUID: Otherwise another TU doing the same with
  // its own 'Anonymous' would get the same name and get silently dropped at
  // device link time.
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]Z10templateFnIN12_GLOBAL__N_19AnonymousEEvRN4sycl3_V15queueEEUlvE_(
  // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]Z10templateFnIN12_GLOBAL__N_19AnonymousEEvRN4sycl3_V15queueEEUlvE_"; }
  // HOST-DAG: c"_ZTSU21cuid_[[HASH]]Z10templateFnIN12_GLOBAL__N_19AnonymousEEvRN4sycl3_V15queueEEUlvE_\00"
  // Don't add CUID if we don't have -cuid:
  // NOCUID-DAG: _ZTSZ10templateFnIN12_GLOBAL__N_19AnonymousEEvRN4sycl3_V15queueEEUlvE_

  // The same template instantiated with an externally visible type
  // (templateFn<External>) is the same kernel in every TU, so don't add CUID:
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSZ10templateFnI8ExternalEvRN4sycl3_V15queueEEUlvE_(
  // INT-HEADER-DAG: getName() { return "_ZTSZ10templateFnI8ExternalEvRN4sycl3_V15queueEEUlvE_"; }
  // HOST-DAG: c"_ZTSZ10templateFnI8ExternalEvRN4sycl3_V15queueEEUlvE_\00"

  // Same for an explicit kernel name that is a template specialization:
  q.submit([&](sycl::handler &h) { h.single_task<KTmpl<T>>([] {}); });
  (void) __builtin_sycl_unique_stable_name(KTmpl<T>);

  // With templateFn<Anonymous>:
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]5KTmplIN12_GLOBAL__N_19AnonymousEE(
  // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]5KTmplIN12_GLOBAL__N_19AnonymousEE"; }
  // HOST-DAG: c"_ZTSU21cuid_[[HASH]]5KTmplIN12_GLOBAL__N_19AnonymousEE\00"
  // NOCUID-DAG: _ZTS5KTmplIN12_GLOBAL__N_19AnonymousEE
  // HOST-NOCUID-DAG: c"_ZTS5KTmplIN12_GLOBAL__N_19AnonymousEE\00"
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AV?$KTmpl@UAnonymous@?A0x{{[0-9A-F]+}}@@@<cuid_[[HASH]]>@@"(
  // MSVC-INT-HEADER-DAG: getName() { return "?AV?$KTmpl@UAnonymous@?A0x{{[0-9A-F]+}}@@@<cuid_[[HASH]]>@@"; }
  // MSVC-NOCUID-DAG: define {{.*}}spir_kernel void @"?AV?$KTmpl@UAnonymous@?A0x{{[0-9A-F]+}}@@@@"(

  // With templateFn<External>:
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTS5KTmplI8ExternalE(
  // INT-HEADER-DAG: getName() { return "_ZTS5KTmplI8ExternalE"; }
  // HOST-DAG: c"_ZTS5KTmplI8ExternalE\00"
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AV?$KTmpl@UExternal@@@@"(
  // MSVC-INT-HEADER-DAG: getName() { return "?AV?$KTmpl@UExternal@@@@"; }
}

namespace {
class KAnon;
}
class KExt;

void explicitNames(sycl::queue &q) {
  // Named kernel, but name is in an unnamed namespace, so add CUID:
  q.submit([&](sycl::handler &h) { h.single_task<KAnon>([] {}); });
  (void) __builtin_sycl_unique_stable_name(KAnon);
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]N12_GLOBAL__N_15KAnonE(
  // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]N12_GLOBAL__N_15KAnonE"; }
  // HOST-DAG: c"_ZTSU21cuid_[[HASH]]N12_GLOBAL__N_15KAnonE\00"
  // NOCUID-DAG: _ZTSN12_GLOBAL__N_15KAnonE
  // HOST-NOCUID-DAG: c"_ZTSN12_GLOBAL__N_15KAnonE\00"
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AVKAnon@?A0x{{[0-9A-F]+}}@<cuid_[[HASH]]>@@"(
  // MSVC-INT-HEADER-DAG: getName() { return "?AVKAnon@?A0x{{[0-9A-F]+}}@<cuid_[[HASH]]>@@"; }
  // MSVC-NOCUID-DAG: define {{.*}}spir_kernel void @"?AVKAnon@?A0x{{[0-9A-F]+}}@@"(

  // Externally visible kernel name, don't add CUID:
  q.submit([&](sycl::handler &h) { h.single_task<KExt>([] {}); });
  (void) __builtin_sycl_unique_stable_name(KExt);
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTS4KExt(
  // INT-HEADER-DAG: getName() { return "_ZTS4KExt"; }
  // HOST-DAG: c"_ZTS4KExt\00"
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AVKExt@@"(
  // MSVC-INT-HEADER-DAG: getName() { return "?AVKExt@@"; }
}

// Kernel names aren't restricted to class types, so pointers, references, and
// array elements are mangled as the type they point to or contain. Thus, they
// are TU-local if the referenced type is.
static void staticPointerNames(sycl::queue &q) {
  // Pointer, reference, and array of classes local to a function with internal
  // linkage, add CUID:
  q.submit([&](sycl::handler &h) {
    h.single_task<class KStaticPtr *>([] {});
    // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]PZZL18staticPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E10KStaticPtr(
    // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]PZZL18staticPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E10KStaticPtr"; }
    // NOCUID-DAG: _ZTSPZZL18staticPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E10KStaticPtr
    // MSVC-DAG: define {{.*}}spir_kernel void @"?APEAVKStaticPtr@?1???R<lambda_1>@?0??staticPointerNames@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z<cuid_[[HASH]]>@@"(
    // MSVC-NOCUID-DAG: define {{.*}}spir_kernel void @"?APEAVKStaticPtr@?1???R<lambda_1>@?0??staticPointerNames@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z@"(
    h.single_task<class KStaticRef &>([] {});
    // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]RZZL18staticPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E10KStaticRef(
    // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]RZZL18staticPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E10KStaticRef"; }
    // NOCUID-DAG: _ZTSRZZL18staticPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E10KStaticRef
    // MSVC-DAG: define {{.*}}spir_kernel void @"?AAEAVKStaticRef@?1???R<lambda_1>@?0??staticPointerNames@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z<cuid_[[HASH]]>@@"(
    // MSVC-NOCUID-DAG: define {{.*}}spir_kernel void @"?AAEAVKStaticRef@?1???R<lambda_1>@?0??staticPointerNames@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z@"(
    h.single_task<class KStaticArr[2]>([] {});
    // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]A2_ZZL18staticPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E10KStaticArr(
    // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]A2_ZZL18staticPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E10KStaticArr"; }
    // NOCUID-DAG: _ZTSA2_ZZL18staticPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E10KStaticArr
    // MSVC-DAG: define {{.*}}spir_kernel void @"?AY01VKStaticArr@?1???R<lambda_1>@?0??staticPointerNames@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z<cuid_[[HASH]]>@@"(
    // MSVC-NOCUID-DAG: define {{.*}}spir_kernel void @"?AY01VKStaticArr@?1???R<lambda_1>@?0??staticPointerNames@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z@"(
    // Multiple levels of pointers with cv-qualifiers are looked through too:
    h.single_task<const class KStaticPtrPtr *const *>([] {});
    // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]PKPKZZL18staticPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E13KStaticPtrPtr(
    // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]PKPKZZL18staticPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E13KStaticPtrPtr"; }
    // NOCUID-DAG: _ZTSPKPKZZL18staticPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E13KStaticPtrPtr
    // MSVC-DAG: define {{.*}}spir_kernel void @"?APEBQEBVKStaticPtrPtr@?1???R<lambda_1>@?0??staticPointerNames@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z<cuid_[[HASH]]>@@"(
    // MSVC-NOCUID-DAG: define {{.*}}spir_kernel void @"?APEBQEBVKStaticPtrPtr@?1???R<lambda_1>@?0??staticPointerNames@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z@"(
  });
}

void extPointerNames(sycl::queue &q) {
  // Pointer, reference, and array of classes local to an externally visible
  // function. Like the classes themselves (see extNamed), don't add CUID:
  q.submit([&](sycl::handler &h) {
    h.single_task<class KLocalPtr *>([] {});
    // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSPZZ15extPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E9KLocalPtr(
    // INT-HEADER-DAG: getName() { return "_ZTSPZZ15extPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E9KLocalPtr"; }
    // MSVC-DAG: define {{.*}}spir_kernel void @"?APEAVKLocalPtr@?1???R<lambda_1>@?0??extPointerNames@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z@"(
    h.single_task<class KLocalRef &>([] {});
    // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSRZZ15extPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E9KLocalRef(
    // INT-HEADER-DAG: getName() { return "_ZTSRZZ15extPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E9KLocalRef"; }
    // MSVC-DAG: define {{.*}}spir_kernel void @"?AAEAVKLocalRef@?1???R<lambda_1>@?0??extPointerNames@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z@"(
    h.single_task<class KLocalArr[2]>([] {});
    // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSA2_ZZ15extPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E9KLocalArr(
    // INT-HEADER-DAG: getName() { return "_ZTSA2_ZZ15extPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E9KLocalArr"; }
    // MSVC-DAG: define {{.*}}spir_kernel void @"?AY01VKLocalArr@?1???R<lambda_1>@?0??extPointerNames@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z@"(
    // Multiple levels of pointers with cv-qualifiers are looked through too:
    h.single_task<const class KLocalPtrPtr *const *>([] {});
    // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSPKPKZZ15extPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E12KLocalPtrPtr(
    // INT-HEADER-DAG: getName() { return "_ZTSPKPKZZ15extPointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E12KLocalPtrPtr"; }
    // MSVC-DAG: define {{.*}}spir_kernel void @"?APEBQEBVKLocalPtrPtr@?1???R<lambda_1>@?0??extPointerNames@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z@"(
  });
}

inline void inlinePointerNames(sycl::queue &q) {
  // Pointer to a class local to an inline function, don't add CUID:
  q.submit([&](sycl::handler &h) { h.single_task<class KInlineLocal *>([] {}); });
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSPZZ18inlinePointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E12KInlineLocal(
  // INT-HEADER-DAG: getName() { return "_ZTSPZZ18inlinePointerNamesRN4sycl3_V15queueEENKUlRNS0_7handlerEE_clES4_E12KInlineLocal"; }
  // MSVC-DAG: define {{.*}}spir_kernel void @"?APEAVKInlineLocal@?0???R<lambda_1>@?0??inlinePointerNames@@YAXAEAVqueue@_V1@sycl@@@Z@QEBA?A?<auto>@@AEAVhandler@45@@Z@"(
}

void pointerNames(sycl::queue &q) {
  // Pointer to a class in an unnamed namespace, add CUID:
  q.submit([&](sycl::handler &h) { h.single_task<KAnon *>([] {}); });
  (void) __builtin_sycl_unique_stable_name(KAnon *);
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]PN12_GLOBAL__N_15KAnonE(
  // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]PN12_GLOBAL__N_15KAnonE"; }
  // HOST-DAG: c"_ZTSU21cuid_[[HASH]]PN12_GLOBAL__N_15KAnonE\00"
  // NOCUID-DAG: _ZTSPN12_GLOBAL__N_15KAnonE
  // HOST-NOCUID-DAG: c"_ZTSPN12_GLOBAL__N_15KAnonE\00"
  // MSVC-DAG: define {{.*}}spir_kernel void @"?APEAVKAnon@?A0x{{[0-9A-F]+}}@<cuid_[[HASH]]>@@"(
  // MSVC-INT-HEADER-DAG: getName() { return "?APEAVKAnon@?A0x{{[0-9A-F]+}}@<cuid_[[HASH]]>@@"; }
  // MSVC-NOCUID-DAG: define {{.*}}spir_kernel void @"?APEAVKAnon@?A0x{{[0-9A-F]+}}@@"(

  // Pointer to an externally visible class, don't add CUID:
  q.submit([&](sycl::handler &h) { h.single_task<KExt *>([] {}); });
  (void) __builtin_sycl_unique_stable_name(KExt *);
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSP4KExt(
  // INT-HEADER-DAG: getName() { return "_ZTSP4KExt"; }
  // HOST-DAG: c"_ZTSP4KExt\00"
  // MSVC-DAG: define {{.*}}spir_kernel void @"?APEAVKExt@@"(

  // Pointer to a const class in an unnamed namespace, add CUID:
  q.submit([&](sycl::handler &h) { h.single_task<const KAnon *>([] {}); });
  (void) __builtin_sycl_unique_stable_name(const KAnon *);
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]PKN12_GLOBAL__N_15KAnonE(
  // HOST-DAG: c"_ZTSU21cuid_[[HASH]]PKN12_GLOBAL__N_15KAnonE\00"
  // MSVC-DAG: define {{.*}}spir_kernel void @"?APEBVKAnon@?A0x{{[0-9A-F]+}}@<cuid_[[HASH]]>@@"(

  // Pointers as template arguments:
  q.submit([&](sycl::handler &h) { h.single_task<KTmpl<KAnon *>>([] {}); });
  (void) __builtin_sycl_unique_stable_name(KTmpl<KAnon *>);
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]5KTmplIPN12_GLOBAL__N_15KAnonEE(
  // HOST-DAG: c"_ZTSU21cuid_[[HASH]]5KTmplIPN12_GLOBAL__N_15KAnonEE\00"
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AV?$KTmpl@PEAVKAnon@?A0x{{[0-9A-F]+}}@@@<cuid_[[HASH]]>@@"(
  q.submit([&](sycl::handler &h) { h.single_task<KTmpl<KExt *>>([] {}); });
  (void) __builtin_sycl_unique_stable_name(KTmpl<KExt *>);
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTS5KTmplIP4KExtE(
  // HOST-DAG: c"_ZTS5KTmplIP4KExtE\00"
  // MSVC-DAG: define {{.*}}spir_kernel void @"?AV?$KTmpl@PEAVKExt@@@@"(

  // Member pointer to a class in an unnamed namespace, add CUID:
  q.submit([&](sycl::handler &h) { h.single_task<int KAnon::*>([] {}); });
  (void) __builtin_sycl_unique_stable_name(int KAnon::*);
  // DEVICE-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]MN12_GLOBAL__N_15KAnonEi(
  // INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]MN12_GLOBAL__N_15KAnonEi"; }
  // HOST-DAG: c"_ZTSU21cuid_[[HASH]]MN12_GLOBAL__N_15KAnonEi\00"
  // The Microsoft mangled name doesn't end in a qualified name, so CUID can't
  // be inserted into it, and the kernel falls back to the stable name instead.
  // MSVC-DAG: define {{.*}}spir_kernel void @_ZTSU21cuid_[[HASH]]MN12_GLOBAL__N_15KAnonEi(
  // MSVC-INT-HEADER-DAG: getName() { return "_ZTSU21cuid_[[HASH]]MN12_GLOBAL__N_15KAnonEi"; }
  // MSVC-NOCUID-DAG: define {{.*}}spir_kernel void @"?APEQKAnon@?A0x{{[0-9A-F]+}}@@H"(
}

int main() {
  sycl::queue q;

  staticUnnamed(q);
  templateFn<Anonymous>(q);
  templateFn<External>(q);
  extUnnamed(q);
  inlineFn(q);
  staticInlineFn(q);
  InClassDef{}.run(q);
  varLambdas(q);
  funcObjects(q);
  staticNamed(q);
  sameNamedLocals(q);
  extNamed(q);
  explicitNames(q);
  staticPointerNames(q);
  extPointerNames(q);
  inlinePointerNames(q);
  pointerNames(q);
}
