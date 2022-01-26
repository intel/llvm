// RUN: %clang_cc1 -triple x86_64-linux-pc -fsycl-is-host -disable-llvm-passes -std=c++17 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -triple x86_64-linux-pc -fsycl-is-host -disable-llvm-passes -std=c++17 -emit-llvm -fsycl-unique-prefix=THE_PREFIX %s -o - | FileCheck %s --check-prefixes=PREFIX,CHECK

// A set of tests to validate the naming behavior of
// __builtin_sycl_unique_stable_id, both as it is altered by a kernel being
// named/instantiated, and for internal/global linkage.

#include "Inputs/sycl.hpp"

// Typically local statics are internal global symbols.
// CHECK: @[[FUNC_VAR:.+]] = private unnamed_addr constant
// PREFIX-SAME: THE_PREFIX
// CHECK-SAME: ____ZZ4FuncvE7FuncVar\00"

// CHECK: @[[GLOBAL_INT:.+]] = private unnamed_addr constant{{.+}} c"_Z9GlobalInt\00"

// Static/constexpr in global cause these to be local to the TU.
// CHECK: @[[STATIC_GLOBAL_INT:.+]] = private unnamed_addr constant
// PREFIX-SAME: THE_PREFIX
// CHECK-SAME: ____ZL15StaticGlobalInt\00"

// CHECK: @[[CONSTEXPR_GLOBAL_INT:.+]] = private unnamed_addr constant
// PREFIX-SAME: THE_PREFIX
// CHECK-SAME: ____ZL18ConstexprGlobalInt\00"

// CHECK: @[[STATIC_CONSTEXPR_GLOBAL_INT:.+]] = private unnamed_addr constant
// PREFIX-SAME: THE_PREFIX
// CHECK-SAME: ____ZL24StaticConstexprGlobalInt\00"


// Named-Namespace scope works the same as the global namespace.
// CHECK: @[[NS_INT:.+]] = private unnamed_addr constant{{.+}} c"_ZN2NS5NSIntE\00"

// Static/constexpr in global cause these to be local to the TU.
// CHECK: @[[STATIC_NS_INT:.+]] = private unnamed_addr constant
// PREFIX-SAME: THE_PREFIX
// CHECK-SAME: ____ZN2NSL11StaticNSIntE\00"

// CHECK: @[[CONSTEXPR_NS_INT:.+]] = private unnamed_addr constant
// PREFIX-SAME: THE_PREFIX
// CHECK-SAME: ____ZN2NSL14ConstexprNSIntE\00"

// CHECK: @[[STATIC_CONSTEXPR_NS_INT:.+]] = private unnamed_addr constant
// PREFIX-SAME: THE_PREFIX
// CHECK-SAME: ____ZN2NSL20StaticConstexprNSIntE\00"

// Anonymous-Namespace scope is all internal linkage.
// CHECK: @[[ANONNS_INT:.+]] = private unnamed_addr constant
// PREFIX-SAME: THE_PREFIX
// CHECK-SAME: ____ZN12_GLOBAL__N_19AnonNSIntE\00"

// Static/constexpr in global cause these to be local to the TU.
// CHECK: @[[STATIC_ANONNS_INT:.+]] = private unnamed_addr constant
// PREFIX-SAME: THE_PREFIX
// CHECK-SAME: ____ZN12_GLOBAL__N_115StaticAnonNSIntE\00"

// CHECK: @[[CONSTEXPR_ANONNS_INT:.+]] = private unnamed_addr constant
// PREFIX-SAME: THE_PREFIX
// CHECK-SAME: ____ZN12_GLOBAL__N_118ConstexprAnonNSIntE\00"

// CHECK: @[[STATIC_CONSTEXPR_ANONNS_INT:.+]] = private unnamed_addr constant
// PREFIX-SAME: THE_PREFIX
// CHECK-SAME: ____ZN12_GLOBAL__N_124StaticConstexprAnonNSIntE\00"

// Struct-statics are external.
// CHECK: @[[STRUCT_STATIC_INT:.+]] = private unnamed_addr constant{{.+}} c"_ZN6Struct15StaticStructIntE\00"
// CHECK: @[[STRUCT_STATIC_CONSTEXPR_INT:.+]] = private unnamed_addr constant{{.+}} c"_ZN6Struct24StaticConstexprStructIntE\00"

// Wrapped struct static works the same way.
// CHECK: @[[WRAPPED_GLOBAL_INT:.+]] = private unnamed_addr constant{{.+}} c"_Z9GlobalInt\00"

// Lambdas are all internal linkage.
// CHECK: @[[LOCAL_LAMBDA_1:.+]] = private unnamed_addr constant
// PREFIX-SAME: THE_PREFIX
// CHECK-SAME: ____ZZZ4mainENKUlvE_clEvE12LocalLambda1\00

// CHECK: @[[LOCAL_LAMBDA_2:.+]] = private unnamed_addr constant
// PREFIX-SAME: THE_PREFIX
// CHECK-SAME: ____ZZZ4mainENKUlvE0_clEvE12LocalLambda2\00

// CHECK: @[[LOCAL_LAMBDA_3:.+]] = private unnamed_addr constant
// PREFIX-SAME: THE_PREFIX
// CHECK-SAME: ____ZZZ4mainENKUlvE1_clEvE12LocalLambda3\00

// Because this one is in a template, it is a linkonce_odr global.
// CHECK: @[[TEMPL_FUNC_VAR:.+]] = private unnamed_addr constant{{.+}} c"_ZZ12TemplateFuncIfEvvE7FuncVar\00"

extern "C" int puts(const char *);

template <typename Ty>
auto func() -> decltype(__builtin_sycl_unique_stable_id(Ty::str));

struct Derp {
  static constexpr const char str[] = "derp derp derp";
};

template <typename KernelType>
void some_template(KernelType kernelFunc) {}

int GlobalInt;
static int StaticGlobalInt;
constexpr int ConstexprGlobalInt = 0;
static constexpr int StaticConstexprGlobalInt = 0;

namespace NS {
int NSInt;
static int StaticNSInt;
constexpr int ConstexprNSInt = 0;
static constexpr int StaticConstexprNSInt = 0;
}; // namespace NS

namespace {
int AnonNSInt;
static int StaticAnonNSInt;
constexpr int ConstexprAnonNSInt = 0;
static constexpr int StaticConstexprAnonNSInt = 0;
}; // namespace

struct Struct {
  static int StaticStructInt;
  constexpr static int StaticConstexprStructInt = 5;
};

template<auto &S>
struct Wrapper {
  static constexpr const char *ID = __builtin_sycl_unique_stable_id(S);
};


void Func() {
  static double FuncVar;
  constexpr const char *ID = __builtin_sycl_unique_stable_id(FuncVar);
  // CHECK: define{{.+}} void @_Z4Funcv()
  // CHECK: store i8* getelementptr inbounds
  // CHECK-SAME: @[[FUNC_VAR]]
};

template<typename T>
void TemplateFunc() {
  static T FuncVar;
  constexpr const char *ID = __builtin_sycl_unique_stable_id(FuncVar);
};

int main() {
  some_template(func<Derp>);
  // CHECK: call void @_Z13some_templateIPFPKcvEEvT_(i8* ()* noundef @_Z4funcI4DerpEDTcl31__builtin_sycl_unique_stable_idsrT_3strEEv)
  // Demangles to:
  // call void @void some_template<char const* (*)()>(char const* (*)())(i8* ()* @decltype(__builtin_sycl_unique_stable_id(Derp::str)) func<Derp>())

  puts(__builtin_sycl_unique_stable_id(GlobalInt));
  // CHECK: call i32 @puts({{.+}} @[[GLOBAL_INT]],
  puts(__builtin_sycl_unique_stable_id(StaticGlobalInt));
  // CHECK: call i32 @puts({{.+}} @[[STATIC_GLOBAL_INT]],
  puts(__builtin_sycl_unique_stable_id(ConstexprGlobalInt));
  // CHECK: call i32 @puts({{.+}} @[[CONSTEXPR_GLOBAL_INT]],
  puts(__builtin_sycl_unique_stable_id(StaticConstexprGlobalInt));
  // CHECK: call i32 @puts({{.+}} @[[STATIC_CONSTEXPR_GLOBAL_INT]],

  puts(__builtin_sycl_unique_stable_id(NS::NSInt));
  // CHECK: call i32 @puts({{.+}} @[[NS_INT]],
  puts(__builtin_sycl_unique_stable_id(NS::StaticNSInt));
  // CHECK: call i32 @puts({{.+}} @[[STATIC_NS_INT]],
  puts(__builtin_sycl_unique_stable_id(NS::ConstexprNSInt));
  // CHECK: call i32 @puts({{.+}} @[[CONSTEXPR_NS_INT]],
  puts(__builtin_sycl_unique_stable_id(NS::StaticConstexprNSInt));
  // CHECK: call i32 @puts({{.+}} @[[STATIC_CONSTEXPR_NS_INT]],

  puts(__builtin_sycl_unique_stable_id(AnonNSInt));
  // CHECK: call i32 @puts({{.+}} @[[ANONNS_INT]],
  puts(__builtin_sycl_unique_stable_id(StaticAnonNSInt));
  // CHECK: call i32 @puts({{.+}} @[[STATIC_ANONNS_INT]],
  puts(__builtin_sycl_unique_stable_id(ConstexprAnonNSInt));
  // CHECK: call i32 @puts({{.+}} @[[CONSTEXPR_ANONNS_INT]],
  puts(__builtin_sycl_unique_stable_id(StaticConstexprAnonNSInt));
  // CHECK: call i32 @puts({{.+}} @[[STATIC_CONSTEXPR_ANONNS_INT]],

  puts(__builtin_sycl_unique_stable_id(Struct::StaticStructInt));
  // CHECK: call i32 @puts({{.+}} @[[STRUCT_STATIC_INT]],
  puts(__builtin_sycl_unique_stable_id(Struct::StaticConstexprStructInt));
  // CHECK: call i32 @puts({{.+}} @[[STRUCT_STATIC_CONSTEXPR_INT]],
  puts(Wrapper<GlobalInt>::ID);
  // CHECK: call i32 @puts({{.+}} @[[WRAPPED_GLOBAL_INT]],

  // Ensure 'kernel naming' modifies the builtin. Wrapped in a lambda to make
  // sure it has its name changed when the kernel is named. All should have
  // internal linkage, since lambdas do.
  // This one is unmodified.
  []() {
    static int LocalLambda1;
    puts(__builtin_sycl_unique_stable_id(LocalLambda1));
    // CHECK: call i32 @puts({{.+}} @[[LOCAL_LAMBDA_1]],
  }();

  // Modified by kernel instantiation.
  []() {
    static int LocalLambda2;
    auto Lambda = [](){};
    cl::sycl::kernel_single_task<decltype(Lambda)>(Lambda);
    puts(__builtin_sycl_unique_stable_id(LocalLambda2));
    // CHECK: call i32 @puts({{.+}} @[[LOCAL_LAMBDA_2]],
  }();

  // Modified by mark-kernel-name builtin.
  []() {
    static int LocalLambda3;
    puts(__builtin_sycl_unique_stable_id(LocalLambda3));
    // CHECK: call i32 @puts({{.+}} @[[LOCAL_LAMBDA_3]],
  }();

  TemplateFunc<float>();
  // CHECK: define{{.+}} void @_Z12TemplateFuncIfEvv()
  // CHECK: store i8* getelementptr inbounds
  // CHECK-SAME: @[[TEMPL_FUNC_VAR]]

}
