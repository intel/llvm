// RUN: DISABLE_INFER_AS=1 %clang_cc1 -triple spir64-unknown-linux-sycldevice  -std=c++11 -fsycl-is-device -disable-llvm-passes -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-LEGACY
// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice  -std=c++11 -fsycl-is-device -disable-llvm-passes -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NEW

void test() {
  static const int foo = 0x42;
  // CHECK-LEGACY: @_ZZ4testvE3foo = internal constant i32 66, align 4
  // CHECK-NEW:    @_ZZ4testvE3foo = internal addrspace(1) constant i32 66, align 4

  // CHECK: @[[STR:[.a-zA-Z0-9_]+]] = private unnamed_addr constant [14 x i8] c"Hello, world!\00", align 1

  // CHECK: %[[ARR:[a-zA-Z0-9]+]] = alloca [42 x i32]

  int i = 0;
  int *pptr = &i;
  // CHECK-LEGACY: store i32* %i, i32** %pptr
  // CHECK-NEW: %[[GEN:[0-9]+]] = addrspacecast i32* %i to i32 addrspace(4)*
  // CHECK-NEW: store i32 addrspace(4)* %[[GEN]], i32 addrspace(4)** %pptr
  *pptr = foo;

  int var23 = 23;
  char *cp = (char *)&var23;
  *cp = 41;
  // CHECK: store i32 23, i32* %[[VAR:[a-zA-Z0-9]+]]
  // CHECK-OLD: [[VARCAST:[a-zA-Z0-9]+]] = bitcast i32* %[[VAR]] to i8*
  // CHECK-OLD: store i8* %[[VARCAST]], i8** %{{.*}}
  // CHECK-NEW: [[VARAS:[a-zA-Z0-9]+]] = addrspacecast i32* %[[VAR]] to i32 addrspace(4)*
  // CHECK-NEW: [[VARCAST:[a-zA-Z0-9]+]] = bitcast i32 addrspace(4)* %[[VARAS]] to i8 addrspace(4)*
  // CHECK-NEW: store i8 addrspace(4)* %[[VARCAST]], i8 addrspace(4)** %{{.*}}

  int arr[42];
  char *cpp = (char *)arr;
  *cpp = 43;
  // CHECK:     %[[ARRDECAY:[a-zA-Z0-9]+]] = getelementptr inbounds [42 x i32], [42 x i32]* %[[ARR]], i64 0, i64 0
  // CHECK-OLD: [[ARRCAST:[a-zA-Z0-9]+]] = bitcast i32* %[[ARRDECAY]] to i8*
  // CHECK-OLD: store i8* %[[ARRCAST]], i8** %{{.*}}
  // CHECK-NEW: %[[ARRAS:[a-zA-Z0-9]+]] = addrspacecast i32* %[[ARRDECAY]] to i32 addrspace(4)*
  // CHECK-NEW: %[[ARRCAST:[a-zA-Z0-9]+]] = bitcast i32 addrspace(4)* %[[ARRAS]] to i8 addrspace(4)*
  // CHECK-NEW: store i8 addrspace(4)* %[[ARRCAST]], i8 addrspace(4)** %{{.*}}

  const char *str = "Hello, world!";
  // CHECK-LEGACY: store i8* getelementptr inbounds ([14 x i8], [14 x i8]* @[[STR]], i64 0, i64 0), i8** %[[STRVAL:[a-zA-Z0-9]+]], align 8
  // CHECK-NEW: store i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([14 x i8], [14 x i8]* @[[STR]], i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)** %[[STRVAL:[a-zA-Z0-9]+]], align 8

  i = str[0];

  const char *phi_str = i > 2 ? str : "Another hello world!";
  (void)phi_str;
  // CHECK: %[[COND:[a-zA-Z0-9]+]] = icmp sgt i32 %{{.*}}, 2
  // CHECK: br i1 %[[COND]], label %[[CONDTRUE:[.a-zA-Z0-9]+]], label %[[CONDFALSE:[.a-zA-Z0-9]+]]

  // CHECK: [[CONDTRUE]]:
  // CHECK-LEGACY-NEXT: %[[VALTRUE:[a-zA-Z0-9]+]] = load i8*, i8** %[[STRVAL]]
  // CHECK-LEGACY-NEXT: br label %[[CONDEND:[.a-zA-Z0-9]+]]
  // CHECK-NEW-NEXT: %[[VALTRUE:[a-zA-Z0-9]+]] = load i8 addrspace(4)*, i8 addrspace(4)** %[[STRVAL]]
  // CHECK-NEW-NEXT: br label %[[CONDEND:[.a-zA-Z0-9]+]]

  // CHECK: [[CONDFALSE]]:
  // CHECK-LEGACY-NEXT: br label %[[CONDEND]]

  // CHECK-LEGACY: [[CONDEND]]:
  // CHECK-NEW: [[CONDEND]]:
  // CHECK-NEW-NEXT: phi i8 addrspace(4)* [ %[[VALTRUE]], %[[CONDTRUE]] ], [ addrspacecast (i8* getelementptr inbounds ([21 x i8], [21 x i8]* @{{.*}}, i64 0, i64 0) to i8 addrspace(4)*), %[[CONDFALSE]] ]
  // CHECK-LEGACY-NEXT: phi i8* [ %[[VALTRUE]], %[[CONDTRUE]] ], [ getelementptr inbounds ([21 x i8], [21 x i8]* @{{.*}}, i64 0, i64 0), %[[CONDFALSE]] ]

  const char *select_null = i > 2 ? "Yet another Hello world" : nullptr;
  (void)select_null;
  // CHECK-LEGACY: select i1 %{{.*}}, i8* getelementptr inbounds ([24 x i8], [24 x i8]* @{{.*}}, i64 0, i64 0), i8* null
  // CHECK-NEW: select i1 %{{.*}}, i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([24 x i8], [24 x i8]* @{{.*}}, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* null

  const char *select_str_trivial1 = true ? str : "Another hello world!";
  (void)select_str_trivial1;
  // CHECK-LEGACY: %[[TRIVIALTRUE:[a-zA-Z0-9]+]] = load i8*, i8** %[[STRVAL]]
  // CHECK-LEGACY: store i8* %[[TRIVIALTRUE]], i8** %{{.*}}, align 8
  // CHECK-NEW: %[[TRIVIALTRUE:[a-zA-Z0-9]+]] = load i8 addrspace(4)*, i8 addrspace(4)** %[[STRVAL]]
  // CHECK-NEW: store i8 addrspace(4)* %[[TRIVIALTRUE]], i8 addrspace(4)** %{{.*}}, align 8

  const char *select_str_trivial2 = false ? str : "Another hello world!";
  (void)select_str_trivial2;
  // CHECK-LEGACY: store i8* getelementptr inbounds ([21 x i8], [21 x i8]* @{{.*}}, i64 0, i64 0), i8** %{{.*}}
  // CHECK-NEW: store i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([21 x i8], [21 x i8]* @{{.*}}, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)** %{{.*}}
}


template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}


int main() {
  kernel_single_task<class fake_kernel>([]() { test(); });
  return 0;
}

// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows-msvc
