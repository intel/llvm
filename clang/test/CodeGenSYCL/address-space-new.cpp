// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

struct SpaceWaster {
  int i, j;
};

struct HasX {
  int x;
};

struct Y : SpaceWaster, HasX {};

SYCL_EXTERNAL void bar(HasX &hx);

void baz(Y &y) {
  bar(y);
}

void test() {
  static const int foo = 0x42;
  // CHECK:    @_ZZ4testvE3foo = internal addrspace(1) constant i32 66, align 4

  // Intentionally leave a part of an array uninitialized. This triggers a
  // different code path contrary to a fully initialized array.
  static const unsigned bars[256] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20
  };
  (void)bars;
  // CHECK:    @_ZZ4testvE4bars = internal addrspace(1) constant <{ [21 x i32], [235 x i32] }> <{ [21 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20], [235 x i32] zeroinitializer }>, align 4

  // CHECK: @[[STR:[.a-zA-Z0-9_]+]] = private unnamed_addr addrspace(1) constant [14 x i8] c"Hello, world!\00", align 1

  // CHECK: %[[ARR:[a-zA-Z0-9]+]] = alloca [42 x i32]
  // CHECK: %[[GEN:.*]] = addrspacecast i32* %i to i32 addrspace(4)*
  // CHECK: [[ARR_ASCAST:%.*]] = addrspacecast [42 x i32]* %[[ARR]] to [42 x i32] addrspace(4)*

  int i = 0;
  int *pptr = &i;
  // CHECK: store i32 addrspace(4)* %[[GEN]], i32 addrspace(4)* addrspace(4)* %pptr.ascast
  bool is_i_ptr = (pptr == &i);
  // CHECK: %[[VALPPTR:[0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %pptr.ascast
  // CHECK: %cmp{{[0-9]*}} = icmp eq i32 addrspace(4)* %[[VALPPTR]], %i.ascast
  *pptr = foo;

  int var23 = 23;
  char *cp = (char *)&var23;
  *cp = 41;
  // CHECK: store i32 23, i32 addrspace(4)* %[[VAR:[a-zA-Z0-9]+]]
  // CHECK: [[VARCAST:[a-zA-Z0-9]+]] = bitcast i32 addrspace(4)* %[[VARAS:[a-zA-Z0-9]+]] to i8 addrspace(4)*
  // CHECK: store i8 addrspace(4)* %[[VARCAST]], i8 addrspace(4)* addrspace(4)* %{{.*}}

  int arr[42];
  char *cpp = (char *)arr;
  *cpp = 43;
  // CHECK: [[ARRAYDECAY1:%.*]] = getelementptr inbounds [42 x i32], [42 x i32] addrspace(4)* [[ARR_ASCAST]], i64 0, i64 0
  // CHECK: [[ADD_PTR:%.*]] = getelementptr inbounds i32, i32 addrspace(4)* [[ARRAYDECAY1]], i64 10
  // CHECK: store i32 addrspace(4)* [[ADD_PTR]], i32 addrspace(4)* addrspace(4)* %{{.*}}

  int *aptr = arr + 10;
  if (aptr < arr + sizeof(arr))
    *aptr = 44;
  // CHECK: [[TMP13:%.*]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %aptr.ascast
  // CHECK: [[ARRAYDECAY2:%.*]] = getelementptr inbounds [42 x i32], [42 x i32] addrspace(4)* [[ARR_ASCAST]], i64 0, i64 0
  // CHECK: [[ADD_PTR3:%.*]] = getelementptr inbounds i32, i32 addrspace(4)* [[ARRAYDECAY2]], i64 168
  // CHECK: [[CMP4:%.*]] = icmp ult i32 addrspace(4)* [[TMP13]], [[ADD_PTR3]]

  const char *str = "Hello, world!";
  // CHECK: store i8 addrspace(4)* getelementptr inbounds ([14 x i8], [14 x i8] addrspace(4)* addrspacecast ([14 x i8] addrspace(1)* @.str to [14 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* addrspace(4)* %[[STRVAL:[a-zA-Z0-9.]+]], align 8

  i = str[0];

  const char *phi_str = i > 2 ? str : "Another hello world!";
  (void)phi_str;
  // CHECK: %[[COND:[a-zA-Z0-9]+]] = icmp sgt i32 %{{.*}}, 2
  // CHECK: br i1 %[[COND]], label %[[CONDTRUE:[.a-zA-Z0-9]+]], label %[[CONDFALSE:[.a-zA-Z0-9]+]]

  // CHECK: [[CONDTRUE]]:
  // CHECK-NEXT: %[[VALTRUE:[a-zA-Z0-9]+]] = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %[[STRVAL]]
  // CHECK-NEXT: br label %[[CONDEND:[.a-zA-Z0-9]+]]

  // CHECK: [[CONDFALSE]]:

  // CHECK: [[CONDEND]]:
  // CHECK-NEXT: phi i8 addrspace(4)* [ %[[VALTRUE]], %[[CONDTRUE]] ], [ getelementptr inbounds ([21 x i8], [21 x i8] addrspace(4)* addrspacecast ([21 x i8] addrspace(1)* @{{.*}} to [21 x i8] addrspace(4)*), i64 0, i64 0), %[[CONDFALSE]] ]

  const char *select_null = i > 2 ? "Yet another Hello world" : nullptr;
  (void)select_null;
  // CHECK: select i1 %{{.*}}, i8 addrspace(4)* getelementptr inbounds ([24 x i8], [24 x i8] addrspace(4)* addrspacecast ([24 x i8] addrspace(1)* @{{.*}} to [24 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* null

  const char *select_str_trivial1 = true ? str : "Another hello world!";
  (void)select_str_trivial1;
  // CHECK: %[[TRIVIALTRUE:[a-zA-Z0-9]+]] = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %[[STRVAL]]
  // CHECK: store i8 addrspace(4)* %[[TRIVIALTRUE]], i8 addrspace(4)* addrspace(4)* %{{.*}}, align 8

  const char *select_str_trivial2 = false ? str : "Another hello world!";
  (void)select_str_trivial2;
  // CHECK: store i8 addrspace(4)* getelementptr inbounds ([21 x i8], [21 x i8] addrspace(4)* addrspacecast ([21 x i8] addrspace(1)* @{{.*}} to [21 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* addrspace(4)* %{{.*}}

  Y yy;
  baz(yy);
  // CHECK: define {{.*}}spir_func void @{{.*}}baz{{.*}}
  // CHECK: %[[FIRST:[a-zA-Z0-9]+]] = bitcast %struct.Y addrspace(4)* %{{.*}} to i8 addrspace(4)*
  // CHECK: %[[OFFSET:[a-zA-Z0-9]+]].ptr = getelementptr inbounds i8, i8 addrspace(4)* %[[FIRST]], i64 8
  // CHECK: %[[SECOND:[a-zA-Z0-9]+]] = bitcast i8 addrspace(4)* %[[OFFSET]].ptr to %struct.HasX addrspace(4)*
  // CHECK: call spir_func void @{{.*}}bar{{.*}}(%struct.HasX addrspace(4)* noundef align 4 dereferenceable(4) %[[SECOND]])
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([]() { test(); });
  return 0;
}
