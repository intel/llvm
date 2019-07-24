// RUN: DISABLE_INFER_AS=1 %clang_cc1 -triple spir64-unknown-linux-sycldevice  -std=c++11 -fsycl-is-device -disable-llvm-passes -emit-llvm -x c++ %s -o - | opt -asfix -S -o - | FileCheck %s --check-prefixes CHECK,CHECK-OLD
// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice  -std=c++11 -fsycl-is-device -disable-llvm-passes -emit-llvm -x c++ %s -o - | opt -asfix -S -o - | FileCheck %s --check-prefixes CHECK,CHECK-NEW
void bar(int & Data) {}
// CHECK-OLD-DAG: define spir_func void @[[RAW_REF:[a-zA-Z0-9_]+]](i32* dereferenceable(4) %
// CHECK-NEW-DAG: define spir_func void @[[RAW_REF:[a-zA-Z0-9_]+]](i32 addrspace(4)* dereferenceable(4) %
void bar2(int & Data) {}
// CHECK-OLD-DAG: define spir_func void @[[RAW_REF2:[a-zA-Z0-9_]+]](i32* dereferenceable(4) %
// CHECK-NEW-DAG: define spir_func void @[[RAW_REF2:[a-zA-Z0-9_]+]](i32 addrspace(4)* dereferenceable(4) %
void bar(__attribute__((ocl_local)) int &Data) {}
// CHECK-DAG: define spir_func void [[LOC_REF:@[a-zA-Z0-9_]+]](i32 addrspace(3)* dereferenceable(4) %
void foo(int * Data) {}
// CHECK-OLD-DAG: define spir_func void @[[RAW_PTR:[a-zA-Z0-9_]+]](i32* %
// CHECK-NEW-DAG: define spir_func void @[[RAW_PTR:[a-zA-Z0-9_]+]](i32 addrspace(4)* %
void foo2(int * Data) {}
// CHECK-OLD-DAG: define spir_func void @[[RAW_PTR2:[a-zA-Z0-9_]+]](i32* %
// CHECK-NEW-DAG: define spir_func void @[[RAW_PTR2:[a-zA-Z0-9_]+]](i32 addrspace(4)* %
void foo(__attribute__((address_space(3))) int * Data) {}
// CHECK-DAG: define spir_func void [[LOC_PTR:@[a-zA-Z0-9_]+]](i32 addrspace(3)* %

template<typename T>
void tmpl(T t){}
// See Check Lines below.

void usages() {
  // CHECK-DAG: [[GLOB:%[a-zA-Z0-9]+]] = alloca i32 addrspace(1)*
  __attribute__((address_space(1))) int *GLOB;
  // CHECK-DAG: [[LOC:%[a-zA-Z0-9]+]] = alloca i32 addrspace(3)*
  __attribute__((ocl_local)) int *LOC;
  // CHECK-OLD-DAG: [[NoAS:%[a-zA-Z0-9]+]] = alloca i32*
  // CHECK-NEW-DAG: [[NoAS:%[a-zA-Z0-9]+]] = alloca i32 addrspace(4)*
  int *NoAS;

  // CHECK-DAG: [[PRIV:%[a-zA-Z0-9]+]] = alloca i32*
  __attribute__((ocl_private)) int *PRIV;

  bar(*GLOB);
  // CHECK-DAG: [[GLOB_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[GLOB]]
  // CHECK-DAG: [[GLOB_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD]] to i32 addrspace(4)*
  // CHECK-OLD-DAG: call spir_func void @new.[[RAW_REF]](i32 addrspace(4)* [[GLOB_CAST]])
  // CHECK-NEW-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* dereferenceable(4) [[GLOB_CAST]])
  bar2(*GLOB);
  // CHECK-DAG: [[GLOB_LOAD2:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[GLOB]]
  // CHECK-DAG: [[GLOB_CAST2:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD2]] to i32 addrspace(4)*
  // CHECK-OLD-DAG: call spir_func void @new.[[RAW_REF2]](i32 addrspace(4)* [[GLOB_CAST2]])
  // CHECK-NEW-DAG: call spir_func void @[[RAW_REF2]](i32 addrspace(4)* dereferenceable(4) [[GLOB_CAST2]])

  bar(*LOC);
  // CHECK-DAG: [[LOC_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)** [[LOC]]
  // CHECK-DAG: call spir_func void [[LOC_REF]](i32 addrspace(3)* dereferenceable(4) [[LOC_LOAD]])
  bar2(*LOC);
  // CHECK-DAG: [[LOC_LOAD2:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)** [[LOC]]
  // CHECK-DAG: [[LOC_CAST2:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(3)* [[LOC_LOAD2]] to i32 addrspace(4)*
  // CHECK-OLD-DAG: call spir_func void @new.[[RAW_REF2]](i32 addrspace(4)* [[LOC_CAST2]])
  // CHECK-NEW-DAG: call spir_func void @[[RAW_REF2]](i32 addrspace(4)* dereferenceable(4) [[LOC_CAST2]])

  bar(*NoAS);
  // CHECK-OLD-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load i32*, i32** [[NoAS]]
  // CHECK-OLD-DAG: call spir_func void @[[RAW_REF]](i32* dereferenceable(4) [[NoAS_LOAD]])
  // CHECK-NEW-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)** [[NoAS]]
  // CHECK-NEW-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* dereferenceable(4) [[NoAS_LOAD]])
  bar2(*NoAS);
  // CHECK-OLD-DAG: [[NoAS_LOAD2:%[a-zA-Z0-9]+]] = load i32*, i32** [[NoAS]]
  // CHECK-OLD-DAG: call spir_func void @[[RAW_REF2]](i32* dereferenceable(4) [[NoAS_LOAD2]])
  // CHECK-NEW-DAG: [[NoAS_LOAD2:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)** [[NoAS]]
  // CHECK-NEW-DAG: call spir_func void @[[RAW_REF2]](i32 addrspace(4)* dereferenceable(4) [[NoAS_LOAD2]])

  foo(GLOB);
  // CHECK-DAG: [[GLOB_LOAD3:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[GLOB]]
  // CHECK-DAG: [[GLOB_CAST3:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD3]] to i32 addrspace(4)*
  // CHECK-OLD-DAG: call spir_func void @new.[[RAW_PTR]](i32 addrspace(4)* [[GLOB_CAST3]])
  // CHECK-NEW-DAG: call spir_func void @[[RAW_PTR]](i32 addrspace(4)* [[GLOB_CAST3]])
  foo2(GLOB);
  // CHECK-DAG: [[GLOB_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[GLOB]]
  // CHECK-DAG: [[GLOB_CAST4:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD4]] to i32 addrspace(4)*
  // CHECK-OLD-DAG: call spir_func void @new.[[RAW_PTR2]](i32 addrspace(4)* [[GLOB_CAST4]])
  // CHECK-NEW-DAG: call spir_func void @[[RAW_PTR2]](i32 addrspace(4)* [[GLOB_CAST4]])
  foo(LOC);
  // CHECK-DAG: [[LOC_LOAD3:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)** [[LOC]]
  // CHECK-DAG: call spir_func void [[LOC_PTR]](i32 addrspace(3)* [[LOC_LOAD3]])
  foo2(LOC);
  // CHECK-DAG: [[LOC_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)** [[LOC]]
  // CHECK-DAG: [[LOC_CAST4:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(3)* [[LOC_LOAD4]] to i32 addrspace(4)*
  // CHECK-OLD-DAG: call spir_func void @new.[[RAW_PTR2]](i32 addrspace(4)* [[LOC_CAST4]])
  // CHECK-NEW-DAG: call spir_func void @[[RAW_PTR2]](i32 addrspace(4)* [[LOC_CAST4]])
  foo(NoAS);
  // CHECK-OLD-DAG: [[NoAS_LOAD3:%[a-zA-Z0-9]+]] = load i32*, i32** [[NoAS]]
  // CHECK-OLD-DAG: call spir_func void @[[RAW_PTR]](i32* [[NoAS_LOAD3]])
  // CHECK-NEW-DAG: [[NoAS_LOAD3:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)** [[NoAS]]
  // CHECK-NEW-DAG: call spir_func void @[[RAW_PTR]](i32 addrspace(4)* [[NoAS_LOAD3]])
  foo2(NoAS);
  // CHECK-OLD-DAG: [[NoAS_LOAD4:%[a-zA-Z0-9]+]] = load i32*, i32** [[NoAS]]
  // CHECK-OLD-DAG: call spir_func void @[[RAW_PTR2]](i32* [[NoAS_LOAD4]])
  // CHECK-NEW-DAG: [[NoAS_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)** [[NoAS]]
  // CHECK-NEW-DAG: call spir_func void @[[RAW_PTR2]](i32 addrspace(4)* [[NoAS_LOAD4]])

  // Ensure that we still get 3 different template instantiations.
  tmpl(GLOB);
  // CHECK-DAG: [[GLOB_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[GLOB]]
  // CHECK-DAG: call spir_func void [[GLOB_TMPL:@[a-zA-Z0-9_]+]](i32 addrspace(1)* [[GLOB_LOAD4]])
  tmpl(LOC);
  // CHECK-DAG: [[LOC_LOAD5:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)** [[LOC]]
  // CHECK-DAG: call spir_func void [[LOC_TMPL:@[a-zA-Z0-9_]+]](i32 addrspace(3)* [[LOC_LOAD5]])
  tmpl(PRIV);
  // CHECK-DAG: [[PRIV_LOAD5:%[a-zA-Z0-9]+]] = load i32*, i32** [[PRIV]]
  // CHECK-DAG: call spir_func void [[PRIV_TMPL:@[a-zA-Z0-9_]+]](i32* [[PRIV_LOAD5]])
  tmpl(NoAS);
  // CHECK-OLD-DAG: [[NoAS_LOAD5:%[a-zA-Z0-9]+]] = load i32*, i32** [[NoAS]]
  // CHECK-OLD-DAG: call spir_func void [[AS0_TMPL:@[a-zA-Z0-9_]+]](i32* [[NoAS_LOAD5]])
  // CHECK-NEW-DAG: [[NoAS_LOAD5:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)** [[NoAS]]
  // CHECK-NEW-DAG: call spir_func void [[GEN_TMPL:@[a-zA-Z0-9_]+]](i32 addrspace(4)* [[NoAS_LOAD5]])
}

// CHECK-DAG: define linkonce_odr spir_func void [[GLOB_TMPL]](i32 addrspace(1)* %
// CHECK-DAG: define linkonce_odr spir_func void [[LOC_TMPL]](i32 addrspace(3)* %
// CHECK-OLD-DAG: define linkonce_odr spir_func void [[AS0_TMPL]](i32* %
// CHECK-NEW-DAG: define linkonce_odr spir_func void [[PRIV_TMPL]](i32* %
// CHECK-NEW-DAG: define linkonce_odr spir_func void [[GEN_TMPL]](i32 addrspace(4)* %

void usages2() {
  __attribute__((address_space(0))) int *PRIV_NUM;
  // CHECK-DAG: [[PRIV_NUM:%[a-zA-Z0-9_]+]] = alloca i32*
  __attribute__((address_space(0))) int *PRIV_NUM2;
  // CHECK-DAG: [[PRIV_NUM2:%[a-zA-Z0-9_]+]] = alloca i32*
  __attribute__((ocl_private)) int *PRIV;
  // CHECK-DAG: [[PRIV:%[a-zA-Z0-9_]+]] = alloca i32*
  __attribute__((address_space(1))) int *GLOB_NUM;
  // CHECK-DAG: [[GLOB_NUM:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(1)*
  __attribute__((ocl_global)) int *GLOB;
  // CHECK-DAG: [[GLOB:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(1)*
  __attribute__((address_space(2))) int *CONST_NUM;
  // CHECK-DAG: [[CONST_NUM:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(2)*
  __attribute__((ocl_constant)) int *CONST;
  // CHECK-DAG: [[CONST:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(2)*
  __attribute__((address_space(3))) int *LOCAL_NUM;
  // CHECK-DAG: [[LOCAL_NUM:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(3)*
  __attribute__((ocl_local)) int *LOCAL;
  // CHECK-DAG: [[LOCAL:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(3)*

  bar(*PRIV_NUM);
  // CHECK-DAG: [[PRIV_NUM_LOAD:%[a-zA-Z0-9]+]] = load i32*, i32** [[PRIV_NUM]]
  // CHECK-OLD-DAG: call spir_func void @[[RAW_REF]](i32* dereferenceable(4) [[PRIV_NUM_LOAD]])
  // CHECK-NEW-DAG: [[PRIV_NUM_ASCAST:%[a-zA-Z0-9]+]] = addrspacecast i32* [[PRIV_NUM_LOAD]] to i32 addrspace(4)*
  // CHECK-NEW-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* dereferenceable(4) [[PRIV_NUM_ASCAST]])
  bar(*PRIV_NUM2);
  // CHECK-DAG: [[PRIV_NUM2_LOAD:%[a-zA-Z0-9]+]] = load i32*, i32** [[PRIV_NUM2]]
  // CHECK-OLD-DAG: call spir_func void @[[RAW_REF]](i32* dereferenceable(4) [[PRIV_NUM2_LOAD]])
  // CHECK-NEW-DAG: [[PRIV_NUM2_ASCAST:%[a-zA-Z0-9]+]] = addrspacecast i32* [[PRIV_NUM2_LOAD]] to i32 addrspace(4)*
  // CHECK-NEW-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* dereferenceable(4) [[PRIV_NUM2_ASCAST]])
  bar(*PRIV);
  // CHECK-DAG: [[PRIV_LOAD:%[a-zA-Z0-9]+]] = load i32*, i32** [[PRIV]]
  // CHECK-OLD-DAG: call spir_func void @[[RAW_REF]](i32* dereferenceable(4) [[PRIV_LOAD]])
  // CHECK-NEW-DAG: [[PRIV_ASCAST:%[a-zA-Z0-9]+]] = addrspacecast i32* [[PRIV_LOAD]] to i32 addrspace(4)*
  // CHECK-NEW-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* dereferenceable(4) [[PRIV_ASCAST]])
  bar(*GLOB_NUM);
  // CHECK-DAG: [[GLOB_NUM_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[GLOB_NUM]]
  // CHECK-DAG: [[GLOB_NUM_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_NUM_LOAD]] to i32 addrspace(4)*
  // CHECK-OLD-DAG: call spir_func void @new.[[RAW_REF]](i32 addrspace(4)* [[GLOB_NUM_CAST]])
  // CHECK-NEW-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* dereferenceable(4) [[GLOB_NUM_CAST]])
  bar(*GLOB);
  // CHECK-DAG: [[GLOB_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[GLOB]]
  // CHECK-DAG: [[GLOB_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD]] to i32 addrspace(4)*
  // CHECK-OLD-DAG: call spir_func void @new.[[RAW_REF]](i32 addrspace(4)* [[GLOB_CAST]])
  // CHECK-NEW-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* dereferenceable(4) [[GLOB_CAST]])
  bar(*CONST_NUM);
  // CHECK-DAG: [[CONST_NUM_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(2)*, i32 addrspace(2)** [[CONST_NUM]]
  // CHECK-DAG: [[CONST_NUM_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(2)* [[CONST_NUM_LOAD]] to i32 addrspace(4)*
  // CHECK-OLD-DAG: call spir_func void @new.[[RAW_REF]](i32 addrspace(4)* [[CONST_NUM_CAST]])
  // CHECK-NEW-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* dereferenceable(4) [[CONST_NUM_CAST]])
  bar(*CONST);
  // CHECK-DAG: [[CONST_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(2)*, i32 addrspace(2)** [[CONST]]
  // CHECK-DAG: [[CONST_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(2)* [[CONST_LOAD]] to i32 addrspace(4)*
  // CHECK-OLD-DAG: call spir_func void @new.[[RAW_REF]](i32 addrspace(4)* [[CONST_CAST]])
  // CHECK-NEW-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* dereferenceable(4) [[CONST_CAST]])
  bar2(*LOCAL_NUM);
  // CHECK-DAG: [[LOCAL_NUM_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)** [[LOCAL_NUM]]
  // CHECK-DAG: [[LOCAL_NUM_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(3)* [[LOCAL_NUM_LOAD]] to i32 addrspace(4)*
  // CHECK-OLD-DAG: call spir_func void @new.[[RAW_REF2]](i32 addrspace(4)* [[LOCAL_NUM_CAST]])
  // CHECK-NEW-DAG: call spir_func void @[[RAW_REF2]](i32 addrspace(4)* dereferenceable(4) [[LOCAL_NUM_CAST]])
  bar2(*LOCAL);
  // CHECK-DAG: [[LOCAL_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)** [[LOCAL]]
  // CHECK-DAG: [[LOCAL_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(3)* [[LOCAL_LOAD]] to i32 addrspace(4)*
  // CHECK-OLD-DAG: call spir_func void @new.[[RAW_REF2]](i32 addrspace(4)* [[LOCAL_CAST]])
  // CHECK-NEW-DAG: call spir_func void @[[RAW_REF2]](i32 addrspace(4)* dereferenceable(4) [[LOCAL_CAST]])
}

// CHECK-OLD-DAG: define spir_func void @new.[[RAW_REF]](i32 addrspace(4)* dereferenceable(4)
// CHECK-OLD-DAG: define spir_func void @new.[[RAW_REF2]](i32 addrspace(4)* dereferenceable(4)
// CHECK-OLD-DAG: define spir_func void @new.[[RAW_PTR]](i32 addrspace(4)*
// CHECK-OLD-DAG: define spir_func void @new.[[RAW_PTR2]](i32 addrspace(4)*

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}
int main() {
  kernel_single_task<class fake_kernel>([]() { usages();usages2(); });
  return 0;
}

