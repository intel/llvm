// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
void bar(int & Data) {}
// CHECK-DAG: define {{.*}}spir_func void @[[RAW_REF:[a-zA-Z0-9_]+]](i32 addrspace(4)* align 4 dereferenceable(4) %
void bar2(int & Data) {}
// CHECK-DAG: define {{.*}}spir_func void @[[RAW_REF2:[a-zA-Z0-9_]+]](i32 addrspace(4)* align 4 dereferenceable(4) %
void bar(__attribute__((opencl_local)) int &Data) {}
// CHECK-DAG: define {{.*}}spir_func void [[LOC_REF:@[a-zA-Z0-9_]+]](i32 addrspace(3)* align 4 dereferenceable(4) %
void bar3(__attribute__((opencl_global)) int &Data) {}
// CHECK-DAG: define {{.*}}spir_func void @[[GLOB_REF:[a-zA-Z0-9_]+]](i32 addrspace(1)* align 4 dereferenceable(4) %
void foo(int * Data) {}
// CHECK-DAG: define {{.*}}spir_func void @[[RAW_PTR:[a-zA-Z0-9_]+]](i32 addrspace(4)* %
void foo2(int * Data) {}
// CHECK-DAG: define {{.*}}spir_func void @[[RAW_PTR2:[a-zA-Z0-9_]+]](i32 addrspace(4)* %
void foo(__attribute__((opencl_local)) int *Data) {}
// CHECK-DAG: define {{.*}}spir_func void [[LOC_PTR:@[a-zA-Z0-9_]+]](i32 addrspace(3)* %
void foo3(__attribute__((opencl_global)) int *Data) {}
// CHECK-DAG: define {{.*}}spir_func void @[[GLOB_PTR:[a-zA-Z0-9_]+]](i32 addrspace(1)* %

template<typename T>
void tmpl(T t){}
// See Check Lines below.

void usages() {
  // CHECK-DAG: [[GLOBA:%[a-zA-Z0-9]+]] = alloca i32 addrspace(1)*
  // CHECK-DAG: [[GLOB:%.*]] = addrspacecast i32 addrspace(1)** [[GLOBA]] to i32 addrspace(1)* addrspace(4)*
  __attribute__((opencl_global)) int *GLOB;
  // CHECK-DAG: [[GLOBDEVA:%[a-zA-Z0-9]+]] = alloca i32 addrspace(5)*
  // CHECK-DAG: [[GLOBDEV:%.*]] = addrspacecast i32 addrspace(5)** [[GLOBDEVA]] to i32 addrspace(5)* addrspace(4)*
  __attribute__((opencl_global_device)) int *GLOBDEV;
  // CHECK-DAG: [[GLOBHOSTA:%[a-zA-Z0-9]+]] = alloca i32 addrspace(6)*
  // CHECK-DAG: [[GLOBHOST:%.*]] = addrspacecast i32 addrspace(6)** [[GLOBHOSTA]] to i32 addrspace(6)* addrspace(4)*
  __attribute__((opencl_global_host)) int *GLOBHOST;
  // CHECK-DAG: [[LOCA:%[a-zA-Z0-9]+]] = alloca i32 addrspace(3)*
  // CHECK-DAG: [[LOC:%.*]] = addrspacecast i32 addrspace(3)** [[LOCA]] to i32 addrspace(3)* addrspace(4)*
  __attribute__((opencl_local)) int *LOC;
  // CHECK-DAG: [[NoASA:%[a-zA-Z0-9]+]] = alloca i32 addrspace(4)*
  // CHECK-DAG: [[NoAS:%.*]] = addrspacecast i32 addrspace(4)** [[NoASA]] to i32 addrspace(4)* addrspace(4)*
  int *NoAS;
  // CHECK-DAG: [[PRIVA:%[a-zA-Z0-9]+]] = alloca i32*
  // CHECK-DAG: [[PRIV:%.*]] = addrspacecast i32** [[PRIVA]] to i32* addrspace(4)*
  __attribute__((opencl_private)) int *PRIV;

  bar(*GLOB);
  // CHECK-DAG: [[GLOB_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* [[GLOB]]
  // CHECK-DAG: [[GLOB_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* align 4 dereferenceable(4) [[GLOB_CAST]])
  bar2(*GLOB);
  // CHECK-DAG: [[GLOB_LOAD2:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* [[GLOB]]
  // CHECK-DAG: [[GLOB_CAST2:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD2]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_REF2]](i32 addrspace(4)* align 4 dereferenceable(4) [[GLOB_CAST2]])

  bar(*GLOBDEV);
  // CHECK-DAG: [[GLOBDEV_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(5)*, i32 addrspace(5)* addrspace(4)* [[GLOBDEV]]
  // CHECK-DAG: [[GLOBDEV_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(5)* [[GLOBDEV_LOAD]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* align 4 dereferenceable(4) [[GLOBDEV_CAST]])
  bar2(*GLOBDEV);
  // CHECK-DAG: [[GLOBDEV_LOAD2:%[a-zA-Z0-9]+]] = load i32 addrspace(5)*, i32 addrspace(5)* addrspace(4)* [[GLOBDEV]]
  // CHECK-DAG: [[GLOBDEV_CAST2:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(5)* [[GLOBDEV_LOAD2]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_REF2]](i32 addrspace(4)* align 4 dereferenceable(4) [[GLOBDEV_CAST2]])
  bar3(*GLOBDEV);
  // CHECK-DAG: [[GLOBDEV_LOAD3:%[a-zA-Z0-9]+]] = load i32 addrspace(5)*, i32 addrspace(5)* addrspace(4)* [[GLOBDEV]]
  // CHECK-DAG: [[GLOBDEV_CAST3:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(5)* [[GLOBDEV_LOAD3]] to i32 addrspace(1)*
  // CHECK-DAG: call spir_func void @[[GLOB_REF]](i32 addrspace(1)* align 4 dereferenceable(4) [[GLOBDEV_CAST3]])

  bar(*GLOBHOST);
  // CHECK-DAG: [[GLOBHOST_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(6)*, i32 addrspace(6)* addrspace(4)* [[GLOBHOST]]
  // CHECK-DAG: [[GLOBHOST_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(6)* [[GLOBHOST_LOAD]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* align 4 dereferenceable(4) [[GLOBHOST_CAST]])
  bar2(*GLOBHOST);
  // CHECK-DAG: [[GLOBHOST_LOAD2:%[a-zA-Z0-9]+]] = load i32 addrspace(6)*, i32 addrspace(6)* addrspace(4)* [[GLOBHOST]]
  // CHECK-DAG: [[GLOBHOST_CAST2:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(6)* [[GLOBHOST_LOAD2]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_REF2]](i32 addrspace(4)* align 4 dereferenceable(4) [[GLOBHOST_CAST2]])
  bar3(*GLOBHOST);
  // CHECK-DAG: [[GLOBHOST_LOAD3:%[a-zA-Z0-9]+]] = load i32 addrspace(6)*, i32 addrspace(6)* addrspace(4)* [[GLOBHOST]]
  // CHECK-DAG: [[GLOBHOST_CAST3:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(6)* [[GLOBHOST_LOAD3]] to i32 addrspace(1)*
  // CHECK-DAG: call spir_func void @[[GLOB_REF]](i32 addrspace(1)* align 4 dereferenceable(4) [[GLOBHOST_CAST3]])

  bar(*LOC);
  // CHECK-DAG: [[LOC_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(4)* [[LOC]]
  // CHECK-DAG: call spir_func void [[LOC_REF]](i32 addrspace(3)* align 4 dereferenceable(4) [[LOC_LOAD]])
  bar2(*LOC);
  // CHECK-DAG: [[LOC_LOAD2:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(4)* [[LOC]]
  // CHECK-DAG: [[LOC_CAST2:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(3)* [[LOC_LOAD2]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_REF2]](i32 addrspace(4)* align 4 dereferenceable(4) [[LOC_CAST2]])

  bar(*NoAS);
  // CHECK-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* [[NoAS]]
  // CHECK-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* align 4 dereferenceable(4) [[NoAS_LOAD]])
  bar2(*NoAS);
  // CHECK-DAG: [[NoAS_LOAD2:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* [[NoAS]]
  // CHECK-DAG: call spir_func void @[[RAW_REF2]](i32 addrspace(4)* align 4 dereferenceable(4) [[NoAS_LOAD2]])

  foo(GLOB);
  // CHECK-DAG: [[GLOB_LOAD3:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* [[GLOB]]
  // CHECK-DAG: [[GLOB_CAST3:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD3]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_PTR]](i32 addrspace(4)* [[GLOB_CAST3]])
  foo2(GLOB);
  // CHECK-DAG: [[GLOB_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* [[GLOB]]
  // CHECK-DAG: [[GLOB_CAST4:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD4]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_PTR2]](i32 addrspace(4)* [[GLOB_CAST4]])
  foo(GLOBDEV);
  // CHECK-DAG: [[GLOBDEV_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(5)*, i32 addrspace(5)* addrspace(4)* [[GLOBDEV]]
  // CHECK-DAG: [[GLOBDEV_CAST4:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(5)* [[GLOBDEV_LOAD4]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_PTR]](i32 addrspace(4)* [[GLOBDEV_CAST4]])
  foo2(GLOBDEV);
  // CHECK-DAG: [[GLOBDEV_LOAD5:%[a-zA-Z0-9]+]] = load i32 addrspace(5)*, i32 addrspace(5)* addrspace(4)* [[GLOBDEV]]
  // CHECK-DAG: [[GLOBDEV_CAST5:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(5)* [[GLOBDEV_LOAD5]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_PTR2]](i32 addrspace(4)* [[GLOBDEV_CAST5]])
  foo3(GLOBDEV);
  // CHECK-DAG: [[GLOBDEV_LOAD6:%[a-zA-Z0-9]+]] = load i32 addrspace(5)*, i32 addrspace(5)* addrspace(4)* [[GLOBDEV]]
  // CHECK-DAG: [[GLOBDEV_CAST6:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(5)* [[GLOBDEV_LOAD6]] to i32 addrspace(1)*
  // CHECK-DAG: call spir_func void @[[GLOB_PTR]](i32 addrspace(1)* [[GLOBDEV_CAST6]])
  foo(GLOBHOST);
  // CHECK-DAG: [[GLOBHOST_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(6)*, i32 addrspace(6)* addrspace(4)* [[GLOBHOST]]
  // CHECK-DAG: [[GLOBHOST_CAST4:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(6)* [[GLOBHOST_LOAD4]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_PTR]](i32 addrspace(4)* [[GLOBHOST_CAST4]])
  foo2(GLOBHOST);
  // CHECK-DAG: [[GLOBHOST_LOAD5:%[a-zA-Z0-9]+]] = load i32 addrspace(6)*, i32 addrspace(6)* addrspace(4)* [[GLOBHOST]]
  // CHECK-DAG: [[GLOBHOST_CAST5:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(6)* [[GLOBHOST_LOAD5]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_PTR2]](i32 addrspace(4)* [[GLOBHOST_CAST5]])
  foo3(GLOBHOST);
  // CHECK-DAG: [[GLOBHOST_LOAD6:%[a-zA-Z0-9]+]] = load i32 addrspace(6)*, i32 addrspace(6)* addrspace(4)* [[GLOBHOST]]
  // CHECK-DAG: [[GLOBHOST_CAST6:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(6)* [[GLOBHOST_LOAD6]] to i32 addrspace(1)*
  // CHECK-DAG: call spir_func void @[[GLOB_PTR]](i32 addrspace(1)* [[GLOBHOST_CAST6]])
  foo(LOC);
  // CHECK-DAG: [[LOC_LOAD3:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(4)* [[LOC]]
  // CHECK-DAG: call spir_func void [[LOC_PTR]](i32 addrspace(3)* [[LOC_LOAD3]])
  foo2(LOC);
  // CHECK-DAG: [[LOC_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(4)* [[LOC]]
  // CHECK-DAG: [[LOC_CAST4:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(3)* [[LOC_LOAD4]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_PTR2]](i32 addrspace(4)* [[LOC_CAST4]])
  foo(NoAS);
  // CHECK-DAG: [[NoAS_LOAD3:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* [[NoAS]]
  // CHECK-DAG: call spir_func void @[[RAW_PTR]](i32 addrspace(4)* [[NoAS_LOAD3]])
  foo2(NoAS);
  // CHECK-DAG: [[NoAS_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* [[NoAS]]
  // CHECK-DAG: call spir_func void @[[RAW_PTR2]](i32 addrspace(4)* [[NoAS_LOAD4]])

  // Ensure that we still get 5 different template instantiations.
  tmpl(GLOB);
  // CHECK-DAG: [[GLOB_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* [[GLOB]]
  // CHECK-DAG: call spir_func void [[GLOB_TMPL:@[a-zA-Z0-9_]+]](i32 addrspace(1)* [[GLOB_LOAD4]])
  tmpl(GLOBDEV);
  // CHECK-DAG: [[GLOBDEV_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(5)*, i32 addrspace(5)* addrspace(4)* [[GLOBDEV]]
  // CHECK-DAG: call spir_func void [[GLOBDEV_TMPL:@[a-zA-Z0-9_]+]](i32 addrspace(5)* [[GLOBDEV_LOAD4]])
  tmpl(GLOBHOST);
  // CHECK-DAG: [[GLOBHOST_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(6)*, i32 addrspace(6)* addrspace(4)* [[GLOBHOST]]
  // CHECK-DAG: call spir_func void [[GLOBHOST_TMPL:@[a-zA-Z0-9_]+]](i32 addrspace(6)* [[GLOBHOST_LOAD4]])
  tmpl(LOC);
  // CHECK-DAG: [[LOC_LOAD5:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(4)* [[LOC]]
  // CHECK-DAG: call spir_func void [[LOC_TMPL:@[a-zA-Z0-9_]+]](i32 addrspace(3)* [[LOC_LOAD5]])
  tmpl(PRIV);
  // CHECK-DAG: [[PRIV_LOAD5:%[a-zA-Z0-9]+]] = load i32*, i32* addrspace(4)* [[PRIV]]
  // CHECK-DAG: call spir_func void [[PRIV_TMPL:@[a-zA-Z0-9_]+]](i32* [[PRIV_LOAD5]])
  tmpl(NoAS);
  // CHECK-DAG: [[NoAS_LOAD5:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* [[NoAS]]
  // CHECK-DAG: call spir_func void [[GEN_TMPL:@[a-zA-Z0-9_]+]](i32 addrspace(4)* [[NoAS_LOAD5]])
}

// CHECK-DAG: define linkonce_odr spir_func void [[GLOB_TMPL]](i32 addrspace(1)* %
// CHECK-DAG: define linkonce_odr spir_func void [[GLOBDEV_TMPL]](i32 addrspace(5)* %
// CHECK-DAG: define linkonce_odr spir_func void [[GLOBHOST_TMPL]](i32 addrspace(6)* %
// CHECK-DAG: define linkonce_odr spir_func void [[LOC_TMPL]](i32 addrspace(3)* %
// CHECK-DAG: define linkonce_odr spir_func void [[PRIV_TMPL]](i32* %
// CHECK-DAG: define linkonce_odr spir_func void [[GEN_TMPL]](i32 addrspace(4)* %

void usages2() {
  __attribute__((opencl_private)) int *PRIV;
  // CHECK-DAG: [[PRIV:%[a-zA-Z0-9_]+]] = alloca i32*
  __attribute__((opencl_global)) int *GLOB;
  // CHECK-DAG: [[GLOB:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(1)*
  __attribute__((opencl_global_device)) int *GLOBDEV;
  // CHECK-DAG: [[GLOBDEV:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(5)*
  __attribute__((opencl_global_host)) int *GLOBHOST;
  // CHECK-DAG: [[GLOBHOST:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(6)*
  __attribute__((opencl_constant)) int *CONST;
  // CHECK-DAG: [[CONST:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(2)*
  __attribute__((opencl_local)) int *LOCAL;
  // CHECK-DAG: [[LOCAL:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(3)*

  bar(*PRIV);
  // CHECK-DAG: [[PRIV_LOAD:%[a-zA-Z0-9]+]] = load i32*, i32* addrspace(4)* [[PRIV]]
  // CHECK-DAG: [[PRIV_ASCAST:%[a-zA-Z0-9]+]] = addrspacecast i32* [[PRIV_LOAD]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* align 4 dereferenceable(4) [[PRIV_ASCAST]])
  bar(*GLOB);
  // CHECK-DAG: [[GLOB_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* [[GLOB]]
  // CHECK-DAG: [[GLOB_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* align 4 dereferenceable(4) [[GLOB_CAST]])
  bar(*GLOBDEV);
  // CHECK-DAG: [[GLOBDEV_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(5)*, i32 addrspace(5)* addrspace(4)* [[GLOBDEV]]
  // CHECK-DAG: [[GLOBDEV_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(5)* [[GLOBDEV_LOAD]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* align 4 dereferenceable(4) [[GLOBDEV_CAST]])
  bar(*GLOBHOST);
  // CHECK-DAG: [[GLOBHOST_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(6)*, i32 addrspace(6)* addrspace(4)* [[GLOBHOST]]
  // CHECK-DAG: [[GLOBHOST_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(6)* [[GLOBHOST_LOAD]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* align 4 dereferenceable(4) [[GLOBHOST_CAST]])
  bar2(*LOCAL);
  // CHECK-DAG: [[LOCAL_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(4)* [[LOCAL]]
  // CHECK-DAG: [[LOCAL_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(3)* [[LOCAL_LOAD]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_REF2]](i32 addrspace(4)* align 4 dereferenceable(4) [[LOCAL_CAST]])
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}
int main() {
  kernel_single_task<class fake_kernel>([]() { usages();usages2(); });
  return 0;
}
