// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -ast-dump -Wno-sycl-2017-compat %s | FileCheck %s

// Tests for AST of Intel FPGA [[intel::use_stall_enable_clusters]] function attribute on Host.

// Test attribute is presented on function.
[[intel::use_stall_enable_clusters]] void test() {}
// CHECK: FunctionDecl{{.*}}test 'void ()'
// CHECK: SYCLIntelUseStallEnableClustersAttr

// Test attribute is presented on function call operator (of a function object).
struct FuncObj {
  [[intel::use_stall_enable_clusters]] void operator()() const {}
  // CHECK: CXXRecordDecl{{.*}}implicit struct FuncObj
  // CHECK-NEXT: CXXMethodDecl{{.*}}operator() 'void () const'
  // CHECK-NEXT-NEXT:SYCLIntelUseStallEnableClustersAttr
};

// Test attribute is presented on lambda function.
void test3() {
  auto lambda = []() [[intel::use_stall_enable_clusters]]{};
  lambda();
  // CHECK: FunctionDecl{{.*}}test3 'void ()'
  // CHECK: LambdaExpr
  // CHECK: SYCLIntelUseStallEnableClustersAttr
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

class KernelFunctor {
public:
  [[intel::use_stall_enable_clusters]] void operator()() const {}
};

void foo() {
  // Test attribute is presented on function metadata.
  KernelFunctor f;
  kernel<class kernel_name_1>(f);
  // CHECK: FunctionDecl{{.*}}used kernel 'void (const KernelFunctor &)'
  // CHECK: CXXRecordDecl{{.*}}implicit class KernelFunctor
  // CHECK: CXXMethodDecl{{.*}}used operator() 'void () const'
  // CHECK-NEXT: CompoundStmt{{.*}}
  // CHECK-NEXT-NEXT: SYCLIntelUseStallEnableClustersAttr
}
