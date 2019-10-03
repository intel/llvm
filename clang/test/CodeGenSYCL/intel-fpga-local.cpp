// RUN: %clang_cc1 -x c++ -triple spir64-unknown-linux-sycldevice -std=c++11 -disable-llvm-passes -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

//CHECK: [[ANN1:@.str]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{numbanks:4}
//CHECK: [[ANN2:@.str.[0-9]*]] = {{.*}}{register:1}
//CHECK: [[ANN3:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}
//CHECK: [[ANN4:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{bankwidth:4}
//CHECK: [[ANN5:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{max_private_copies:8}
//CHECK: [[ANN10:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{pump:1}
//CHECK: [[ANN11:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{pump:2}
//CHECK: [[ANN12:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{merge:foo:depth}
//CHECK: [[ANN13:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{merge:bar:width}
//CHECK: [[ANN14:@.str.[0-9]*]] = {{.*}}{max_replicates:2}
//CHECK: [[ANN15:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{simple_dual_port:1}
//CHECK: [[ANN6:@.str.[0-9]*]] = {{.*}}{memory:BLOCK_RAM}{sizeinfo:4}
//CHECK: [[ANN7:@.str.[0-9]*]] = {{.*}}{memory:MLAB}{sizeinfo:4}
//CHECK: [[ANN8:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{bankwidth:8}
//CHECK: [[ANN9:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{max_private_copies:4}
//CHECK: [[ANN16:@.str.[0-9]*]] = {{.*}}foobar
//CHECK: [[ANN17:@.str.[0-9]*]] = {{.*}}{memory:MLAB}{sizeinfo:4,500}
//CHECK: [[ANN18:@.str.[0-9]*]] = {{.*}}{memory:BLOCK_RAM}{sizeinfo:4,10,2}

//CHECK: @llvm.global.annotations
//CHECK-SAME: { i8 addrspace(1)* bitcast (i32 addrspace(1)* @_ZZ3quxiE5a_one to i8 addrspace(1)*)
//CHECK-SAME: [[ANN1]]{{.*}}i32 154
//CHECK-SAME: { i8 addrspace(1)* bitcast (i32 addrspace(1)* @_ZZ3quxiE5b_two to i8 addrspace(1)*)
//CHECK-SAME: [[ANN16]]{{.*}}i32 158

void foo() {
  //CHECK: %[[VAR_ONE:[0-9]+]] = bitcast{{.*}}var_one
  //CHECK: %[[VAR_ONE1:var_one[0-9]+]] = bitcast{{.*}}var_one
  //CHECK: llvm.var.annotation{{.*}}%[[VAR_ONE1]],{{.*}}[[ANN1]]
  int var_one [[intelfpga::numbanks(4)]];
  //CHECK: %[[VAR_TWO:[0-9]+]] = bitcast{{.*}}var_two
  //CHECK: %[[VAR_TWO1:var_two[0-9]+]] = bitcast{{.*}}var_two
  //CHECK: llvm.var.annotation{{.*}}%[[VAR_TWO1]],{{.*}}[[ANN2]]
  int var_two [[intelfpga::register]];
  //CHECK: %[[VAR_THREE:[0-9]+]] = bitcast{{.*}}var_three
  //CHECK: %[[VAR_THREE1:var_three[0-9]+]] = bitcast{{.*}}var_three
  //CHECK: llvm.var.annotation{{.*}}%[[VAR_THREE1]],{{.*}}[[ANN3]]
  int var_three [[intelfpga::memory]];
  //CHECK: %[[VAR_FOUR:[0-9]+]] = bitcast{{.*}}var_four
  //CHECK: %[[VAR_FOUR1:var_four[0-9]+]] = bitcast{{.*}}var_four
  //CHECK: llvm.var.annotation{{.*}}%[[VAR_FOUR1]],{{.*}}[[ANN4]]
  int var_four [[intelfpga::bankwidth(4)]];
}

struct foo_two {
  int f1 [[intelfpga::numbanks(4)]];
  int f2 [[intelfpga::register]];
  int f3 [[intelfpga::memory]];
  int f4 [[intelfpga::bankwidth(4)]];
  int f5 [[intelfpga::max_private_copies(8)]];
  int f6 [[intelfpga::singlepump]];
  int f7 [[intelfpga::doublepump]];
  int f8 [[intelfpga::merge("foo", "depth")]];
  int f9 [[intelfpga::merge("bar", "width")]];
  int f10 [[intelfpga::max_replicates(2)]];
  int f11 [[intelfpga::simple_dual_port]];
};

void bar() {
  struct foo_two s1;
  //CHECK: %[[FIELD1:.*]] = getelementptr inbounds %struct.{{.*}}.foo_two{{.*}}
  //CHECK: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD1]]{{.*}}[[ANN1]]
  s1.f1 = 0;
  //CHECK: %[[FIELD2:.*]] = getelementptr inbounds %struct.{{.*}}.foo_two{{.*}}
  //CHECK: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD2]]{{.*}}[[ANN2]]
  s1.f2 = 0;
  //CHECK: %[[FIELD3:.*]] = getelementptr inbounds %struct.{{.*}}.foo_two{{.*}}
  //CHECK: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD3]]{{.*}}[[ANN3]]
  s1.f3 = 0;
  //CHECK: %[[FIELD4:.*]] = getelementptr inbounds %struct.{{.*}}.foo_two{{.*}}
  //CHECK: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD4]]{{.*}}[[ANN4]]
  s1.f4 = 0;
  //CHECK: %[[FIELD5:.*]] = getelementptr inbounds %struct.{{.*}}.foo_two{{.*}}
  //CHECK: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD5]]{{.*}}[[ANN5]]
  s1.f5 = 0;
  //CHECK: %[[FIELD6:.*]] = getelementptr inbounds %struct.{{.*}}.foo_two{{.*}}
  //CHECK: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD6]]{{.*}}[[ANN10]]
  s1.f6 = 0;
  //CHECK: %[[FIELD7:.*]] = getelementptr inbounds %struct.{{.*}}.foo_two{{.*}}
  //CHECK: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD7]]{{.*}}[[ANN11]]
  s1.f7 = 0;
  //CHECK: %[[FIELD8:.*]] = getelementptr inbounds %struct.{{.*}}.foo_two{{.*}}
  //CHECK: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD8]]{{.*}}[[ANN12]]
  s1.f8 = 0;
  //CHECK: %[[FIELD9:.*]] = getelementptr inbounds %struct.{{.*}}.foo_two{{.*}}
  //CHECK: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD9]]{{.*}}[[ANN13]]
  s1.f9 = 0;
  //CHECK: %[[FIELD10:.*]] = getelementptr inbounds %struct.{{.*}}.foo_two{{.*}}
  //CHECK: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD10]]{{.*}}[[ANN14]]
  s1.f10 = 0;
  //CHECK: %[[FIELD11:.*]] = getelementptr inbounds %struct.{{.*}}.foo_two{{.*}}
  //CHECK: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD11]]{{.*}}[[ANN15]]
  s1.f11 = 0;
}

void baz() {
  //CHECK: %[[V_ONE:[0-9]+]] = bitcast{{.*}}v_one
  //CHECK: %[[V_ONE1:v_one[0-9]+]] = bitcast{{.*}}v_one
  //CHECK: llvm.var.annotation{{.*}}%[[V_ONE1]],{{.*}}[[ANN1]]
  int v_one [[intelfpga::numbanks(4)]];
  //CHECK: %[[V_TWO:[0-9]+]] = bitcast{{.*}}v_two
  //CHECK: %[[V_TWO1:v_two[0-9]+]] = bitcast{{.*}}v_two
  //CHECK: llvm.var.annotation{{.*}}%[[V_TWO1]],{{.*}}[[ANN2]]
  int v_two [[intelfpga::register]];
  //CHECK: %[[V_THREE:[0-9]+]] = bitcast{{.*}}v_three
  //CHECK: %[[V_THREE1:v_three[0-9]+]] = bitcast{{.*}}v_three
  //CHECK: llvm.var.annotation{{.*}}%[[V_THREE1]],{{.*}}[[ANN3]]
  int v_three [[intelfpga::memory]];
  //CHECK: %[[V_FOUR:[0-9]+]] = bitcast{{.*}}v_four
  //CHECK: %[[V_FOUR1:v_four[0-9]+]] = bitcast{{.*}}v_four
  //CHECK: llvm.var.annotation{{.*}}%[[V_FOUR1]],{{.*}}[[ANN6]]
  int v_four [[intelfpga::memory("BLOCK_RAM")]];
  //CHECK: %[[V_FIVE:[0-9]+]] = bitcast{{.*}}v_five
  //CHECK: %[[V_FIVE1:v_five[0-9]+]] = bitcast{{.*}}v_five
  //CHECK: llvm.var.annotation{{.*}}%[[V_FIVE1]],{{.*}}[[ANN7]]
  int v_five [[intelfpga::memory("MLAB")]];
  //CHECK: %[[V_SIX:[0-9]+]] = bitcast{{.*}}v_six
  //CHECK: %[[V_SIX1:v_six[0-9]+]] = bitcast{{.*}}v_six
  //CHECK: llvm.var.annotation{{.*}}%[[V_SIX1]],{{.*}}[[ANN8]]
  int v_six [[intelfpga::bankwidth(8)]];
  //CHECK: %[[V_SEVEN:[0-9]+]] = bitcast{{.*}}v_seven
  //CHECK: %[[V_SEVEN1:v_seven[0-9]+]] = bitcast{{.*}}v_seven
  //CHECK: llvm.var.annotation{{.*}}%[[V_SEVEN1]],{{.*}}[[ANN9]]
  int v_seven [[intelfpga::max_private_copies(4)]];
  //CHECK: %[[V_EIGHT:[0-9]+]] = bitcast{{.*}}v_eight
  //CHECK: %[[V_EIGHT1:v_eight[0-9]+]] = bitcast{{.*}}v_eight
  //CHECK: llvm.var.annotation{{.*}}%[[V_EIGHT1]],{{.*}}[[ANN10]]
  int v_eight [[intelfpga::singlepump]];
  //CHECK: %[[V_NINE:[0-9]+]] = bitcast{{.*}}v_nine
  //CHECK: %[[V_NINE1:v_nine[0-9]+]] = bitcast{{.*}}v_nine
  //CHECK: llvm.var.annotation{{.*}}%[[V_NINE1]],{{.*}}[[ANN11]]
  int v_nine [[intelfpga::doublepump]];
  //CHECK: %[[V_TEN:[0-9]+]] = bitcast{{.*}}v_ten
  //CHECK: %[[V_TEN1:v_ten[0-9]+]] = bitcast{{.*}}v_ten
  //CHECK: llvm.var.annotation{{.*}}%[[V_TEN1]],{{.*}}[[ANN12]]
  int v_ten [[intelfpga::merge("foo", "depth")]];
  //CHECK: %[[V_ELEVEN:[0-9]+]] = bitcast{{.*}}v_eleven
  //CHECK: %[[V_ELEVEN1:v_eleven[0-9]+]] = bitcast{{.*}}v_eleven
  //CHECK: llvm.var.annotation{{.*}}%[[V_ELEVEN1]],{{.*}}[[ANN13]]
  int v_eleven [[intelfpga::merge("bar", "width")]];
  //CHECK: %[[V_TWELVE:[0-9]+]] = bitcast{{.*}}v_twelve
  //CHECK: %[[V_TWELVE1:v_twelve[0-9]+]] = bitcast{{.*}}v_twelve
  //CHECK: llvm.var.annotation{{.*}}%[[V_TWELVE1]],{{.*}}[[ANN14]]
  int v_twelve [[intelfpga::max_replicates(2)]];
  //CHECK: %[[V_THIRTEEN:[0-9]+]] = bitcast{{.*}}v_thirteen
  //CHECK: %[[V_THIRTEEN1:v_thirteen[0-9]+]] = bitcast{{.*}}v_thirteen
  //CHECK: llvm.var.annotation{{.*}}%[[V_THIRTEEN1]],{{.*}}[[ANN15]]
  int v_thirteen [[intelfpga::simple_dual_port]];
}

void qux(int a) {
  static int a_one [[intelfpga::numbanks(4)]];
  //CHECK: load{{.*}}a_one
  //CHECK: store{{.*}}a_one
  a_one = a_one + a;
  static int b_two [[clang::annotate("foobar")]];
  //CHECK: load{{.*}}b_two
  //CHECK: store{{.*}}b_two
  b_two = b_two + a;
}

void size_info() {
  //CHECK: %[[VAR_A:[0-9]+]] = bitcast{{.*}}var_a
  //CHECK: %[[VAR_A1:var_a[0-9]+]] = bitcast{{.*}}var_a
  //CHECK: llvm.var.annotation{{.*}}%[[VAR_A1]],{{.*}}[[ANN17]]
  [[intelfpga::memory("MLAB")]] int var_a[500];
  //CHECK: %[[VAR_B:[0-9]+]] = bitcast{{.*}}var_b
  //CHECK: %[[VAR_B1:var_b[0-9]+]] = bitcast{{.*}}var_b
  //CHECK: llvm.var.annotation{{.*}}%[[VAR_B1]],{{.*}}[[ANN18]]
  [[intelfpga::memory("BLOCK_RAM")]] int var_b[10][2];
}

void field_addrspace_cast() {
  struct state {
    [[intelfpga::numbanks(2)]] int mem[8];

    // The initialization code is not relevant to this example.
    // It prevents the compiler from optimizing away access to this struct.
    state() {
      for (auto i = 0; i < 8; i++) {
        mem[i] = i;
      }
    }
  } state_var;
  // CHECK: define internal {{.*}} @_ZZ20field_addrspace_castvEN5stateC2Ev
  // CHECK: %[[MEM:[a-zA-Z0-9]+]] = getelementptr inbounds %{{.*}}, %struct._ZTSZ20field_addrspace_castvE5state.state addrspace(4)* %{{.*}}, i32 0, i32 0
  // CHECK: %[[BITCAST:[0-9]+]] = bitcast [8 x i32] addrspace(4)* %[[MEM]] to i8 addrspace(4)*
  // CHECK: %[[ANN:[0-9]+]] = call i8 addrspace(4)* @llvm.ptr.annotation.p4i8(i8 addrspace(4)* %[[BITCAST]], {{.*}}, {{.*}})
  // CHECK: %{{[0-9]+}} = bitcast i8 addrspace(4)* %[[ANN]] to [8 x i32] addrspace(4)
  state_var.mem[0] = 42;
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    foo();
    bar();
    baz();
    qux(42);
    size_info();
    field_addrspace_cast();
  });
  return 0;
}
