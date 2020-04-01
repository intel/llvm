// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s -check-prefixes CHECK-DEVICE,CHECK-BOTH
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s -check-prefixes CHECK-HOST,CHECK-BOTH

// CHECK-BOTH: @_ZZ15attrs_on_staticvE15static_numbanks = internal{{.*}}constant i32 20, align 4
// CHECK-DEVICE:  [[ANN_numbanks_4:@.str]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{numbanks:4}
// CHECK-BOTH: @_ZZ15attrs_on_staticvE15static_annotate = internal{{.*}}constant i32 30, align 4
// CHECK-BOTH:    [[ANN_annotate:@.str[.0-9]*]] = {{.*}}foobar
// CHECK-BOTH: @_ZZ15attrs_on_staticvE16static_force_p2d = internal{{.*}}constant i32 40, align 4
// CHECK-DEVICE:  [[ANN_force_pow2_depth_0:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{force_pow2_depth:0}
// CHECK-DEVICE:  [[ANN_register:@.str.[0-9]*]] = {{.*}}{register:1}
// CHECK-DEVICE:  [[ANN_memory_default:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}
// CHECK-DEVICE:  [[ANN_mlab_sizeinfo_500:@.str.[0-9]*]] = {{.*}}{memory:MLAB}{sizeinfo:4,500}
// CHECK-DEVICE:  [[ANN_blockram_sizeinfo_10_2:@.str.[0-9]*]] = {{.*}}{memory:BLOCK_RAM}{sizeinfo:4,10,2}
// CHECK-DEVICE:  [[ANN_bankwidth_4:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{bankwidth:4}
// CHECK-DEVICE:  [[ANN_private_copies_8:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{private_copies:8}
// CHECK-DEVICE:  [[ANN_singlepump:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{pump:1}
// CHECK-DEVICE:  [[ANN_doublepump:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{pump:2}
// CHECK-DEVICE:  [[ANN_merge_depth:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{merge:foo:depth}
// CHECK-DEVICE:  [[ANN_merge_width:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{merge:bar:width}
// CHECK-DEVICE:  [[ANN_max_replicates_2:@.str.[0-9]*]] = {{.*}}{max_replicates:2}
// CHECK-DEVICE:  [[ANN_simple_dual_port:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{simple_dual_port:1}
// CHECK-DEVICE:  [[ANN_bankbits_4_5:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{numbanks:4}{bank_bits:4,5}
// CHECK-DEVICE:  [[ANN_bankbits_numbanks_mlab:@.str.[0-9]*]] = {{.*}}{memory:MLAB}{sizeinfo:4}{numbanks:8}{bank_bits:5,4,3}
// CHECK-DEVICE:  [[ANN_bankbits_bankwidth:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4,10,2}{bankwidth:16}{numbanks:2}{bank_bits:0}
// CHECK-DEVICE:  [[ANN_memory_blockram:@.str.[0-9]*]] = {{.*}}{memory:BLOCK_RAM}{sizeinfo:4}
// CHECK-DEVICE:  [[ANN_memory_mlab:@.str.[0-9]*]] = {{.*}}{memory:MLAB}{sizeinfo:4}
// CHECK-DEVICE:  [[ANN_force_pow2_depth_1:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{force_pow2_depth:1}
// CHECK-DEVICE:  [[ANN_private_copies_4:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{private_copies:4}
// CHECK-DEVICE:  [[ANN_max_replicates_4:@.str.[0-9]*]] = {{.*}}{max_replicates:4}

// CHECK-BOTH: @llvm.global.annotations
// CHECK-DEVICE-SAME: { i8* addrspacecast (i8 addrspace(1)* bitcast (i32 addrspace(1)* @_ZZ15attrs_on_staticvE15static_numbanks to i8 addrspace(1)*) to i8*)
// CHECK-DEVICE-SAME: [[ANN_numbanks_4]]{{.*}} i32 43
// CHECK-DEVICE-SAME: { i8* addrspacecast (i8 addrspace(1)* bitcast (i32 addrspace(1)* @_ZZ15attrs_on_staticvE15static_annotate to i8 addrspace(1)*) to i8*)
// CHECK-HOST-SAME: { i8* bitcast (i32* @_ZZ15attrs_on_staticvE15static_annotate to i8*)
// CHECK-BOTH-SAME: [[ANN_annotate]]{{.*}} i32 44
// CHECK-DEVICE-SAME: { i8* addrspacecast (i8 addrspace(1)* bitcast (i32 addrspace(1)* @_ZZ15attrs_on_staticvE16static_force_p2d to i8 addrspace(1)*) to i8*)
// CHECK-DEVICE-SAME: [[ANN_force_pow2_depth_0]]{{.*}} i32 45
// CHECK-HOST-NOT: llvm.var.annotation
// CHECK-HOST-NOT: llvm.ptr.annotation

void attrs_on_static() {
  const static int static_numbanks [[intelfpga::numbanks(4)]] = 20;
  const static int static_annotate [[clang::annotate("foobar")]] = 30;
  const static int static_force_p2d [[intelfpga::force_pow2_depth(0)]] = 40;
}

void attrs_on_var() {
  // CHECK-DEVICE: %[[VAR_NUMBANKS:[0-9]+]] = bitcast{{.*}}%numbanks
  // CHECK-DEVICE: %[[VAR_NUMBANKS1:numbanks[0-9]+]] = bitcast{{.*}}%numbanks
  // CHECK-DEVICE: @llvm.var.annotation{{.*}}%[[VAR_NUMBANKS1]],{{.*}}[[ANN_numbanks_4]]
  int numbanks [[intelfpga::numbanks(4)]];
  // CHECK-DEVICE: %[[VAR_REGISTER:[0-9]+]] = bitcast{{.*}}%reg
  // CHECK-DEVICE: %[[VAR_REGISTER1:reg[0-9]+]] = bitcast{{.*}}%reg
  // CHECK-DEVICE: @llvm.var.annotation{{.*}}%[[VAR_REGISTER1]],{{.*}}[[ANN_register]]
  int reg [[intelfpga::register]];
  // CHECK-DEVICE: %[[VAR_MEMORY:[0-9]+]] = bitcast{{.*}}%memory
  // CHECK-DEVICE: %[[VAR_MEMORY1:memory[0-9]+]] = bitcast{{.*}}%memory
  // CHECK-DEVICE: @llvm.var.annotation{{.*}}%[[VAR_MEMORY1]],{{.*}}[[ANN_memory_default]]
  int memory [[intelfpga::memory]];
  // CHECK-DEVICE: %[[VAR_SIZE_MLAB:[0-9]+]] = bitcast{{.*}}size_mlab
  // CHECK-DEVICE: %[[VAR_SIZE_MLAB1:size_mlab[0-9]+]] = bitcast{{.*}}size_mlab
  // CHECK-DEVICE: llvm.var.annotation{{.*}}%[[VAR_SIZE_MLAB1]],{{.*}}[[ANN_mlab_sizeinfo_500]]
  [[intelfpga::memory("MLAB")]] int size_mlab[500];
  // CHECK-DEVICE: %[[VAR_size_blockram:[0-9]+]] = bitcast{{.*}}size_blockram
  // CHECK-DEVICE: %[[VAR_size_blockram1:size_blockram[0-9]+]] = bitcast{{.*}}size_blockram
  // CHECK-DEVICE: llvm.var.annotation{{.*}}%[[VAR_size_blockram1]],{{.*}}[[ANN_blockram_sizeinfo_10_2]]
  [[intelfpga::memory("BLOCK_RAM")]] int size_blockram[10][2];
  // CHECK-DEVICE: %[[VAR_BANKWIDTH:[0-9]+]] = bitcast{{.*}}%bankwidth
  // CHECK-DEVICE: %[[VAR_BANKWIDTH1:bankwidth[a-z0-9]+]] = bitcast{{.*}}%bankwidth
  // CHECK-DEVICE: llvm.var.annotation{{.*}}%[[VAR_BANKWIDTH1]],{{.*}}[[ANN_bankwidth_4]]
  int bankwidth [[intelfpga::bankwidth(4)]];
  // CHECK-DEVICE: %[[VAR_PRIV_COPIES:[0-9]+]] = bitcast{{.*}}%priv_copies
  // CHECK-DEVICE: %[[VAR_PRIV_COPIES1:priv_copies[0-9]+]] = bitcast{{.*}}%priv_copies
  // CHECK-DEVICE: llvm.var.annotation{{.*}}%[[VAR_PRIV_COPIES1]],{{.*}}[[ANN_private_copies_8]]
  int priv_copies [[intelfpga::private_copies(8)]];
  // CHECK-DEVICE: %[[VAR_SINGLEPUMP:[0-9]+]] = bitcast{{.*}}%singlepump
  // CHECK-DEVICE: %[[VAR_SINGLEPUMP1:singlepump[0-9]+]] = bitcast{{.*}}%singlepump
  // CHECK-DEVICE: llvm.var.annotation{{.*}}%[[VAR_SINGLEPUMP1]],{{.*}}[[ANN_singlepump]]
  int singlepump [[intelfpga::singlepump]];
  // CHECK-DEVICE: %[[VAR_DOUBLEPUMP:[0-9]+]] = bitcast{{.*}}%doublepump
  // CHECK-DEVICE: %[[VAR_DOUBLEPUMP1:doublepump[0-9]+]] = bitcast{{.*}}%doublepump
  // CHECK-DEVICE: llvm.var.annotation{{.*}}%[[VAR_DOUBLEPUMP1]],{{.*}}[[ANN_doublepump]]
  int doublepump [[intelfpga::doublepump]];
  // CHECK-DEVICE: %[[VAR_MERGE_DEPTH:[0-9]+]] = bitcast{{.*}}%merge_depth
  // CHECK-DEVICE: %[[VAR_MERGE_DEPTH1:merge_depth[0-9]+]] = bitcast{{.*}}%merge_depth
  // CHECK-DEVICE: llvm.var.annotation{{.*}}%[[VAR_MERGE_DEPTH1]],{{.*}}[[ANN_merge_depth]]
  int merge_depth [[intelfpga::merge("foo", "depth")]];
  // CHECK-DEVICE: %[[VAR_MERGE_WIDTH:[0-9]+]] = bitcast{{.*}}%merge_width
  // CHECK-DEVICE: %[[VAR_MERGE_WIDTH1:merge_width[0-9]+]] = bitcast{{.*}}%merge_width
  // CHECK-DEVICE: llvm.var.annotation{{.*}}%[[VAR_MERGE_WIDTH1]],{{.*}}[[ANN_merge_width]]
  int merge_width [[intelfpga::merge("bar", "width")]];
  // CHECK-DEVICE: %[[VAR_MAXREPL:[0-9]+]] = bitcast{{.*}}%max_repl
  // CHECK-DEVICE: %[[VAR_MAXREPL1:max_repl[0-9]+]] = bitcast{{.*}}%max_repl
  // CHECK-DEVICE: llvm.var.annotation{{.*}}%[[VAR_MAXREPL1]],{{.*}}[[ANN_max_replicates_2]]
  int max_repl [[intelfpga::max_replicates(2)]];
  // CHECK-DEVICE: %[[VAR_DUALPORT:[0-9]+]] = bitcast{{.*}}%dualport
  // CHECK-DEVICE: %[[VAR_DUALPORT1:dualport[0-9]+]] = bitcast{{.*}}%dualport
  // CHECK-DEVICE: llvm.var.annotation{{.*}}%[[VAR_DUALPORT1]],{{.*}}[[ANN_simple_dual_port]]
  int dualport [[intelfpga::simple_dual_port]];
  // CHECK-DEVICE: %[[VAR_BANKBITS:[0-9]+]] = bitcast{{.*}}%bankbits
  // CHECK-DEVICE: %[[VAR_BANKBITS1:bankbits[0-9]+]] = bitcast{{.*}}%bankbits
  // CHECK-DEVICE: @llvm.var.annotation{{.*}}%[[VAR_BANKBITS1]],{{.*}}[[ANN_bankbits_4_5]]
  int bankbits [[intelfpga::bank_bits(4,5)]];
  // CHECK-DEVICE: %[[VAR_BANKBITS_NUMBANKS:[0-9]+]] = bitcast{{.*}}%bankbits_numbanks_mlab
  // CHECK-DEVICE: %[[VAR_BANKBITS_NUMBANKS1:bankbits_numbanks_mlab[0-9]+]] = bitcast{{.*}}%bankbits_numbanks_mlab
  // CHECK-DEVICE: @llvm.var.annotation{{.*}}%[[VAR_BANKBITS_NUMBANKS1]],{{.*}}[[ANN_bankbits_numbanks_mlab]]
  [[intelfpga::bank_bits(5,4,3), intelfpga::numbanks(8), intelfpga::memory("MLAB")]] int bankbits_numbanks_mlab;
  // CHECK-DEVICE: %[[VAR_BANK_BITS_WIDTH:[0-9]+]] = bitcast{{.*}}%bank_bits_width
  // CHECK-DEVICE: %[[VAR_BANK_BITS_WIDTH1:bank_bits_width[0-9]+]] = bitcast{{.*}}%bank_bits_width
  // CHECK-DEVICE: @llvm.var.annotation{{.*}}%[[VAR_BANK_BITS_WIDTH1]],{{.*}}[[ANN_bankbits_bankwidth]]
  [[intelfpga::bank_bits(0), intelfpga::bankwidth(16)]] int bank_bits_width[10][2];
  // CHECK-DEVICE: %[[VAR_FP2D:[0-9]+]] = bitcast{{.*}}%force_p2d
  // CHECK-DEVICE: %[[VAR_FP2D1:force_p2d[0-9]+]] = bitcast{{.*}}%force_p2d
  // CHECK-DEVICE: llvm.var.annotation{{.*}}%[[VAR_FP2D1]],{{.*}}[[ANN_force_pow2_depth_0]]
  int force_p2d [[intelfpga::force_pow2_depth(0)]];
}

void attrs_on_struct() {
  struct attrs_on_struct {
    int numbanks [[intelfpga::numbanks(4)]] ;
    int reg [[intelfpga::register]];
    int memory [[intelfpga::memory]];
    int memory_blockram [[intelfpga::memory("BLOCK_RAM")]];
    int memory_mlab [[intelfpga::memory("MLAB")]];
    int bankwidth [[intelfpga::bankwidth(4)]];
    int privatecopies [[intelfpga::private_copies(8)]];
    int singlepump [[intelfpga::singlepump]];
    int doublepump [[intelfpga::doublepump]];
    int merge_depth [[intelfpga::merge("foo", "depth")]];
    int merge_width [[intelfpga::merge("bar", "width")]];
    int maxreplicates [[intelfpga::max_replicates(2)]];
    int dualport [[intelfpga::simple_dual_port]];
    int bankbits [[intelfpga::bank_bits(4, 5)]];
    int force_p2d [[intelfpga::force_pow2_depth(1)]];
  } s;

  // CHECK-DEVICE: %[[FIELD_NUMBANKS:.*]] = getelementptr inbounds %struct.{{.*}}.attrs_on_struct{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_NUMBANKS]]{{.*}}[[ANN_numbanks_4]]
  s.numbanks = 0;
  // CHECK-DEVICE: %[[FIELD_REGISTER:.*]] = getelementptr inbounds %struct.{{.*}}.attrs_on_struct{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_REGISTER]]{{.*}}[[ANN_register]]
  s.reg = 0;
  // CHECK-DEVICE: %[[FIELD_MEM_DEFAULT:.*]] = getelementptr inbounds %struct.{{.*}}.attrs_on_struct{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_MEM_DEFAULT]]{{.*}}[[ANN_memory_default]]
  s.memory = 0;
  // CHECK-DEVICE: %[[FIELD_MEM_BLOCKRAM:.*]] = getelementptr inbounds %struct.{{.*}}.attrs_on_struct{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_MEM_BLOCKRAM]]{{.*}}[[ANN_memory_blockram]]
  s.memory_blockram = 0;
  // CHECK-DEVICE: %[[FIELD_MEM_MLAB:.*]] = getelementptr inbounds %struct.{{.*}}.attrs_on_struct{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_MEM_MLAB]]{{.*}}[[ANN_memory_mlab]]
  s.memory_mlab = 0;
  // CHECK-DEVICE: %[[FIELD_BANKWIDTH:.*]] = getelementptr inbounds %struct.{{.*}}.attrs_on_struct{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_BANKWIDTH]]{{.*}}[[ANN_bankwidth_4]]
  s.bankwidth = 0;
  // CHECK-DEVICE: %[[FIELD_PRIV_COPIES:.*]] = getelementptr inbounds %struct.{{.*}}.attrs_on_struct{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_PRIV_COPIES]]{{.*}}[[ANN_private_copies_8]]
  s.privatecopies = 0;
  // CHECK-DEVICE: %[[FIELD_SINGLEPUMP:.*]] = getelementptr inbounds %struct.{{.*}}.attrs_on_struct{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_SINGLEPUMP]]{{.*}}[[ANN_singlepump]]
  s.singlepump = 0;
  // CHECK-DEVICE: %[[FIELD_DOUBLEPUMP:.*]] = getelementptr inbounds %struct.{{.*}}.attrs_on_struct{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_DOUBLEPUMP]]{{.*}}[[ANN_doublepump]]
  s.doublepump = 0;
  // CHECK-DEVICE: %[[FIELD_MERGE_DEPTH:.*]] = getelementptr inbounds %struct.{{.*}}.attrs_on_struct{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_MERGE_DEPTH]]{{.*}}[[ANN_merge_depth]]
  s.merge_depth = 0;
  // CHECK-DEVICE: %[[FIELD_MERGE_WIDTH:.*]] = getelementptr inbounds %struct.{{.*}}.attrs_on_struct{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_MERGE_WIDTH]]{{.*}}[[ANN_merge_width]]
  s.merge_width = 0;
  // CHECK-DEVICE: %[[FIELD_MAX_REPLICATES:.*]] = getelementptr inbounds %struct.{{.*}}.attrs_on_struct{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_MAX_REPLICATES]]{{.*}}[[ANN_max_replicates_2]]
  s.maxreplicates = 0;
  // CHECK-DEVICE: %[[FIELD_DUALPORT:.*]] = getelementptr inbounds %struct.{{.*}}.attrs_on_struct{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_DUALPORT]]{{.*}}[[ANN_simple_dual_port]]
  s.dualport = 0;
  // CHECK-DEVICE: %[[FIELD_BANKBITS:.*]] = getelementptr inbounds %struct.{{.*}}.attrs_on_struct{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_BANKBITS]]{{.*}}[[ANN_bankbits_4_5]]
  s.bankbits = 0;
  // CHECK-DEVICE: %[[FIELD_FP2D:.*]] = getelementptr inbounds %struct.{{.*}}.attrs_on_struct{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_FP2D]]{{.*}}[[ANN_force_pow2_depth_1]]
  s.force_p2d = 0;
}

// CHECK-HOST-NOT: llvm.var.annotation
// CHECK-HOST-NOT: llvm.ptr.annotation

template <int A, int B, int C>
void attrs_with_template_param() {
  // CHECK-DEVICE: %[[TEMPL_NUMBANKS:numbanks[0-9]+]] = bitcast{{.*}}%numbanks
  // CHECK-DEVICE: @llvm.var.annotation{{.*}}%[[TEMPL_NUMBANKS]],{{.*}}[[ANN_numbanks_4]]
  int numbanks [[intelfpga::numbanks(A)]];
  // CHECK-DEVICE: %[[TEMPL_BANKWIDTH:bankwidth[a-z0-9]+]] = bitcast{{.*}}%bankwidth
  // CHECK-DEVICE: llvm.var.annotation{{.*}}%[[TEMPL_BANKWIDTH]],{{.*}}[[ANN_bankwidth_4]]
  int bankwidth [[intelfpga::bankwidth(A)]];
  // CHECK-DEVICE: %[[TEMPL_PRIV_COPIES:priv_copies[0-9]+]] = bitcast{{.*}}%priv_copies
  // CHECK-DEVICE: llvm.var.annotation{{.*}}%[[TEMPL_PRIV_COPIES]],{{.*}}[[ANN_private_copies_4]]
  int priv_copies [[intelfpga::private_copies(A)]];
  // CHECK-DEVICE: %[[TEMPL_MAXREPL:max_repl[0-9]+]] = bitcast{{.*}}%max_repl
  // CHECK-DEVICE: llvm.var.annotation{{.*}}%[[TEMPL_MAXREPL]],{{.*}}[[ANN_max_replicates_4]]
  int max_repl [[intelfpga::max_replicates(A)]];
  // CHECK-DEVICE: %[[TEMPL_BANKBITS:bankbits[0-9]+]] = bitcast{{.*}}%bankbits
  // CHECK-DEVICE: @llvm.var.annotation{{.*}}%[[TEMPL_BANKBITS]],{{.*}}[[ANN_bankbits_4_5]]
  int bankbits [[intelfpga::bank_bits(A, B)]];
  // CHECK-DEVICE: %[[TEMPL_FP2D:force_p2d[0-9]+]] = bitcast{{.*}}%force_p2d
  // CHECK-DEVICE: @llvm.var.annotation{{.*}}%[[TEMPL_FP2D]]{{.*}}[[ANN_force_pow2_depth_1]]
  int force_p2d [[intelfpga::force_pow2_depth(C)]];

  struct templ_on_struct_fields {
    int numbanks [[intelfpga::numbanks(A)]] ;
    int bankwidth [[intelfpga::bankwidth(A)]];
    int privatecopies [[intelfpga::private_copies(A)]];
    int maxreplicates [[intelfpga::max_replicates(A)]];
    int bankbits [[intelfpga::bank_bits(A, B)]];
    int force_p2d [[intelfpga::force_pow2_depth(C)]];
  } s;

  // CHECK-DEVICE: %[[FIELD_NUMBANKS:.*]] = getelementptr inbounds %struct.{{.*}}.templ_on_struct_fields{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_NUMBANKS]]{{.*}}[[ANN_numbanks_4]]
  s.numbanks = 0;
  // CHECK-DEVICE: %[[FIELD_BANKWIDTH:.*]] = getelementptr inbounds %struct.{{.*}}.templ_on_struct_fields{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_BANKWIDTH]]{{.*}}[[ANN_bankwidth_4]]
  s.bankwidth = 0;
  // CHECK-DEVICE: %[[FIELD_PRIV_COPIES:.*]] = getelementptr inbounds %struct.{{.*}}.templ_on_struct_fields{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_PRIV_COPIES]]{{.*}}[[ANN_private_copies_4]]
  s.privatecopies = 0;
  // CHECK-DEVICE: %[[FIELD_MAX_REPLICATES:.*]] = getelementptr inbounds %struct.{{.*}}.templ_on_struct_fields{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_MAX_REPLICATES]]{{.*}}[[ANN_max_replicates_4]]
  s.maxreplicates = 0;
  // CHECK-DEVICE: %[[FIELD_BANKBITS:.*]] = getelementptr inbounds %struct.{{.*}}.templ_on_struct_fields{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_BANKBITS]]{{.*}}[[ANN_bankbits_4_5]]
  s.bankbits = 0;
  // CHECK-DEVICE: %[[FIELD_FP2D:.*]] = getelementptr inbounds %struct.{{.*}}.templ_on_struct_fields{{.*}}
  // CHECK-DEVICE: call i32* @llvm.ptr.annotation.p0i32{{.*}}%[[FIELD_FP2D]]{{.*}}[[ANN_force_pow2_depth_1]]
  s.force_p2d = 0;
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
  // CHECK-DEVICE: define internal {{.*}} @_ZZ20field_addrspace_castvEN5stateC2Ev
  // CHECK-DEVICE: %[[MEM:[a-zA-Z0-9]+]] = getelementptr inbounds %{{.*}}, %struct._ZTSZ20field_addrspace_castvE5state.state addrspace(4)* %{{.*}}, i32 0, i32 0
  // CHECK-DEVICE: %[[BITCAST:[0-9]+]] = bitcast [8 x i32] addrspace(4)* %[[MEM]] to i8 addrspace(4)*
  // CHECK-DEVICE: %[[ANN:[0-9]+]] = call i8 addrspace(4)* @llvm.ptr.annotation.p4i8(i8 addrspace(4)* %[[BITCAST]], {{.*}}, {{.*}})
  // CHECK-DEVICE: %{{[0-9]+}} = bitcast i8 addrspace(4)* %[[ANN]] to [8 x i32] addrspace(4)
  state_var.mem[0] = 42;
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    attrs_on_static();
    attrs_on_var();
    attrs_on_struct();
    field_addrspace_cast();
    attrs_with_template_param<4, 5, 1>();
  });
  return 0;
}
