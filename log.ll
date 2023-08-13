; ModuleID = '/export/users/wenwan/src/sycl-ws/clang/test/CodeGenCXX/amdgcn-func-arg.cpp'
source_filename = "/export/users/wenwan/src/sycl-ws/clang/test/CodeGenCXX/amdgcn-func-arg.cpp"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8"
target triple = "amdgcn"

%class.A = type { i32 }
%class.B = type { [100 x i32] }

$_ZN1AC1Ev = comdat any

$_ZN1AD1Ev = comdat any

$_ZN1AC2Ev = comdat any

$_ZN1AD2Ev = comdat any

@g_a = addrspace(1) global %class.A zeroinitializer, align 4
@__dso_handle = external hidden addrspace(1) global i8
@g_b = addrspace(1) global %class.B zeroinitializer, align 4
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_amdgcn_func_arg.cpp, ptr null }]

; Function Attrs: noinline nounwind
define internal void @__cxx_global_var_init() #0 {
entry:
  call void @_ZN1AC1Ev(ptr noundef nonnull align 4 dereferenceable(4) addrspacecast (ptr addrspace(1) @g_a to ptr))
  %0 = call i32 @__cxa_atexit(ptr @_ZN1AD1Ev, ptr addrspacecast (ptr addrspace(1) @g_a to ptr), ptr addrspacecast (ptr addrspace(1) @__dso_handle to ptr)) #2
  ret void
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr void @_ZN1AC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %this) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8, addrspace(5)
  %this.addr.ascast = addrspacecast ptr addrspace(5) %this.addr to ptr
  store ptr %this, ptr %this.addr.ascast, align 8
  %this1 = load ptr, ptr %this.addr.ascast, align 8
  call void @_ZN1AC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this1)
  ret void
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr void @_ZN1AD1Ev(ptr noundef nonnull align 4 dereferenceable(4) %this) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8, addrspace(5)
  %this.addr.ascast = addrspacecast ptr addrspace(5) %this.addr to ptr
  store ptr %this, ptr %this.addr.ascast, align 8
  %this1 = load ptr, ptr %this.addr.ascast, align 8
  call void @_ZN1AD2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this1) #2
  ret void
}

; Function Attrs: nounwind
declare i32 @__cxa_atexit(ptr, ptr, ptr) #2

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z22func_with_indirect_arg1A(ptr addrspace(5) noundef %a) #3 {
entry:
  %a.indirect_addr = alloca ptr, align 8, addrspace(5)
  %p = alloca ptr, align 8, addrspace(5)
  %a.indirect_addr.ascast = addrspacecast ptr addrspace(5) %a.indirect_addr to ptr
  %p.ascast = addrspacecast ptr addrspace(5) %p to ptr
  %a.ascast = addrspacecast ptr addrspace(5) %a to ptr
  store ptr %a.ascast, ptr %a.indirect_addr.ascast, align 8
  store ptr %a.ascast, ptr %p.ascast, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z22test_indirect_arg_autov() #3 {
entry:
  %a = alloca %class.A, align 4, addrspace(5)
  %agg.tmp = alloca %class.A, align 4, addrspace(5)
  %a.ascast = addrspacecast ptr addrspace(5) %a to ptr
  %agg.tmp.ascast = addrspacecast ptr addrspace(5) %agg.tmp to ptr
  call void @_ZN1AC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %a.ascast)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp.ascast, ptr align 4 %a.ascast, i64 4, i1 false)
  %agg.tmp.ascast.ascast = addrspacecast ptr %agg.tmp.ascast to ptr addrspace(5)
  call void @_Z22func_with_indirect_arg1A(ptr addrspace(5) noundef %agg.tmp.ascast.ascast)
  call void @_ZN1AD1Ev(ptr noundef nonnull align 4 dereferenceable(4) %agg.tmp.ascast) #2
  call void @_Z17func_with_ref_argR1A(ptr noundef nonnull align 4 dereferenceable(4) %a.ascast)
  call void @_ZN1AD1Ev(ptr noundef nonnull align 4 dereferenceable(4) %a.ascast) #2
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #4

declare void @_Z17func_with_ref_argR1A(ptr noundef nonnull align 4 dereferenceable(4)) #5

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z24test_indirect_arg_globalv() #3 {
entry:
  %agg.tmp = alloca %class.A, align 4, addrspace(5)
  %agg.tmp.ascast = addrspacecast ptr addrspace(5) %agg.tmp to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp.ascast, ptr align 4 addrspacecast (ptr addrspace(1) @g_a to ptr), i64 4, i1 false)
  %agg.tmp.ascast.ascast = addrspacecast ptr %agg.tmp.ascast to ptr addrspace(5)
  call void @_Z22func_with_indirect_arg1A(ptr addrspace(5) noundef %agg.tmp.ascast.ascast)
  call void @_ZN1AD1Ev(ptr noundef nonnull align 4 dereferenceable(4) %agg.tmp.ascast) #2
  call void @_Z17func_with_ref_argR1A(ptr noundef nonnull align 4 dereferenceable(4) addrspacecast (ptr addrspace(1) @g_a to ptr))
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z19func_with_byval_arg1B(ptr addrspace(5) noundef byref(%class.B) align 4 %0) #3 {
entry:
  %coerce = alloca %class.B, align 4, addrspace(5)
  %p = alloca ptr, align 8, addrspace(5)
  %b = addrspacecast ptr addrspace(5) %coerce to ptr
  %p.ascast = addrspacecast ptr addrspace(5) %p to ptr
  call void @llvm.memcpy.p0.p5.i64(ptr align 4 %b, ptr addrspace(5) align 4 %0, i64 400, i1 false)
  store ptr %b, ptr %p.ascast, align 8
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p5.i64(ptr noalias nocapture writeonly, ptr addrspace(5) noalias nocapture readonly, i64, i1 immarg) #4

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z19test_byval_arg_autov() #3 {
entry:
  %b = alloca %class.B, align 4, addrspace(5)
  %agg.tmp = alloca %class.B, align 4, addrspace(5)
  %b.ascast = addrspacecast ptr addrspace(5) %b to ptr
  %agg.tmp.ascast = addrspacecast ptr addrspace(5) %agg.tmp to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp.ascast, ptr align 4 %b.ascast, i64 400, i1 false)
  %agg.tmp.ascast.ascast = addrspacecast ptr %agg.tmp.ascast to ptr addrspace(5)
  call void @_Z19func_with_byval_arg1B(ptr addrspace(5) noundef byref(%class.B) align 4 %agg.tmp.ascast.ascast)
  call void @_Z17func_with_ref_argR1B(ptr noundef nonnull align 4 dereferenceable(400) %b.ascast)
  ret void
}

declare void @_Z17func_with_ref_argR1B(ptr noundef nonnull align 4 dereferenceable(400)) #5

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z21test_byval_arg_globalv() #3 {
entry:
  %agg.tmp = alloca %class.B, align 4, addrspace(5)
  %agg.tmp.ascast = addrspacecast ptr addrspace(5) %agg.tmp to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp.ascast, ptr align 4 addrspacecast (ptr addrspace(1) @g_b to ptr), i64 400, i1 false)
  %agg.tmp.ascast.ascast = addrspacecast ptr %agg.tmp.ascast to ptr addrspace(5)
  call void @_Z19func_with_byval_arg1B(ptr addrspace(5) noundef byref(%class.B) align 4 %agg.tmp.ascast.ascast)
  call void @_Z17func_with_ref_argR1B(ptr noundef nonnull align 4 dereferenceable(400) addrspacecast (ptr addrspace(1) @g_b to ptr))
  ret void
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr void @_ZN1AC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8, addrspace(5)
  %this.addr.ascast = addrspacecast ptr addrspace(5) %this.addr to ptr
  store ptr %this, ptr %this.addr.ascast, align 8
  %this1 = load ptr, ptr %this.addr.ascast, align 8
  %x = getelementptr inbounds %class.A, ptr %this1, i32 0, i32 0
  store i32 0, ptr %x, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr void @_ZN1AD2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8, addrspace(5)
  %this.addr.ascast = addrspacecast ptr addrspace(5) %this.addr to ptr
  store ptr %this, ptr %this.addr.ascast, align 8
  %this1 = load ptr, ptr %this.addr.ascast, align 8
  ret void
}

; Function Attrs: noinline nounwind
define internal void @_GLOBAL__sub_I_amdgcn_func_arg.cpp() #0 {
entry:
  call void @__cxx_global_var_init()
  ret void
}

attributes #0 = { noinline nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nounwind }
attributes #3 = { mustprogress noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"amdgpu_code_object_version", i32 400}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!"clang version 18.0.0 (https://github.com/intel/llvm 40721c1e985aae37523ad64e92faeb6e12136c51)"}
