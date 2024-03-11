; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_variable_length_array
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: noinline nounwind optnone uwtable
define weak dso_local spir_kernel void @K(ptr addrspace(1) %S.ul.GEP.1) local_unnamed_addr #0 {
newFuncRoot:
  %.ascast1 = addrspacecast ptr addrspace(1) %S.ul.GEP.1 to ptr addrspace(4)
  %S.ul.GEP.1.addr = alloca ptr addrspace(4), align 8
  store ptr addrspace(4) %.ascast1, ptr %S.ul.GEP.1.addr, align 8
  %S.ul.GEP.1.value = load ptr addrspace(4), ptr %S.ul.GEP.1.addr, align 8
  %"$loop_ctr46" = alloca i64, align 8
  %"$loop_ctr50" = alloca i64, align 8
  %"$loop_ctr38" = alloca i64, align 8
  %"var$102" = alloca i64, align 8
  %"var$103" = alloca i32, align 4
  %temp = alloca i32, align 4
  br label %fallthru

; CHECK-LABEL: fallthru
; CHECK-LLVM: %"ascastB$val41" = alloca i32, i64 %div.3
fallthru:                          ; preds = %newFuncRoot
  %"$stacksave37" = call spir_func ptr @llvm.stacksave()
  %S.ul.GEP.1_fetch.194 = load i32, ptr addrspace(4) %S.ul.GEP.1.value, align 1
  %int_sext39 = sext i32 %S.ul.GEP.1_fetch.194 to i64
  %rel.39 = icmp sgt i32 0, %S.ul.GEP.1_fetch.194
  %slct.13 = select i1 %rel.39, i32 0, i32 %S.ul.GEP.1_fetch.194
  %int_sext40 = sext i32 %slct.13 to i64
  %mul.11 = mul nsw i64 %int_sext40, 4
  %div.3 = sdiv i64 %mul.11, 4
  %"ascastB$val41" = alloca i32, i64 %div.3, align 4
  store i64 1, ptr %"var$102", align 1
  store i32 %S.ul.GEP.1_fetch.194, ptr %temp, align 1
  store i32 1, ptr %"var$103", align 1
  %"ascastB$val_fetch.197" = load i32, ptr %temp, align 1
  %rel.40 = icmp slt i32 %"ascastB$val_fetch.197", 1
  br i1 %rel.40, label %bb270, label %bb269.preheader

; CHECK-LABEL: bb269
; CHECK-LLVM: %1 = getelementptr inbounds i32, ptr %"ascastB$val41", i64 %0
bb269:                                            ; preds = %bb269.preheader, %bb269
  %"var$102_fetch.202" = load i64, ptr %"var$102", align 1
  %0 = sub nsw i64 %"var$102_fetch.202", 1
  %1 = getelementptr inbounds i32, ptr %"ascastB$val41", i64 %0
  %add.17 = add nsw i64 %"var$102_fetch.202", 1
  store i64 %add.17, ptr %"var$102", align 1
  %"var$103_fetch.203" = load i32, ptr %"var$103", align 1
  %add.18 = add nsw i32 %"var$103_fetch.203", 1
  store i32 %add.18, ptr %"var$103", align 1
  %"var$103_fetch.204" = load i32, ptr %"var$103", align 1
  %"ascastB$val_fetch.205" = load i32, ptr %temp, align 1
  %rel.41 = icmp sle i32 %"var$103_fetch.204", %"ascastB$val_fetch.205"
  br i1 %rel.41, label %bb269, label %bb270.loopexit

bb270.loopexit:                                   ; preds = %bb269
  br label %bb270

bb270:                                            ; preds = %bb270.loopexit, %fallthru
  store i64 1, ptr %"$loop_ctr38", align 1
  br label %loop_test315

loop_test315:                                     ; preds = ,loop_body316, %bb270
  %"$loop_ctr_fetch.208" = load i64, ptr %"$loop_ctr38", align 1
  %rel.42 = icmp sle i64 %"$loop_ctr_fetch.208", %int_sext39
  br i1 %rel.42, label %loop_body316, label %loop_exit317

; CHECK-LABEL: loop_body316
; CHECK-LLVM: %3 = getelementptr inbounds i32, ptr %"ascastB$val41", i64 %2
loop_body316:                                     ; preds = %loop_test315
  %"$loop_ctr_fetch.206" = load i64, ptr %"$loop_ctr38", align 1
  %2 = sub nsw i64 %"$loop_ctr_fetch.206", 1
  %3 = getelementptr inbounds i32, ptr %"ascastB$val41", i64 %2
  %"ascastB$val[]_fetch.207" = load i32, ptr %3, align 1
  %"$loop_ctr_fetch.195" = load i64, ptr %"$loop_ctr38", align 1
  br label %loop_test315

loop_exit317:                                     ; preds = %loop_body316
  call spir_func void @llvm.stackrestore(ptr %"$stacksave37")
  %S.ul.GEP.1_fetch.210 = load i32, ptr addrspace(4) %S.ul.GEP.1.value, align 1
  %int_sext47 = sext i32 %S.ul.GEP.1_fetch.210 to i64
  store i64 1, ptr %"$loop_ctr46", align 1
  br label %loop_test323

loop_test323:                                     ; preds = %loop_body324, %loop_exit317
  %"$loop_ctr_fetch.212" = load i64, ptr %"$loop_ctr46", align 1
  %rel.45 = icmp sle i64 %"$loop_ctr_fetch.212", %int_sext47
  br i1 %rel.45, label %loop_body324, label %loop_exit325

loop_body324:                                     ; preds = %loop_test323
  %"$loop_ctr_fetch.211" = load i64, ptr %"$loop_ctr46", align 1
  br label %loop_test323

loop_exit325:                                     ; preds = %loop_test323
  %S.ul.GEP.1_fetch.214 = load i32, ptr addrspace(4) %S.ul.GEP.1.value, align 1
  %int_sext51 = sext i32 %S.ul.GEP.1_fetch.214 to i64
  store i64 1, ptr %"$loop_ctr50", align 1
  br label %loop_test327

loop_test327:                                     ; preds = %loop_body328, %loop_exit325
  %"$loop_ctr_fetch.216" = load i64, ptr %"$loop_ctr50", align 1
  %rel.48 = icmp sle i64 %"$loop_ctr_fetch.216", %int_sext51
  br i1 %rel.48, label %loop_body328, label %loop_exit329

loop_body328:                                     ; preds = %loop_test327
  %"$loop_ctr_fetch.215" = load i64, ptr %"$loop_ctr50", align 1
  br label %loop_test327

loop_exit329:                                     ; preds = %loop_test327
  ret void

bb269.preheader:                                  ; preds = %fallthru
  br label %bb269
}

; Function Attrs: nofree nosync nounwind willreturn mustprogress
declare void @llvm.stackrestore(ptr) #1

; Function Attrs: nofree nosync nounwind willreturn mustprogress
declare ptr @llvm.stacksave() #1

attributes #0 = { noinline nounwind optnone uwtable }
attributes #1 = { nofree nosync nounwind willreturn mustprogress }

