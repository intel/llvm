; Copyright (C) Codeplay Software Limited
;
; Licensed under the Apache License, Version 2.0 (the "License") with LLVM
; Exceptions; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     https://github.com/uxlfoundation/oneapi-construction-kit/blob/main/LICENSE.txt
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
; WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
; License for the specific language governing permissions and limitations
; under the License.
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: veczc -k foo -vecz-simd-width=4 -vecz-passes="function(break-crit-edges,lower-switch),mergereturn" -S < %s | FileCheck %s --check-prefix=FOO
; RUN: veczc -k bar -vecz-simd-width=4 -vecz-passes="function(break-crit-edges,lower-switch),mergereturn" -S < %s | FileCheck %s --check-prefix=BAR

; The pass did previously not report the correct Modified status in the case
; where a function had at most one return block, and an unified unreachable
; block was created. This was caught by the pass return status check that is
; hidden under EXPENSIVE_CHECKS.

; FOO-LABEL: define i32 @__vecz_v4_foo(
; FOO: for.foo.body2:
; FOO-NEXT: br label %UnifiedUnreachableBlock

; FOO: for.foo.end:
; FOO-NEXT: br label %UnifiedUnreachableBlock

; FOO: UnifiedUnreachableBlock:
; FOO-NEXT: unreachable

define i32 @foo() {
entry:
  br label %for.foo.cond

for.foo.cond:                                         ; preds = %entry
  br i1 false, label %for.foo.body, label %for.foo.end3

for.foo.body:                                         ; preds = %for.foo.cond
  br label %for.foo.cond1

for.foo.cond1:                                        ; preds = %for.foo.body
  br i1 false, label %for.foo.body2, label %for.foo.end

for.foo.body2:                                        ; preds = %for.foo.cond1
  unreachable

for.foo.end:                                          ; preds = %for.foo.cond1
  unreachable

for.foo.end3:                                         ; preds = %for.foo.cond
  ret i32 undef
}

; BAR-LABEL: define void @__vecz_v4_bar(
; BAR: for.bar.body2:
; BAR-NEXT: br label %UnifiedUnreachableBlock

; BAR: for.bar.end:
; BAR-NEXT: br label %UnifiedUnreachableBlock

; BAR: UnifiedUnreachableBlock:
; BAR-NEXT: unreachable

define void @bar() {
entry:
  br label %for.bar.cond

for.bar.cond:                                         ; preds = %entry
  br i1 false, label %for.bar.body, label %for.bar.end

for.bar.body:                                         ; preds = %for.bar.cond
  br label %for.bar.cond1

for.bar.cond1:                                        ; preds = %for.bar.body
  br i1 false, label %for.bar.body2, label %for.bar.end

for.bar.body2:                                        ; preds = %for.bar.cond1
  unreachable

for.bar.end:                                          ; preds = %for.bar.cond1
  unreachable
}
