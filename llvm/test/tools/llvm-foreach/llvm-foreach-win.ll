; REQUIRES: system-windows
;
; RUN: echo 'Content of first file' > %t1.tgt
; RUN: echo 'Content of second file' > %t2.tgt
; RUN: echo "%t1.tgt" > %t.list
; RUN: echo "%t2.tgt" >> %t.list
; RUN: llvm-foreach --in-replace="{}" --in-file-list=%t.list -- echo "{}" > %t.res
; RUN: FileCheck < %t.res %s
; CHECK: [[FIRST:.+1.tgt]]
; CHECK: [[SECOND:.+2.tgt]]
;
; RUN: llvm-foreach --in-replace="{}" --out-replace=%t --out-ext=out --in-file-list=%t.list --out-file-list=%t.out.list -- xcopy /y "{}" %t
; RUN: FileCheck < %t.out.list %s --check-prefix=CHECK-LIST
; CHECK-LIST: [[FIRST:.+\.out]]
; CHECK-LIST: [[SECOND:.+\.out]]
; RUN: llvm-foreach --in-replace="{}" --in-file-list=%t.out.list -- FileCheck --input-file="{}" %s --check-prefix=CHECK-CONTENT
; CHECK-CONTENT: Content of

; RUN: echo 'something' > %t3.tgt
; RUN: echo 'something again' > %t4.tgt
; RUN: echo "%t3.tgt" > %t1.list
; RUN: echo "%t4.tgt" >> %t1.list
; RUN: llvm-foreach --in-replace="{}" --in-replace="in" --in-file-list=%t.list --in-file-list=%t1.list -- echo -first-part-of-arg={}.out -first-part-of-arg=in.out > %t1.res
; RUN: FileCheck < %t1.res %s --check-prefix=CHECK-DOUBLE-LISTS
; CHECK-DOUBLE-LISTS: -first-part-of-arg=[[FIRST:.+1.tgt.out]] -first-part-of-arg=[[THIRD:.+3.tgt.out]]
; CHECK-DOUBLE-LISTS: -first-part-of-arg=[[SECOND:.+2.tgt.out]] -first-part-of-arg=[[FOURTH:.+4.tgt.out]]
