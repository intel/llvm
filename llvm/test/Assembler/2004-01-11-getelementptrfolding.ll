; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | \
; RUN:   not grep "getelementptr.*getelementptr"
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

%struct.TTriangleItem = type { ptr, ptr, [3 x %struct.TUVVertex] }
%struct.TUVVertex = type { i16, i16, i16, i16 }
@data_triangleItems = internal constant [2908 x %struct.TTriangleItem] zeroinitializer; <ptr> [#uses=2]

define void @foo() {
        store i16 0, ptr getelementptr ([2908 x %struct.TTriangleItem], ptr @data_triangleItems, i64 0, i64 0, i32 2, i64 0, i32 0)
        ret void
}

