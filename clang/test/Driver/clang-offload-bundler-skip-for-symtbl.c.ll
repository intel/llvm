@__AsanKernelMetadata = global i64 0
;CHECK-NOT: __AsanKernelMetadata
@__MsanKernelMetadata = global i64 0
;CHECK-NOT: __MsanKernelMetadata
@__TsanKernelMetadata = global i64 0
;CHECK-NOT: __TsanKernelMetadata
@__AsanDeviceGlobalMetadata = global i64 0
;CHECK-NOT: __AsanDeviceGlobalMetadata
@__MsanDeviceGlobalMetadata = global i64 0
;CHECK-NOT: __MsanDeviceGlobalMetadata
@__TsanDeviceGlobalMetadata = global i64 0
;CHECK-NOT: __TsanDeviceGlobalMetadata

; The sanitizer metadata globals carry a per-module unique id suffix, so the
; suffixed names must be skipped as well.
@__AsanKernelMetadata_0123456789abcdef0123456789abcdef = global i64 0
;CHECK-NOT: __AsanKernelMetadata_0123456789abcdef0123456789abcdef
@__MsanKernelMetadata_0123456789abcdef0123456789abcdef = global i64 0
;CHECK-NOT: __MsanKernelMetadata_0123456789abcdef0123456789abcdef
@__TsanKernelMetadata_0123456789abcdef0123456789abcdef = global i64 0
;CHECK-NOT: __TsanKernelMetadata_0123456789abcdef0123456789abcdef
@__AsanDeviceGlobalMetadata_0123456789abcdef0123456789abcdef = global i64 0
;CHECK-NOT: __AsanDeviceGlobalMetadata_0123456789abcdef0123456789abcdef
@__MsanDeviceGlobalMetadata_0123456789abcdef0123456789abcdef = global i64 0
;CHECK-NOT: __MsanDeviceGlobalMetadata_0123456789abcdef0123456789abcdef
@__TsanDeviceGlobalMetadata_0123456789abcdef0123456789abcdef = global i64 0
;CHECK-NOT: __TsanDeviceGlobalMetadata_0123456789abcdef0123456789abcdef

@not_skipping = global i64 0
;CHECK: not_skipping

@__another_global = global i64 0
;CHECK: __another_global
