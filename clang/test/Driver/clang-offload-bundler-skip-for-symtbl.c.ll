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

@not_skipping = global i64 0
;CHECK: not_skipping

@__another_global = global i64 0
;CHECK: __another_global
