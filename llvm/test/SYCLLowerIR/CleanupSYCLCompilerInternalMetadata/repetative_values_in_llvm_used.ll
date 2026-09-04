; RUN: opt --passes=cleanup-sycl-metadata-from-llvm-used %s

; Check that CleanupSYCLMetadataFromLLVMUsed pass considers the case
; when llvm.used has one value more than once. It used to perform a double free
; due to llvm::isSafeToDestroyConstant(C) for the provided @C returns true.

@C = linkonce_odr hidden addrspace(1) constant <{i64}> <{i64 0}>

@llvm.used = appending global [2 x ptr addrspace(2)] [ptr addrspace(2) addrspacecast (ptr addrspace(1) @C to ptr addrspace(2)), ptr addrspace(2) addrspacecast (ptr addrspace(1) @C to ptr addrspace(2))], section "llvm.metadata"

