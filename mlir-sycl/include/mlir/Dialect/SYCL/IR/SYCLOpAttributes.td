//===-- SYCLOpsAttributes.td - SYCL dialect attribute definitions *tablegen===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_OP_ATTRIBUTES
#define SYCL_OP_ATTRIBUTES

include "mlir/IR/EnumAttr.td"

include "SYCLOps.td"

class SYCL_I32EnumAttr<string name, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::sycl";
}

////////////////////////////////////////////////////////////////////////////////
// AccessMode
////////////////////////////////////////////////////////////////////////////////

def ReadAccessMode
    : I32EnumAttrCase<"Read", 1024, "read">;
def WriteAccessMode
    : I32EnumAttrCase<"Write", 1025, "write">;
def ReadWriteAccessMode
    : I32EnumAttrCase<"ReadWrite", 1026, "read_write">;
def DiscardWriteAccessMode
    : I32EnumAttrCase<"DiscardWrite", 1027, "discard_write">;
def DiscardReadWriteAccessMode
    : I32EnumAttrCase<"DiscardReadWrite", 1028, "discard_read_write">;
def AtomicAccessMode
    : I32EnumAttrCase<"Atomic", 1029, "atomic">;

def MemoryAccessModeAttr
    : SYCL_I32EnumAttr<"AccessMode", "sycl.access_mode",
                       [ReadAccessMode, WriteAccessMode, ReadWriteAccessMode,
		        DiscardWriteAccessMode, DiscardReadWriteAccessMode,
			AtomicAccessMode]>;

////////////////////////////////////////////////////////////////////////////////
// Target
////////////////////////////////////////////////////////////////////////////////

def GlobalBufferTarget
    : I32EnumAttrCase<"GlobalBuffer", 2014, "global_buffer">;
def ConstantBufferTarget
    : I32EnumAttrCase<"ConstantBuffer", 2015, "constant_buffer">;
def LocalTarget
    : I32EnumAttrCase<"Local", 2016, "local">;
def ImageTarget
    : I32EnumAttrCase<"Image", 2017, "image">;
def HostBufferTarget
    : I32EnumAttrCase<"HostBuffer", 2018, "host_buffer">;
def HostImageTarget
    : I32EnumAttrCase<"HostImage", 2019, "host_image">;
def ImageArrayTarget
    : I32EnumAttrCase<"ImageArray", 2020, "image_array">;

def TargetAttr
    : SYCL_I32EnumAttr<"Target", "sycl.target",
                       [GlobalBufferTarget, ConstantBufferTarget, LocalTarget,
		       ImageTarget, HostBufferTarget, HostImageTarget,
		       ImageArrayTarget]>;

////////////////////////////////////////////////////////////////////////////////
// AccessAddrSpace
////////////////////////////////////////////////////////////////////////////////

def PrivateAccessAddrSpace
    : I32EnumAttrCase<"PrivateAccess", 0, "private">;
def GlobalAccessAddrSpace
    : I32EnumAttrCase<"GlobalAccess", 1, "global">;
def ConstantAccessAddrSpace
    : I32EnumAttrCase<"ConstantAccess", 2, "constant">;
def LocalAccessAddrSpace
    : I32EnumAttrCase<"LocalAccess", 3, "local">;
def GenericAccessAddrSpace
    : I32EnumAttrCase<"GenericAccess", 4, "generic">;
def ExtIntelGlobalDeviceAccessAddrSpace
    : I32EnumAttrCase<"ExtIntelGlobalDeviceAccess", 5,
                      "ext_intel_global_device_access">;
def ExtIntelHostAccessAddrSpace
    : I32EnumAttrCase<"ExtIntelHostAccess", 6, "ext_intel_host_access">;

def AccessAddrSpaceAttr
    : SYCL_I32EnumAttr<"AccessAddrSpace", "sycl.access.address_space",
                       [PrivateAccessAddrSpace, GlobalAccessAddrSpace,
                       ConstantAccessAddrSpace, LocalAccessAddrSpace,
                       GenericAccessAddrSpace,
                       ExtIntelGlobalDeviceAccessAddrSpace,
                       ExtIntelHostAccessAddrSpace]>;

////////////////////////////////////////////////////////////////////////////////
// AccessDecorated
////////////////////////////////////////////////////////////////////////////////

def NoAccessDecorated : I32EnumAttrCase<"No", 0, "no">;
def YesAccessDecorated : I32EnumAttrCase<"Yes", 1, "yes">;
def LegacyAccessDecorated : I32EnumAttrCase<"Legacy", 2, "legacy">;

def AccessDecoratedAttr
    : SYCL_I32EnumAttr<"AccessDecorated", "sycl.access.decorated",
                       [NoAccessDecorated, YesAccessDecorated,
                       LegacyAccessDecorated]>;

#endif // SYCL_OP_ATTRIBUTES
