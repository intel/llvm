//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_DEF _CLC_OVERLOAD uint __spirv_GlobalOffset_x() {
	__attribute__((address_space(7))) uint * ptr =
		(__attribute__((address_space(7))) uint *)
		__builtin_r600_implicitarg_ptr();
    return ptr[1];
}

_CLC_DEF _CLC_OVERLOAD uint __spirv_GlobalOffset_y() {
	__attribute__((address_space(7))) uint * ptr =
		(__attribute__((address_space(7))) uint *)
		__builtin_r600_implicitarg_ptr();
    return ptr[2];
}

_CLC_DEF _CLC_OVERLOAD uint __spirv_GlobalOffset_z() {
	__attribute__((address_space(7))) uint * ptr =
		(__attribute__((address_space(7))) uint *)
		__builtin_r600_implicitarg_ptr();
    return ptr[3];
}
