//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include <clc/math/tables.h>

DECLARE_TABLE(uchar, PIBITS_TBL, ) = {
    224, 241, 27, 193, 12, 88, 33, 116, 53, 126, 196, 126, 237, 175,
    169, 75, 74, 41, 222, 231, 28, 244, 236, 197, 151, 175, 31,
    235, 158, 212, 181, 168, 127, 121, 154, 253, 24, 61, 221, 38,
    44, 159, 60, 251, 217, 180, 125, 180, 41, 104, 45, 70, 188,
    188, 63, 96, 22, 120, 255, 95, 226, 127, 236, 160, 228, 247,
    46, 126, 17, 114, 210, 231, 76, 13, 230, 88, 71, 230, 4, 249,
    125, 209, 154, 192, 113, 166, 19, 18, 237, 186, 212, 215, 8,
    162, 251, 156, 166, 196, 114, 172, 119, 248, 115, 72, 70, 39,
    168, 187, 36, 25, 128, 75, 55, 9, 233, 184, 145, 220, 134, 21,
    239, 122, 175, 142, 69, 249, 7, 65, 14, 241, 100, 86, 138, 109,
    3, 119, 211, 212, 71, 95, 157, 240, 167, 84, 16, 57, 185, 13,
    230, 139, 2, 0, 0, 0, 0, 0, 0, 0
};

uint4 TABLE_MANGLE(pibits_tbl)(size_t idx) {
    return *(__constant uint4 *)(PIBITS_TBL + idx);
}
