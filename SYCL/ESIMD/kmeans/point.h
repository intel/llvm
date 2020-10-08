//==---------------- point.h  - DPC++ ESIMD on-device test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POINT_H_INCLUDED
#define POINT_H_INCLUDED

typedef struct {
  float x;
  float y;
  int cluster;
} Point;

#define DWORD_PER_POINT 3

typedef struct {
  float x;
  float y;
  int num_points;
} Centroid;

#define DWORD_PER_CENTROID 3

typedef struct {
  float x_sum;
  float y_sum;
  int num_points;
} Accum;

#define DWORD_PER_ACCUM 3

#endif // POINT_H_INCLUDED
