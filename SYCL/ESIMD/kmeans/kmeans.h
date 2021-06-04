//==---------------- kmeans.h  - DPC++ ESIMD on-device test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KMEANS_H_INCLUDED
#define KMEANS_H_INCLUDED

static constexpr unsigned int NUM_POINTS = 98304 * 8;
static constexpr unsigned int POINTS_PER_THREAD = 256;
static constexpr unsigned int POINTS_PER_WORKITEM = 64;
static constexpr unsigned int NUM_CENTROIDS_ACTUAL = 20;
// round to next 16 so as to handle reading centroids easily
// read 16 centroids at one time
static constexpr unsigned int NUM_CENTROIDS_ALLOCATED =
    ((NUM_CENTROIDS_ACTUAL + 15) / 16) * 16;
static constexpr unsigned int SIMD_SIZE = 16;
static constexpr unsigned int NUM_ITERATIONS = 7;

typedef struct {
  float x;
  float y;
  int cluster;
} Point;

typedef struct {
  float x;
  float y;
  int num_points;
} Centroid;

typedef struct {
  float x_sum;
  float y_sum;
  int num_points;
} Accum;

// AOSOA (array of structure of arrays)
typedef union {
  float xyn[SIMD_SIZE + SIMD_SIZE + SIMD_SIZE];
  struct {
    float x[SIMD_SIZE];
    float y[SIMD_SIZE];
    int cluster[SIMD_SIZE];
  };
} Point4;

typedef union {
  float xyn[SIMD_SIZE + SIMD_SIZE + SIMD_SIZE];
  struct {
    float x[SIMD_SIZE];
    float y[SIMD_SIZE];
    int num_points[SIMD_SIZE];
  };
} Centroid4;

typedef struct {
  float x_sum[NUM_POINTS / POINTS_PER_THREAD];
  float y_sum[NUM_POINTS / POINTS_PER_THREAD];
  int num_points[NUM_POINTS / POINTS_PER_THREAD];
} Accum4;

#endif // KMEANS_H_INCLUDED
