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
