//==---------------- kmeans.cpp  - DPC++ ESIMD on-device test --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda
// RUN: %clangxx-esimd -fsycl %s -I%S/.. -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out %S/points.big.json
// RUN: %ESIMD_RUN_PLACEHOLDER %t.out %S/points.big.json
//

#include "kmeans.h"
#include "esimd_test_utils.hpp"
#include "point.h"

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>

using namespace cl::sycl;
using namespace sycl::INTEL::gpu;
using namespace std;

// Each HW thread process POINTS_PER_THREAD points. If the total number of
// points is not divisible by POINTS_PER_THREAD. The following code then needs
// to handle out-of-bound reads/write. For now, we ignore the scenario by
// assuming totoal number of points can be divided by POINTS_PER_THREAD.
// kmeans is invoked multiple of iterations for centroids to converge. Only the
// final clustering results of points of the later iteration are recorded.
//

template <class T>
void writeBuf(void *buf, const char *str, T val, uint32_t &idx) {
  uint strl = 0;
  while (*(str + strl) != 0) {
    *((unsigned char *)buf + idx) = *(str + strl);
    strl++;
    idx++;
  }

  // Terminate string
  *((unsigned char *)buf + idx) = 0;
  idx++;

  // Write sizeof val
  *((unsigned char *)buf + idx) = sizeof(T);
  idx++;

  // Write actual value
  auto addr = (T *)((unsigned char *)buf + idx);
  *addr = val;
  idx += sizeof(T);
}

ESIMD_INLINE void cmk_kmeans(
    uint *pts_si, // binding table index of points
    uint *cen_si, // binding table index of centroids
    uint *acc_si, // binding table index of accum for computing new centroids
    unsigned num_points,  // number of points
    bool final_iteration, // if this is the final iteration of kmeans
    uint lid, char *dbgBuf) {
  // read in all centroids
  // this version we can only handle number of centroids no more than 64
  // We don't need cluster field so we read only the first two field
  simd<float, 2 * ROUND_TO_16_NUM_CENTROIDS> centroids;
  auto centroidsMatRef =
      centroids.format<float, 2, ROUND_TO_16_NUM_CENTROIDS>();
  simd<uint32_t, 16> offsets(0, 1);
  offsets = offsets * DWORD_PER_POINT;

#if 0
  // 1000 bytes per thread
  uint _pos = lid * 1000;
#endif

#pragma unroll
  for (unsigned i = 0; i < ROUND_TO_16_NUM_CENTROIDS;
       i += 16) // round up to next 16
  {
    // untyped read will pack R, G into SOA
    // matrix.row(0): x x x x . . . . // 16 position x
    // matrix.row(1): y y y y . . . . // 16 position y
    simd<uint32_t, 16> addrs = (i * DWORD_PER_CENTROID);
    addrs += offsets;
    addrs *= sizeof(float);

    centroidsMatRef.select<2, 1, 16, 1>(0, i) =
        gather4<float, 16, ChannelMaskType::ESIMD_GR_ENABLE>((float *)cen_si,
                                                             addrs);
  }

  simd<float, 3 * ROUND_TO_16_NUM_CENTROIDS> accum(0);
  auto accumMatRef = accum.format<float, 3, ROUND_TO_16_NUM_CENTROIDS>();
  auto num =
      accumMatRef.row(2).format<unsigned, 1, ROUND_TO_16_NUM_CENTROIDS>();

  uint linear_tid = lid;
  // each thread handles 256 points
  unsigned start = linear_tid * POINTS_PER_THREAD *
                   DWORD_PER_POINT; // each point has 3 DWORD

  // use untyped read to read in points.
  // Point is x, y, c
  // the returned result will be shuffled. x, y, c will be packed nicely
  for (unsigned i = 0; i < POINTS_PER_THREAD; i += 16) {
    simd<float, 2 * 16> pos;
    auto posMatRef = pos.format<float, 2, 16>();
    simd<unsigned, 16> cluster(0);
    simd<uint32_t, 16> addrs = start;
    addrs += (i * DWORD_PER_POINT);
    addrs += offsets;
    addrs *= sizeof(float);

    pos = gather4<float, 16, ChannelMaskType::ESIMD_GR_ENABLE>((float *)pts_si,
                                                               addrs);

    simd<float, 16> dx = posMatRef.row(0) - centroidsMatRef.row(0)[0];
    simd<float, 16> dy = posMatRef.row(1) - centroidsMatRef.row(1)[0];
    simd<float, 16> min_dist = dx * dx + dy * dy;

#pragma unroll
    for (unsigned j = 1; j < NUM_CENTROIDS; j++) {
      // compute distance
      dx = posMatRef.row(0) - centroidsMatRef.row(0)[j];
      dy = posMatRef.row(1) - centroidsMatRef.row(1)[j];
      simd<float, 16> dist = dx * dx + dy * dy;
      cluster.merge(j, dist < min_dist);
      min_dist.merge(dist, dist < min_dist);
    }

    // if this is the final invocation of kmeans, write back clustering
    // result
    if (final_iteration) {
      // point: x, y, cluster
      // i * DWORD_PER_POINT + 2 to write to cluster field
      simd<uint32_t, 16> addrs = (i * DWORD_PER_POINT) + 2;
      addrs += offsets;
      addrs += start;
      addrs *= sizeof(unsigned);

      scatter<unsigned, 16>(pts_si, cluster, addrs);
    }
    // go over each point and according to their classified cluster update
    // accum

#pragma unroll
    for (unsigned k = 0; k < 16; k++) {
      unsigned c = cluster[k];
      accumMatRef.select<1, 1, 1, 1>(0, c) += posMatRef.row(0)[k];
    }
#pragma unroll
    for (unsigned k = 0; k < 16; k++) {
      unsigned c = cluster[k];
      accumMatRef.select<1, 1, 1, 1>(1, c) += posMatRef.row(1)[k];
    }
#pragma unroll
    for (unsigned k = 0; k < 16; k++) {
      unsigned c = cluster[k];
      num.select<1, 1, 1, 1>(0, c)++;
    }
  }

  unsigned startoff = linear_tid * DWORD_PER_ACCUM * NUM_CENTROIDS;
#pragma unroll
  for (unsigned i = 0; i < ROUND_TO_16_NUM_CENTROIDS;
       i += 16) // round up to next 16
  {
    auto a16 = accumMatRef.select<3, 1, 16, 1>(0, i).read();
    simd<uint16_t, 16> masks;
    simd<uint32_t, 16> addrs = startoff + (i * DWORD_PER_ACCUM);
    addrs += offsets;
    addrs *= sizeof(float);

    masks =
        ((offsets) + (i * DWORD_PER_ACCUM)) < (NUM_CENTROIDS * DWORD_PER_ACCUM);

    scatter4<float, 16, ChannelMaskType::ESIMD_BGR_ENABLE>((float *)acc_si, a16,
                                                           addrs, masks);
  }
}

// each HW thread sum up 8xNUM_CENTROIDS accum
ESIMD_INLINE void cmk_accum_reduction(uint32_t *acc_si, uint lid) {

  // each thread computes one single centriod
  uint linear_tid = lid;
  simd<float, 3 * ROUND_TO_16_NUM_CENTROIDS> accum;
  simd<float, 3 * ROUND_TO_16_NUM_CENTROIDS> sum(0);
  auto sumMatRef = sum.format<float, 3, ROUND_TO_16_NUM_CENTROIDS>();
  auto num = sum.format<float, 3, ROUND_TO_16_NUM_CENTROIDS>()
                 .row(2)
                 .format<unsigned, 1, ROUND_TO_16_NUM_CENTROIDS>();
  simd<uint32_t, 16> offsets(0, 1);
  offsets = offsets * DWORD_PER_POINT;
  unsigned start =
      linear_tid * ACCUM_REDUCTION_RATIO * NUM_CENTROIDS * DWORD_PER_ACCUM;
#pragma unroll
  for (unsigned i = 0; i < ACCUM_REDUCTION_RATIO; i++) {
    unsigned next = start + i * NUM_CENTROIDS * DWORD_PER_ACCUM;
#pragma unroll
    for (unsigned j = 0; j < ROUND_TO_16_NUM_CENTROIDS;
         j += 16) // round up to next 16
    {
      simd<uint32_t, 16> addrs = offsets;
      offsets += next + (j * DWORD_PER_ACCUM);
      offsets *= sizeof(float);

      accum.select<48, 1>(j * 3) =
          gather4<float, 16, ChannelMaskType::ESIMD_BGR_ENABLE>((float *)acc_si,
                                                                addrs);
    }
    auto accumMatRef = accum.format<float, 3, ROUND_TO_16_NUM_CENTROIDS>();
    sumMatRef.row(0) += accumMatRef.row(0);
    sumMatRef.row(1) += accumMatRef.row(1);
    num += accumMatRef.row(2).format<unsigned, 1, ROUND_TO_16_NUM_CENTROIDS>();
  }

#pragma unroll
  for (unsigned i = 0; i < ROUND_TO_16_NUM_CENTROIDS;
       i += 16) // round up to next 16
  {
    simd<uint16_t, 16> masks;
    auto a16 = sumMatRef.select<16, 1, 3, 1>(i, 0).read();
    simd<uint32_t, 16> addrs = offsets;
    addrs += start + (i * DWORD_PER_ACCUM);
    addrs *= sizeof(float);

    masks = (offsets + i * DWORD_PER_ACCUM < NUM_CENTROIDS * DWORD_PER_ACCUM);

    scatter4<float, 16, ChannelMaskType::ESIMD_BGR_ENABLE>((float *)acc_si, a16,
                                                           addrs, masks);
  }
}

ESIMD_INLINE void cmk_compute_centroid_position(
    uint32_t *cen_si, // binding table index of centroids
    uint32_t
        *acc_si, // binding table index of accum for computing new centroids
    uint lid,
    char *dbgBuf) { // linear group id
#if 0
  uint _pos = lid * 120;
  writeBuf(dbgBuf, "lid", lid, _pos);
#endif

  simd<uint32_t, 16> offsets(0, 1);
  const unsigned stride = NUM_CENTROIDS * ACCUM_REDUCTION_RATIO;
  offsets = offsets * DWORD_PER_ACCUM * stride;

  // each thread computes one single centriod
  uint linear_tid = lid;

  simd<float, 16> X(0);
  simd<float, 16> Y(0);
  simd<unsigned, 16> N(0);

  unsigned num_accum_record =
      (NUM_POINTS / (POINTS_PER_THREAD * ACCUM_REDUCTION_RATIO));

  // process 4 reads per iterations to hide latency
#pragma unroll
  for (unsigned i = 0; i < (num_accum_record >> 6) << 6; i += 64) {
    // untyped read will pack R, G, B into SOA
    // matrix.row(0): x x x x . . . . // 16 position x
    // matrix.row(1): y y y y . . . . // 16 position y
    // matrix.row(1): n n n n . . . . // 16 position num of points
    simd<uint32_t, 16> addrs = offsets;
    addrs += (linear_tid * DWORD_PER_ACCUM);

    simd<float, 3 * 16> accum0;
    auto accum0MatRef = accum0.format<float, 3, 16>();
    accum0 = gather4<float, 16, ChannelMaskType::ESIMD_BGR_ENABLE>(
        (float *)acc_si,
        (addrs + (i * DWORD_PER_ACCUM * stride)) * sizeof(float));
    simd<float, 3 * 16> accum1;
    auto accum1MatRef = accum1.format<float, 3, 16>();
    accum1 = gather4<float, 16, ChannelMaskType::ESIMD_BGR_ENABLE>(
        (float *)acc_si,
        (addrs + ((i + 16) * DWORD_PER_ACCUM * stride)) * sizeof(float));
    simd<float, 3 * 16> accum2;
    auto accum2MatRef = accum2.format<float, 3, 16>();
    accum2 = gather4<float, 16, ChannelMaskType::ESIMD_BGR_ENABLE>(
        (float *)acc_si,
        (addrs + ((i + 32) * DWORD_PER_ACCUM * stride)) * sizeof(float));
    simd<float, 3 * 16> accum3;
    auto accum3MatRef = accum3.format<float, 3, 16>();
    accum3 = gather4<float, 16, ChannelMaskType::ESIMD_BGR_ENABLE>(
        (float *)acc_si,
        (addrs + ((i + 48) * DWORD_PER_ACCUM * stride)) * sizeof(float));
    X += accum0MatRef.row(0) + accum1MatRef.row(0) + accum2MatRef.row(0) +
         accum3MatRef.row(0);
    Y += accum0MatRef.row(1) + accum1MatRef.row(1) + accum2MatRef.row(1) +
         accum3MatRef.row(1);
    N += accum0MatRef.row(2).format<unsigned, 1, 16>() +
         accum1MatRef.row(2).format<unsigned, 1, 16>() +
         accum2MatRef.row(2).format<unsigned, 1, 16>() +
         accum3MatRef.row(2).format<unsigned, 1, 16>();
  }
  // process remaining loop iterations
#pragma unroll
  for (unsigned i = (num_accum_record >> 6) << 6; i < num_accum_record;
       i += 16) {
    simd<float, 3 * 16> accum0;
    auto accum0MatRef = accum0.format<float, 3, 16>();
    simd<uint32_t, 16> addrs = offsets;
    addrs += (i * DWORD_PER_ACCUM * stride);
    addrs += (linear_tid * DWORD_PER_ACCUM);
    addrs *= sizeof(float);
    accum0 = gather4<float, 16, ChannelMaskType::ESIMD_BGR_ENABLE>(
        (float *)acc_si, addrs);
    X += accum0MatRef.row(0).read();
    Y += accum0MatRef.row(1).read();
    N += accum0MatRef.row(2).format<unsigned, 1, 16>().read();
  }

  simd<float, 16> centroid(0.0f);
  unsigned num = reduce<unsigned>(N, std::plus<>());

  centroid.select<1, 1>(0) = reduce<float>(X, std::plus<>()) / num;
  centroid.select<1, 1>(1) = reduce<float>(Y, std::plus<>()) / num;
  auto centroidsInt = centroid.format<unsigned int>();
  centroidsInt.select<1, 1>(2) = num;

  // update centroid(linear_tid)

  simd<ushort, 16> mask(0);
  mask.select<3, 1>(0) = 1;
  simd<uint32_t, 16> addrs;
  addrs = linear_tid;
  addrs *= DWORD_PER_CENTROID;
  simd<uint32_t, 8> init8(0, 1);
  addrs.select<8, 1>(0) += init8;
  addrs *= sizeof(float);

  scatter<float, 16>((float *)cen_si, centroid, addrs, mask);
}

// Begin copy-paste from kmeans.cpp

typedef bool boolean;

#define NUM_ITERATIONS 7

inline float dist(Point p, Centroid c) {
  float dx = p.x - c.x;
  float dy = p.y - c.y;
  return dx * dx + dy * dy;
}

void clustering(Point *pts,         // points
                unsigned num_pts,   // number of points
                Centroid *ctrds,    // centroids
                unsigned num_ctrds) // number of centroids
{

  for (auto i = 0; i < num_pts; i++) {
    float min_dis = -1;
    auto cluster_idx = 0;
    // for each point, compute the min distance to centroids
    for (auto j = 0; j < num_ctrds; j++) {
      float dis = dist(pts[i], ctrds[j]);

      if (dis < min_dis || min_dis == -1) {
        min_dis = dis;
        cluster_idx = j;
      }
    }
    pts[i].cluster = cluster_idx;
  }
  // compute new positions of centroids
  Accum *accum = (Accum *)malloc(num_ctrds * sizeof(Accum));
  memset(accum, 0, num_ctrds * sizeof(Accum));
  for (auto i = 0; i < num_pts; i++) {
    auto c = pts[i].cluster;
    accum[c].x_sum += pts[i].x;
    accum[c].y_sum += pts[i].y;
    accum[c].num_points++;
  }
  for (auto j = 0; j < num_ctrds; j++) {
    ctrds[j].x = accum[j].x_sum / accum[j].num_points;
    ctrds[j].y = accum[j].y_sum / accum[j].num_points;
    ctrds[j].num_points = accum[j].num_points;
  }
  delete accum;
}

#define max(a, b) (((a) > (b)) ? (a) : (b))

boolean verify_result(Point *gpu_pts,      // gpu points result
                      Point *cpu_pts,      // cpu points result
                      Centroid *gpu_ctrds, // gpu centroids result
                      Centroid *cpu_ctrds, // cpu centroids result
                      unsigned num_pts,    // number of points
                      unsigned num_ctrds)  // number of centroids
{

  for (auto i = 0; i < num_ctrds; i++) {
    float errX = fabs(gpu_ctrds[i].x - cpu_ctrds[i].x) /
                 max(fabs(gpu_ctrds[i].x), fabs(cpu_ctrds[i].x));
    float errY = fabs(gpu_ctrds[i].y - cpu_ctrds[i].y) /
                 max(fabs(gpu_ctrds[i].y), fabs(cpu_ctrds[i].y));
    float errSize =
        abs(gpu_ctrds[i].num_points - cpu_ctrds[i].num_points) /
        max(abs(gpu_ctrds[i].num_points), abs(cpu_ctrds[i].num_points));
    std::cout << i << ": Wanted (" << cpu_ctrds[i].x << "," << cpu_ctrds[i].y
              << ", " << cpu_ctrds[i].num_points << ")\n";
    std::cout << "Got (" << gpu_ctrds[i].x << "," << gpu_ctrds[i].y << ", "
              << gpu_ctrds[i].num_points << ")\n";
    if (errX >= 0.001f || errY >= 0.001f || errSize >= 0.001f) {
      std::cout << "Error, index " << i << ": Wanted (" << cpu_ctrds[i].x << ","
                << cpu_ctrds[i].y << ", " << cpu_ctrds[i].num_points << ")\n";
      std::cout << "Got (" << gpu_ctrds[i].x << "," << gpu_ctrds[i].y << ", "
                << gpu_ctrds[i].num_points << ")\n";
      return false;
    }
  }

  return true;
}

void dump_accum(Accum *accum, unsigned total_accum_record) {
  std::cout << "           \tx_sum\t\ty_sum\t num_points" << std::endl;
  for (unsigned i = 0; i < total_accum_record; i++) {
    if (i % NUM_CENTROIDS == 0)
      std::cout << std::endl;
    std::cout << "Accum " << i << " \t" << accum[i].x_sum << "\t\t"
              << accum[i].y_sum << " \t" << accum[i].num_points << std::endl;
  }
}
// take initial points and run k mean clustering number of iterations
void cpu_kmeans(Point *pts,          // points
                unsigned num_pts,    // number of points
                Centroid *ctrds,     // centroids
                unsigned num_ctrds,  // number of centroids
                unsigned iterations) // run clustering number of iterations
{

  for (auto i = 0; i < iterations; i++) {
    clustering(pts, num_pts, ctrds, num_ctrds);
  }
}

// GPU has already aggregrated x_sum, y_sum and num_points for each cluster
// this function computes new centroid position
void cpu_compute_centroid_position(Accum *accum, Centroid *centroids,
                                   unsigned num_centroids) {
  for (auto i = 0; i < num_centroids; i++) {
    centroids[i].x = accum[i].x_sum / accum[i].num_points;
    centroids[i].y = accum[i].y_sum / accum[i].num_points;
    centroids[i].num_points = accum[i].num_points;
  }
}

std::vector<std::pair<double, double>> read(std::string &data) {
  std::vector<std::pair<double, double>> ret;
  auto pos = data.find("[");
  pos += 1;

  auto findNextPair = [&data](auto &pos, std::pair<double, double> &p) {
    //  [2.1234,1.234]
    // pos pointing at or before '['
    pos = data.find('[', pos);
    if (pos == std::string::npos)
      return false;

    // Move beyond '['
    pos += 1;

    auto comma = data.find(',', pos);
    std::string firstNumber = data.substr(pos, comma - pos);

    // Move beyond ','
    comma += 1;

    pos = data.find(']', comma);
    std::string secondNumber = data.substr(comma, pos - comma);

    p.first = std::atof(firstNumber.data());
    p.second = std::atof(secondNumber.data());

    return true;
  };

  while (1) {
    std::pair<double, double> d;
    bool valid = findNextPair(pos, d);
    if (!valid)
      break;

    ret.push_back(d);
  }

  return ret;
}

// Wrote afresh to read data from json file instead of
// using Cm's implementation that used a json library.
// There were some compile time issues using the lib.
bool openPointsJSON(const char *filename,
                    std::vector<std::pair<double, double>> &data) {
  std::ifstream fs;
  fs.open(filename, std::ifstream::in);

  if (!fs.is_open()) {
    std::cerr << "Error opening file\n";
    return false;
  }

  std::filebuf *pbuf = fs.rdbuf();
  std::size_t size = pbuf->pubseekoff(0, fs.end, fs.in);
  pbuf->pubseekpos(0, fs.in);

  // allocate memory to contain file data
  char *buffer = new char[size];
  // get file data
  pbuf->sgetn(buffer, size);

  std::string line(buffer);
  delete[] buffer;

  data = read(line);

  fs.close();

  return true;
}

// return msecond
static double report_time(const string &msg, event e) {
  cl_ulong time_start =
      e.get_profiling_info<info::event_profiling::command_start>();
  cl_ulong time_end =
      e.get_profiling_info<info::event_profiling::command_end>();
  double elapsed = (time_end - time_start) / 1e6;
  // cerr << msg << elapsed << " msecs" << std::endl;
  return elapsed;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: kmeans.exe input_file" << std::endl;
    exit(1);
  }

  cl::sycl::property_list props{property::queue::enable_profiling{},
                                property::queue::in_order()};
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler(),
          props);

  auto dev = q.get_device();
  auto ctxt = q.get_context();

  size_t index;

  std::vector<std::pair<double, double>> pointsDataJSON;
  auto success = openPointsJSON(argv[1], pointsDataJSON);

  // 100.000 points it's the repository default.

  // validates json
  if (!success) {
    printf("Error parsing Json file");
    fflush(stdout);
    return -1;
  }

  // allocate memory for points
  Point *points = (Point *)malloc_shared(NUM_POINTS * sizeof(Point), dev, ctxt);
  memset(points, 0, NUM_POINTS * sizeof(Point));

  // load points from json
  unsigned int idx = 0;
  for (auto item : pointsDataJSON) {
    points[idx].x = item.first;
    points[idx].y = item.second;
    ++idx;

    if (idx == NUM_POINTS)
      break;
  }

  std::cout << "read in points" << std::endl;

  // allocate memory for points and centroids
  Centroid *centroids =
      (Centroid *)malloc_shared(NUM_CENTROIDS * sizeof(Centroid), dev, ctxt);
  // init centroids with the first num_centroids points
  for (auto i = 0; i < NUM_CENTROIDS; i++) {
    centroids[i].x = points[i].x;
    centroids[i].y = points[i].y;
    centroids[i].num_points = 0;
  }
  // Accum is for aggregrating (x,y) of the same cluster to compute new
  // centroid positions
  Accum *accum = (Accum *)malloc_shared((NUM_POINTS / POINTS_PER_THREAD) *
                                            NUM_CENTROIDS * sizeof(Accum),
                                        dev, ctxt);
  memset(accum, 0,
         (NUM_POINTS / POINTS_PER_THREAD) * NUM_CENTROIDS * sizeof(Accum));

  // compute CPU kmean results for verifying results later
  std::cout << "compute reference output" << std::endl;
  Point *cpu_points = (Point *)malloc(NUM_POINTS * sizeof(Point));
  memcpy(cpu_points, points, NUM_POINTS * sizeof(Point));
  Centroid *cpu_centroids =
      (Centroid *)malloc(NUM_CENTROIDS * sizeof(Centroid));
  memcpy(cpu_centroids, centroids, NUM_CENTROIDS * sizeof(Centroid));
  cpu_kmeans(cpu_points, NUM_POINTS, cpu_centroids, NUM_CENTROIDS,
             NUM_ITERATIONS);

  std::cout << "compute reference output successful" << std::endl;

  double kernel1_time_in_ns = 0;
  double kernel2_time_in_ns = 0;
  double kernel3_time_in_ns = 0;
#ifndef ENABLE_PRINTF
  constexpr unsigned int SZ = 0;
  char *dbgBuf = nullptr;
#else
  constexpr unsigned int SZ = 10000;
  char *dbgBuf = (char *)malloc_shared(SZ * sizeof(uint), dev, ctxt);
#endif
  memset(dbgBuf, 0, SZ * sizeof(uint));

  //----
  // Actual execution goes here

  unsigned int total_threads = (NUM_POINTS - 1) / POINTS_PER_THREAD + 1;
  auto GlobalRange = cl::sycl::range<1>(total_threads);
  cl::sycl::range<1> LocalRange{1};
  auto GlobalRange1 = cl::sycl::range<1>(NUM_CENTROIDS);
  cl::sycl::range<1> LocalRange1 = cl::sycl::range<1>{1};

  auto submitJobs = [&](bool finalIter) {
    // kmeans
    auto e = q.submit([&](cl::sycl::handler &cgh) {
      cgh.parallel_for<class kMeans>(
          GlobalRange * LocalRange, [=](id<1> i) SYCL_ESIMD_KERNEL {
            using namespace sycl::INTEL::gpu;
            cmk_kmeans((uint *)points, (uint *)centroids, (uint *)accum,
                       NUM_POINTS, finalIter, i, dbgBuf);
          });
    });

    e.wait();
    kernel1_time_in_ns += report_time("kernel1", e);

    printf("Done with kmeans\n");

#if ACCUM_REDUCTION_RATIO > 1
    // accum_reduction
    unsigned total_accum_record = NUM_POINTS / POINTS_PER_THREAD;
    GlobalRange =
        cl::sycl::range<2>(total_accum_record / ACCUM_REDUCTION_RATIO, 1);
    auto e1 = q.submit([&](cl::sycl::handler &cgh) {
      cgh.parallel_for<class kAccumRed>(GlobalRange * LocalRange,
                                        [=](id<1> i) SYCL_ESIMD_KERNEL {
                                          using namespace sycl::INTEL::gpu;
                                          cmk_accum_reduction((uint *)accum, i);
                                        });
    });
    e1.wait();
    kernel2_time_in_ns += report_time("kernel2", e1);
#endif

    // compute centroid position
    auto e2 = q.submit([&](cl::sycl::handler &cgh) {
      cgh.parallel_for<class kCompCentroidPos>(
          GlobalRange1 * LocalRange1, [=](id<1> i) SYCL_ESIMD_KERNEL {
            using namespace sycl::INTEL::gpu;
            cmk_compute_centroid_position((uint *)centroids, (uint *)accum, i,
                                          dbgBuf);
          });
    });
    e2.wait();
    kernel3_time_in_ns += report_time("kernel3", e2);
  };

  try {
    for (auto i = 0; i < NUM_ITERATIONS - 1; i++) {
      submitJobs(false);
    }
    submitJobs(true);
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    delete cpu_points;
    delete cpu_centroids;
    return e.get_cl_code();
  }

  //---

  auto correct = verify_result(points, cpu_points, centroids, cpu_centroids,
                               NUM_POINTS, NUM_CENTROIDS);
#ifdef ENABLE_PRINTF
  std::cout << "Dbgbuf:" << std::endl;
  char tmp[SZ + 1];
  uint idx1 = 0;
  uint totalSize = 0;
  bool strName = true;
  for (unsigned int i = 0; i < SZ; i++) {
    if (strName) {
      tmp[idx1++] = dbgBuf[i];
      if (dbgBuf[i] == 0) {
        // end string
        strName = false;
        continue;
      }
    } else {
      // read actual value
      unsigned int size = dbgBuf[i++];
      totalSize += size;
      if (size > 0) {
        std::cout << (const char *)tmp << "(" << size << " bytes) = ";
        if (size > 1000)
          std::cout << "***** Bad size *****" << std::endl;
        else {

          for (unsigned int j = 0; j != size; j++, i++) {
            printf("0x%x ", ((unsigned char *)dbgBuf)[i]);
          }
          std::cout << std::endl;
        }
        i--;
      }
      for (unsigned int i1 = 0; i1 != 512; i1++)
        tmp[i1] = 0;
      strName = true;
      idx1 = 0;
    }
  }
#endif

  std::cout << std::endl;

  float kernel1_time = kernel1_time_in_ns;
  float kernel2_time = kernel2_time_in_ns;
  float kernel3_time = kernel3_time_in_ns;
  float kernel_time = kernel1_time + kernel2_time + kernel3_time;

  printf("\n--- ESIMD Kernel execution stats begin ---\n");

  printf("NUMBER_OF_POINTS: %d\n", NUM_POINTS);
  printf("NUMBER_OF_CENTROIDS: %d\n", NUM_CENTROIDS);
  printf("NUMBER_OF_ITERATIONS: %d\n", NUM_ITERATIONS);
  printf("POINTS_PER_THREAD: %d\n", POINTS_PER_THREAD);
  printf("ACCUM_REDUCTION_RATIO: %d\n\n", ACCUM_REDUCTION_RATIO);

  printf("Average kernel1 time: %f ms\n", kernel1_time / NUM_ITERATIONS);
  printf("Total kernel1 time: %f ms\n\n", kernel1_time);

  printf("Average kernel2 time: %f ms\n", kernel2_time / NUM_ITERATIONS);
  printf("Total kernel2 time: %f ms\n\n", kernel2_time);

  printf("Average kernel3 time: %f ms\n", kernel3_time / NUM_ITERATIONS);
  printf("Total kernel3 time: %f ms\n\n", kernel3_time);

  printf("Average kernel time: %f ms\n", kernel_time / NUM_ITERATIONS);
  printf("Total kernel time: %f ms\n\n", kernel_time);

  printf("--- ESIMD Kernel execution stats end ---\n\n");

  std::cout << std::endl;

  delete cpu_points;
  delete cpu_centroids;

  std::cout << ((correct) ? "PASSED" : "FAILED") << std::endl;
  return !correct;
}
