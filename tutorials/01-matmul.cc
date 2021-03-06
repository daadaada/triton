#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include <iomanip>
#include <cstring>
#include <sstream>
#include <cstdio>
#include <tuple>
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/tools/bench.hpp"
#include "triton/external/half.hpp"
#include "triton/runtime/function.h"
#include <iomanip>
#include <cmath>
#include "triton/runtime/function.h"

namespace drv = triton::driver;
namespace rt = triton::runtime;

namespace src {

    const char *dot =
R"(
#define STM 8
#define STN 8

__global__ void dot(TYPE * A __noalias __readonly __aligned(16),
                    TYPE * B __noalias __readonly __aligned(16),
                    TYPE * C __noalias __aligned(16),
                    float alpha,
                    int M __retune,
                    int N __retune,
                    int K __retune __multipleof(16),
                    int lda __multipleof(8),
                    int ldb __multipleof(8),
                    int ldc __multipleof(8),
                    int* locks) {
      // prologue
      int pid = get_program_id(0);
      int pidz = get_program_id(2);
      int gridm = (M + TM - 1) / TM;
      int gridn = (N + TN - 1) / TN;
      int width = STM*gridn;
      int stm = pid / width;
      int RSTM  = min(gridm - stm*STM, STM);
      int stn =  (pid % width) / (RSTM*STN);
      int RSTN = min(gridn - stn*STN, STN);
      int laneid = pid % (RSTM * RSTN);
      int lanem = laneid / RSTN;
      int lanen = laneid % RSTN;
      int pidm = stm*STM + lanem;
      int pidn = stn*STN + lanen;
      int rm[TM] = pidm * TM + 0 ... TM;
      int rn[TN] = pidn * TN + 0 ... TN;

      // reduction splitting
      K           = K / TZ;
      int rk[TK]  = pidz * K + 0 ... TK;
      // pointers to operands
      int offa[TM, TK] = rk[newaxis, :] * STRIDE_AK + rm[:, newaxis] * STRIDE_AM;
      int offb[TK, TN] = rk[:, newaxis] * STRIDE_BK + rn[newaxis, :] * STRIDE_BN;
      TYPE* pa[TM, TK] = A + offa;
      TYPE* pb[TK, TN] = B + offb;
      // prefetches operands
      bool checka[TM, TK] = rk[newaxis, :] < K;
      bool checkb[TK, TN] = rk[:, newaxis] < K;
      TYPE a[TM, TK] = checka ? *pa : 0;
      TYPE b[TK, TN] = checkb ? *pb : 0;
      pa += TK * STRIDE_AK;
      pb += TK * STRIDE_BK;
      // reduction loop
      float acc[TM, TN] = 0;
      for(int k = K; k > 0; k -= TK){
        bool checka[TM, TK] = k > TK;
        bool checkb[TK, TN] = k > TK;
        acc += a @ b;
        a = *?(checka)pa;
        b = *?(checkb)pb;
        pa += TK * STRIDE_AK;
        pb += TK * STRIDE_BK;
      }
      acc = acc * alpha;
      TYPE c[TM, TN] = acc;

      // epilogue
      int rcm[TM] = pidm * TM + 0 ... TM;
      int rcn[TN] = pidn * TN + 0 ... TN;
      int offc[TM, TN] = rcm[:, newaxis] * ldc + rcn[newaxis, :];
      TYPE* pc[TM, TN] = C + offc;
      bool checkc[TM, TN] = rcm[:, newaxis] < M &&
                            rcn[newaxis, :] < N;
#if (TZ==1)
      *?(checkc) pc = c;
#else
      // accumulate partial result using spin-locks
      int *plock  = locks + rid;
      int *pcount = plock + get_num_programs(0) * get_num_programs(1);
      for(int repeat = 1; repeat == 1; repeat = atomic_cas(plock, 0, 1));
      int count = *pcount;
      if(count == 0)
        *?(checkc) pc = c;
      else
        *?(checkc) pc = c + *?(checkc)pc;
      atomic_xchg(pcount, (count + 1) % TZ);
      atomic_xchg(plock, 0);
#endif
}
)";

}

enum dtype_t {
  FLOAT,
  HALF,
  DOUBLE
};

template<class T>
struct to_string;

template<> struct to_string<half_float::half>{
  static constexpr const char* value = "half";
};

template<> struct to_string<float>{
  static constexpr const char* value = "float";
};

template<> struct to_string<double>{
  static constexpr const char* value = "double";
};

template<class T>
void triton_dot(drv::context* context,  drv::stream* stream, bool AT, bool BT,
                int32_t M, int32_t N, int32_t K,
                const std::vector<int>& a_order, const std::vector<int>& b_order,
                std::vector<double>& bench, bool &test){
  std::string ty = to_string<T>::value;
  size_t dt_nbytes = sizeof(T);
  drv::device* device = context->device();
  int32_t lda = (AT ^ a_order[0]==1) ? K : M;
  int32_t ldb = (BT ^ b_order[0]==1) ? N : K;
  int32_t ldc = N;
  std::vector<std::string> sa = { "1", "lda" };
  std::vector<std::string> sb = { "1", "ldb" };
  // inputs
  auto dc     = std::shared_ptr<drv::buffer>(drv::buffer::create(context, M*N*dt_nbytes));
  auto da     = std::shared_ptr<drv::buffer>(drv::buffer::create(context, M*K*dt_nbytes));
  auto db     = std::shared_ptr<drv::buffer>(drv::buffer::create(context, K*N*dt_nbytes));
  auto dlocks = std::shared_ptr<drv::buffer>(drv::buffer::create(context, 1024*1024*2*4));
  // initialize buffers
  std::vector<T> hc(M*N);
  std::vector<T> ha(M*K);
  std::vector<T> hb(K*N);
  for(size_t i = 0; i < ha.size(); i++)
    ha[i] = (float)rand()/RAND_MAX;
  for(size_t i = 0; i < hb.size(); i++)
    hb[i] = (float)rand()/RAND_MAX;
  // copy buffer
  stream->write(&*da, true, 0, ha);
  stream->write(&*db, true, 0, hb);

  // macros
  rt::options_space_t opts;
  // A access patterns
  opts.defines.push_back({"STRIDE_AK", {AT? sa[a_order[0]] : sa[a_order[1]] }});
  opts.defines.push_back({"STRIDE_AM", {AT? sa[a_order[1]] : sa[a_order[0]] }});
  // B access patterns
  opts.defines.push_back({"STRIDE_BK", {BT? sb[b_order[1]] : sb[b_order[0]] }});
  opts.defines.push_back({"STRIDE_BN", {BT? sb[b_order[0]] : sb[b_order[1]] }});
  // data-type
  opts.defines.push_back({"TYPE", {ty}});
  // tile sizes
  opts.defines.push_back({"TM", {"128"}});
  opts.defines.push_back({"TN", {"128"}});
  opts.defines.push_back({"TK", {"32"}});
  opts.defines.push_back({"TZ", {"1"}});
  opts.num_warps = {4};

  // arguments
  std::stringstream oss;
  rt::add_arg(oss, *da->cu());
  rt::add_arg(oss, *db->cu());
  rt::add_arg(oss, *dc->cu());
  rt::add_arg(oss, (float)1);
  rt::add_arg(oss, M);
  rt::add_arg(oss, N);
  rt::add_arg(oss, K);
  rt::add_arg(oss, lda);
  rt::add_arg(oss, ldb);
  rt::add_arg(oss, ldc);
  rt::add_arg(oss, *dlocks->cu());
  // kernel
  rt::function function(src::dot, opts, device);
  // grid
  auto ceil = [](size_t x, size_t y) { return (x + y - 1) / y; };
  auto grid = [ceil, M, N](const rt::options_t& x) {
    return rt::grid_t{ceil(M, x.D<int>("TM"))*
                      ceil(N, x.D<int>("TN")),
                      (size_t)x.D<int>("TZ")};
  };

  // metrics
  auto tflops = [&](double nanosec) { return 2.*M*N*K / nanosec * 1e-3; };
  double triton_ns = triton::tools::bench([&]() { function((void**)oss.str().data(), oss.str().size(), grid, stream);}, stream);
  bench.push_back(tflops(triton_ns));
}

std::vector<double> bench_dot(drv::context* context, drv::stream* stream,
               dtype_t dtype, bool AT, bool BT,
               int32_t M, int32_t N, int32_t K,
               const std::vector<int>& a_order, const std::vector<int>& b_order) {
  std::vector<double> bench;
  bool test;
  switch(dtype){
    case HALF:   triton_dot<half_float::half>(context, stream, AT, BT, M, N, K, a_order, b_order, bench, test); break;
    case FLOAT:  triton_dot<float>(context, stream, AT, BT, M, N, K, a_order, b_order, bench, test); break;
    case DOUBLE: triton_dot<double>(context, stream, AT, BT, M, N, K, a_order, b_order, bench, test); break;
    default: break;
  }
  return bench;
}


int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context->backend());
  // shapes to benchmark
  typedef std::tuple<std::vector<int>, bool, bool, int, int, int> config_t;
  std::vector<config_t> configs = {
    {{1, 0}, false, false, 8192, 8192, 8192}
  };
  // does the work
  std::vector<int> ord;
  bool AT, BT;
  int32_t M, N, K;
  for(const auto& c: configs){
    std::tie(ord, AT, BT, M, N, K) = c;
    std::cout << "// " << AT << ", " << BT << ", " << M << ", " << N << ", " << K ;
    for(auto perf: bench_dot(context, stream, HALF, AT, BT, M, N, K, ord, ord))
      std::cout << ", " << perf << std::flush;
    std::cout << std::endl;
  }
}
