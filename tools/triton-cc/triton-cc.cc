#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/analysis/allocation.h"
#include "triton/codegen/analysis/liveness.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/analysis/swizzle.h"
#include "triton/codegen/target.h"
#include "triton/codegen/transform/coalesce.h"
#include "triton/codegen/transform/dce.h"
#include "triton/codegen/transform/peephole.h"
#include "triton/codegen/transform/membar.h"
#include "triton/codegen/transform/reassociate.h"
#include "triton/codegen/transform/reorder.h"
#include "triton/codegen/transform/cts.h"
#include "triton/codegen/transform/disassociate.h"
#include "triton/codegen/selection/generator.h"
#include "triton/lang/token.h"
#include "triton/runtime/function.h"
#include "triton/lang/cpp.h"
#include "triton/lang/parser.h"
#include "triton/lang/code_gen.h"
#include "triton/driver/backend.h"
#include "triton/driver/device.h"
#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"
#include "triton/driver/module.h"
#include "triton/driver/error.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/print.h"
#include "triton/runtime/error.h"
#include "triton/tools/bench.hpp"
#include "triton/tools/sha1.hpp"
#include "triton/tools/sys/getenv.hpp"
#include "triton/tools/sys/mkdir.hpp"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"

#include <exception>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
// Usage:
// triton-cc --num_warps 4 gemm.triton 
// output:
// gemm.ll

using namespace triton;
using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"));

static cl::opt<unsigned>
NumWarps("num_warps", cl::desc("number of warps"), cl::value_desc("num_warps"));

static std::string preheader() {
  return  R"(
#define bool _Bool
#define true 1
#define false 0

#define __readonly      __attribute__((readonly))
#define __writeonly     __attribute__((writeonly))
#define __noalias       __attribute__((noalias))
#define __aligned(A)    __attribute__((aligned(A)))
#define __multipleof(A) __attribute__((multipleof(A)))
#define __retune        __attribute__((retune))

#define F32_INFINITY bitcast<float>(0x7F800000)
#define F16_INFINITY bitcast<half>((int16)0x7C00)

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

#define PASTER(a, b, _) a ## _ ## b
#define EVALUATOR(a, b, _)  PASTER(a, b, _)
#define atomic_add(TYPE, TM, TN) EVALUATOR(atomic_add, EVALUATOR(TYPE, EVALUATOR(TM, TN, x), _), _)
#define DECLARATION(TYPE, TM, TN) extern void atomic_add(TYPE, TM, TN)(TYPE*[TM, TN], TYPE[TM, TN], bool[TM, TN])

DECLARATION(float, 64, 64);
DECLARATION(float, 64, 128);
DECLARATION(float, 128, 64);
DECLARATION(float, 128, 128);
extern void atomic_add_half_1x1(half*, half, bool);

DECLARATION(half , 64, 64);
DECLARATION(half , 64, 128);
DECLARATION(half , 128, 64);
DECLARATION(half , 128, 128);
extern void atomic_add_float_1x1(float*, float, bool);

extern int atomic_cas(int*, int, int);
extern int atomic_xchg(int*, int);
extern int get_program_id(int);
extern int get_num_programs(int);
extern int select(bool, int, int);
extern char __constant__ * calloc(int);

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long uint64;
typedef char int8;
typedef short int16;
typedef int int32;
typedef long int64;
)";
}


static std::unique_ptr<ir::module> parseTritonIRFile(std::string filename) {
  std::ifstream f(filename);
  if (!f.is_open()) {
    throw std::runtime_error(filename + " cannot be opened.\n");
  }
  std::stringstream buffer;
  buffer << f.rdbuf();
  std::string src = buffer.str();

  ir::context ctx;
  std::unique_ptr<ir::module> M = std::make_unique<ir::module>("", ctx);

  src = preheader() + src;
  // parse
  TokenSequence tokens;
  Preprocessor cpp(&src, true);
  cpp.Process(tokens);
  Parser parser(tokens);
  parser.Parse();
  Generator gen(&parser);
  gen.Gen(M.get());

  return M;
}

static void runPasses(ir::module & M) {
  std::unique_ptr<codegen::target> target = 
    std::make_unique<codegen::nvidia_cu_target>(70);
  
  LLVMContext ctx;
  std::string name = M.get_function_list()[0]->get_name();
  auto llvm = std::make_unique<llvm::Module>(name, ctx);

  bool cts_use_async = false;

  // create passes
  printf("create passes\n");
  codegen::analysis::align align;
  codegen::analysis::axes axes;
  codegen::transform::cts cts(cts_use_async);
  codegen::transform::disassociate disassociate;
  codegen::analysis::layouts layouts(&axes, &align, NumWarps, target.get());
  codegen::analysis::liveness liveness(&layouts);
  codegen::analysis::swizzle swizzle(&layouts, target.get());
  codegen::analysis::allocation allocation(&liveness);
  codegen::transform::membar barriers(&liveness, &layouts, &allocation);
  codegen::transform::dce dce;
  codegen::transform::peephole peephole(target.get());
  codegen::transform::reassociate reassociate;
  codegen::transform::coalesce coalesce(&align, &layouts);
  codegen::generator isel(&axes, &layouts, &align, &allocation, &swizzle, 
                          target.get(), NumWarps);
  printf("run passes\n");
  dce.run(M);
  disassociate.run(M);
  dce.run(M);
  peephole.run(M);
  dce.run(M);
  align.run(M);
  if(target->is_gpu())
    cts.run(M);
  axes.run(M);
  layouts.run(M);
  coalesce.run(M);
  dce.run(M);
  align.run(M);
  dce.run(M);
  if(target->is_gpu()){
    reassociate.run(M);
    cts.run(M);
  }
  peephole.run(M);
  dce.run(M);
  align.run(M);
  axes.run(M);
  layouts.run(M);
  swizzle.run(M);
  liveness.run(M);
  allocation.run(M);

  barriers.run(M);
  isel.visit(M, *llvm);
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  std::unique_ptr<ir::module> M;
  try {
    M = parseTritonIRFile(InputFilename);
  } catch (const std::exception &e) {
    std::cout << e.what();
    return -1;
  }
  print(*M, std::cout);
  runPasses(*M);

  return 0;
}