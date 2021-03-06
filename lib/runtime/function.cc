#include <string>
#include <mutex>
#include <regex>
#include <functional>
#include <algorithm>
#include <sstream>
#include <memory>
#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/analysis/allocation.h"
#include "triton/codegen/analysis/liveness.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/analysis/swizzle.h"
#include "triton/codegen/transform/coalesce.h"
#include "triton/codegen/transform/dce.h"
#include "triton/codegen/transform/peephole.h"
#include "triton/codegen/transform/membar.h"
#include "triton/codegen/transform/reassociate.h"
#include "triton/codegen/transform/reorder.h"
#include "triton/codegen/transform/cts.h"
#include "triton/codegen/transform/disassociate.h"
#include "triton/codegen/selection/generator.h"
#include "triton/runtime/function.h"
#include "triton/lang/cpp.h"
#include "triton/lang/parser.h"
#include "triton/lang/code_gen.h"
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
#include <mutex>
#include <fstream>

std::mutex mut;

namespace triton{
namespace runtime {

/* --------------------------------- */
/* --------------------------------- */
/* --------------------------------- */

arg_type kernel::convert(ir::type *ty) {
  if(ty->is_integer_ty(1))  return INT1_T;
  if(ty->is_integer_ty(8))  return INT8_T;
  if(ty->is_integer_ty(16)) return INT16_T;
  if(ty->is_integer_ty(32)) return INT32_T;
  if(ty->is_integer_ty(64)) return INT64_T;
  if(ty->is_half_ty())      return HALF_T;
  if(ty->is_float_ty())     return FLOAT_T;
  if(ty->is_double_ty())    return DOUBLE_T;
  if(ty->is_pointer_ty())   return BUFFER_T;
  throw std::runtime_error("unknown type");
}


std::string kernel::preheader() {
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

void kernel::init_ir(const std::string& src) {
  // pre-process
  TokenSequence tokens;
  Preprocessor cpp(&src, true);
  for(auto it: opt.defines)
    cpp.AddMacro(it.first, &it.second);
  cpp.Process(tokens);
  // src -> ast
  Parser parser(tokens);
  parser.Parse();
  // ast -> triton-ir
  ir::module* module = new ir::module("", ctx_);
  Generator gen(&parser);
  gen.Gen(module);
  ir_.reset(module);
}

void kernel::init_ker(){
  // triton-ir -> binary
  std::unique_ptr<driver::module> bin;
  std::unique_ptr<codegen::target> target = dev_->make_target();
  // generate llvm code
  llvm::LLVMContext ctx;
  std::string name = ir_->get_function_list()[0]->get_name();
  std::unique_ptr<llvm::Module> llvm(new llvm::Module(name, ctx));
  // optimizations
  bool cts_use_async = target->as_nvidia()->sm() >= 80;
  // create passes
  codegen::analysis::align align;
  codegen::analysis::axes axes;
  codegen::transform::cts cts(cts_use_async);
  codegen::transform::disassociate disassociate;
  codegen::analysis::layouts layouts(&axes, &align, opt.num_warps, target.get());
  codegen::analysis::liveness liveness(&layouts);
  codegen::analysis::swizzle swizzle(&layouts, target.get());
  codegen::analysis::allocation allocation(&liveness);
  codegen::transform::membar barriers(&liveness, &layouts, &allocation);
  codegen::transform::dce dce;
  codegen::transform::peephole peephole(target.get());
  codegen::transform::reassociate reassociate;
  codegen::transform::coalesce coalesce(&align, &layouts);
  codegen::generator isel(&axes, &layouts, &align, &allocation, &swizzle, target.get(), opt.num_warps);
  // run passes
  dce.run(*ir_);
  disassociate.run(*ir_);
  dce.run(*ir_);
  peephole.run(*ir_);
  dce.run(*ir_);
  align.run(*ir_);
  if(target->is_gpu())
    cts.run(*ir_);
  axes.run(*ir_);
  layouts.run(*ir_);
  coalesce.run(*ir_);
  dce.run(*ir_);
  align.run(*ir_);
  dce.run(*ir_);
  if(target->is_gpu()){
    reassociate.run(*ir_);
    cts.run(*ir_);
  }
  peephole.run(*ir_);
  dce.run(*ir_);
  align.run(*ir_);
  axes.run(*ir_);
  layouts.run(*ir_);
  swizzle.run(*ir_);
  liveness.run(*ir_);
  allocation.run(*ir_);
  if(allocation.allocated_size() > dev_->max_shared_memory())
    throw exception::out_of_shared_memory();
  barriers.run(*ir_);
  isel.visit(*ir_, *llvm);
  //if(res->spilled() > 256)
  //  throw exception::out_of_registers();
  mod_.reset(driver::module::create(dev_, std::move(llvm)));
  ker_.reset(driver::kernel::create(&*mod_, name.c_str()));
}

void kernel::init_sig() {
  ir::function* fn = ir_->get_function_list()[0];
  ir::function_type* ty = fn->get_fn_type();
  for(size_t i = 0; i < ty->get_num_params(); i++){
    sig_.push_back(convert(ty->get_param_ty(i)));
    if(!fn->has_attr(i+1))
      continue;
  }
}

kernel::kernel(const std::string& src, const options_t& opt, driver::device *dev):
  opt(opt), dev_(dev) {
  init_ir(preheader() + src);
  init_ker();
  init_sig();
}

void kernel::operator()(void *args, size_t args_size, driver::stream *stream, const std::vector<size_t>& _grid) const{
  // set grid
  if(_grid.size() > 3)
    throw std::runtime_error("grid size must be no greater than 3");
  std::array<size_t, 3> grid;
  for(size_t i = 0; i < 3; i++)
    grid[i] = (i < _grid.size()) ? _grid[i] : 1;
  // enqueue
  stream->enqueue(&*ker_, grid, {opt.num_warps * 32, 1, 1}, args, args_size);
}

/* --------------------------------- */
/* --------------------------------- */
/* --------------------------------- */

void function::do_loop_nest(std::vector<size_t> const & ranges,
                       std::function<void(std::vector<size_t> const &)> const & f){
  size_t D = ranges.size();
  std::vector<size_t> values(D, 0);
  size_t i = D - 1;
  while(true){
    f(values);
    while(values[i]++ == ranges[i] - 1){
      if(i == 0)
        return;
      values[i--] = 0;
    }
    i = D - 1;
  }
}


void function::init_kernels(const std::string& src, const options_space_t& opts, driver::device *device) {
  // all ranges
  std::vector<size_t> ranges;
  ranges.push_back(opts.num_warps.size());
  for(const auto& x: opts.defines)
    ranges.push_back(x.second.size());
  // functor for source with given option
  std::vector<std::pair<options_t, std::string>> err;
  auto do_make = [&](std::vector<size_t> params) {
    // compilation options
    unsigned i = 0;
    options_t opt;
    opt.num_warps = opts.num_warps[params[i++]];
    for(auto D: opts.defines)
      opt.defines[D.first] = D.second[params[i++]];
    // compile
    try{
      kernels_.push_back({opt, std::make_shared<kernel>(src, opt, device)});
    }catch(const exception::base& e){
      err.push_back({opt, e.what()});
    }
  };
  // multi-threaded compilation
  do_loop_nest(ranges, do_make);
  if(kernels_.empty()){
    std::ostringstream dbg;
    dbg << "Auto-Tuner could not find any valid configuration:" << std::endl;
    for(auto x: err){
      dbg << "[ ";
      dbg << x.first.num_warps << ", ";
      dbg << "{ ";
      for(const auto& y: x.first.defines)
        dbg << '"' << y.first << "\"= \"" << y.second << "\", ";
      dbg << " } ] -> " << x.second << std::endl;
    }
    throw exception::no_valid_configuration(dbg.str());
  }
}

kernel* function::autotune(void* args, size_t args_size, const grid_fn_ty& grid_fn, driver::stream* stream) {
  // fast path -- no autotuning necessary
  if(kernels_.size() == 1)
    return &*kernels_.begin()->second;
  // auto-tuning key
  std::vector<uint64_t> key;
  auto it = cache_.find(key);
  if(it != cache_.end())
    return it->second;
  // run auto-tuner
  double best_ts = INFINITY;
  kernel* ret = nullptr;
  for(auto &x : kernels_){
    kernel* current = &*x.second;
    auto grid = grid_fn(x.first);
    while(grid.size() < 3)
      grid.push_back(1);
    double ts = tools::bench([&]() { (*current)(args, args_size, stream, grid); },
                                     stream, true);
    ret = (ts < best_ts) ? current : ret;
    best_ts = std::min(ts, best_ts);
  }
  stream->synchronize();
  it = cache_.insert({key, ret}).first;
  return it->second;
}

function::function(const std::string& src, const options_space_t& opt, driver::device *device) {
  init_kernels(src, opt, device);
}

void function::operator()(void* args, size_t args_size, const grid_fn_ty& grid_fn, driver::stream *stream) {
  runtime::kernel* fn = autotune(args, args_size, grid_fn, stream);
  (*fn)(args, args_size, stream, grid_fn(fn->opt));
}

void function::operator()(void* args, size_t args_size, const grid_t& grid, driver::stream* stream) {
  return this->operator()(args, args_size, [&grid](const options_t&){ return grid; }, stream);
}


}
}
