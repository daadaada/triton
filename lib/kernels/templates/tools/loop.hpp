#include "isaac/kernels/stream.h"
#include "isaac/kernels/templates/base.h"
#include <string>
#include "to_string.hpp"

namespace isaac
{
namespace templates
{

inline void fetching_loop_info(fetching_policy_type policy, std::string const & bound, kernel_generation_stream & stream, std::string & init, std::string & upper_bound, std::string & inc, std::string const & domain_id, std::string const & domain_size, driver::Device const & device)
{
  if (policy==FETCH_FROM_GLOBAL_STRIDED)
  {
    init = domain_id;
    upper_bound = bound;
    inc = domain_size;
  }
  else if (policy==FETCH_FROM_GLOBAL_CONTIGUOUS)
  {
    std::string _size_t = size_type(device);
    std::string chunk_size = "chunk_size";
    std::string chunk_start = "chunk_start";
    std::string chunk_end = "chunk_end";

    stream << _size_t << " " << chunk_size << " = (" << bound << "+" << domain_size << "-1)/" << domain_size << ";" << std::endl;
    stream << _size_t << " " << chunk_start << " =" << domain_id << "*" << chunk_size << ";" << std::endl;
    stream << _size_t << " " << chunk_end << " = min(" << chunk_start << "+" << chunk_size << ", " << bound << ");" << std::endl;
    init = chunk_start;
    upper_bound = chunk_end;
    inc = "1";
  }
}


template<class Fun>
inline void element_wise_loop_1D(kernel_generation_stream & stream, fetching_policy_type fetch, unsigned int simd_width,
                                 std::string const & i, std::string const & bound, std::string const & domain_id, std::string const & domain_size, driver::Device const & device, Fun const & generate_body)
{
  std::string strwidth = tools::to_string(simd_width);
  std::string boundround = bound + "/" + strwidth;

  std::string init, upper_bound, inc;
  fetching_loop_info(fetch, boundround, stream, init, upper_bound, inc, domain_id, domain_size, device);
  stream << "for(unsigned int " << i << " = " << init << "; " << i << " < " << upper_bound << "; " << i << " += " << inc << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  generate_body(simd_width);
  stream.dec_tab();
  stream << "}" << std::endl;

  if (simd_width>1)
  {
    stream << "for(unsigned int " << i << " = " << boundround << "*" << strwidth << " + " << domain_id << "; " << i << " < " << bound << "; " << i << " += " + domain_size + ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    generate_body(1);
    stream.dec_tab();
    stream << "}" << std::endl;
  }
}

}
}
