#include "aligned-allocator.h"
#include "igemm_init.h"
/*KERNEL IS INCLUDED FROM HERE!*/
#include "igemm_kernel_sse.h"
/*KERNEL IS INCLUDED FROM HERE!*/
#include "igemm_packing.h"
#include "my_igemm.h"
#include "compute_backend.h"
#include <vector>

// REMOVE
#include "data_util.h"
// REMOVE

#define XNN_LOG2_SIZEOF_FLOAT 2
#define XNN_EXTRA_BYTES 16

inline static size_t min(size_t a, size_t b) { return (b < a) ? b : a; }

void igemm(float *k, float *a, float *b, float* c_ret, size_t input_height,
           size_t input_width, size_t kernel_height, size_t kernel_width,
           size_t padding_height, size_t padding_width, size_t subsampling,
           size_t dilation, size_t group_input_channels,
           size_t group_output_channels, unsigned int mr, unsigned int nr,
           unsigned int kr, unsigned int sr) {
  /*
      i : indirection buffer
      k : conv kernels
      a : input
      b : bias?
      c : output
      w : packed weights -> packing during pack_f32_conv
      z : zero buffer (for padding) : only variable that will be initialized and
     deleted during igemm function
  */

  size_t kernel_size = kernel_height * kernel_width;
  const size_t output_pixel_stride = group_output_channels;
  const size_t input_pixel_stride = group_input_channels;
  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t padding_left = padding_width / 2;
  const size_t padding_top = padding_height / 2;
  const size_t output_height =
    (input_height + padding_height - effective_kernel_height) / subsampling + 1;
  const size_t output_width =
    (input_width + padding_width - effective_kernel_width) / subsampling + 1;
  const size_t output_size = output_height * output_width;

  const size_t mc_stride = RoundUp<size_t>(output_size, mr);
  const size_t nc_stride = RoundUp<size_t>(group_output_channels, nr);
  const size_t kc_stride = RoundUp<size_t>(group_input_channels, kr * sr);

  const size_t w_elements = kernel_size * kc_stride * nc_stride + nc_stride;
  const size_t i_elements = mc_stride * kernel_size;
  const size_t c_elements = output_height * output_width * output_pixel_stride;

  const size_t num_buffers =
    1 + DivideRoundUp<size_t>(GetMaxCacheSize(),
                              sizeof(float) * (w_elements + c_elements) +
                                sizeof(void *) * i_elements);

  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  std::vector<float> z(group_input_channels + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float, AlignedAllocator<float, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0.0F);

  xnn_pack_f32_conv_goki_w(
    /*groups=*/1, group_output_channels, kernel_size, group_input_channels, nr,
    kr, sr, k, b, /*scale=*/nullptr, w.data(), /*extra_bytes=*/0,
    /*params=*/nullptr);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_elements, w.begin() + n * w_elements);
  }

  std::vector<const float *> id_buf(i_elements * num_buffers);

  const size_t tiled_output_size = round_up(output_size, mr);
  const void **indirection_buffer =
    reinterpret_cast<const void **>(id_buf.data());

  xnn_indirection_init_conv2d(
    /*output_tile_size=*/mr,
    /*output_start=*/0,
    /*output_end=*/tiled_output_size, indirection_buffer, a, z.data(),
    input_pixel_stride << XNN_LOG2_SIZEOF_FLOAT, input_height, input_width,
    output_height, output_width, kernel_height, kernel_width, subsampling,
    subsampling, dilation, dilation, padding_top, padding_left);
  
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(id_buf.cbegin(), id_buf.cbegin() + i_elements,
              id_buf.begin() + n * i_elements);
  }

  size_t buffer_index = 0;
  buffer_index = (buffer_index + 1) % num_buffers;
  // std::cout<<"buffer_index : " << buffer_index << std::endl;
  // std::cout << " buffer_index * c_elements : "<<  buffer_index * c_elements<<std::endl;



  for (uint32_t m = 0; m < output_size; m += mr) {
    const uint32_t mb = min(output_size - m, mr);
    xnn_f32_igemm_minmax_ukernel_1x8__sse_load1(
      mb, group_output_channels, group_input_channels * sizeof(float),
      kernel_size * mr * sizeof(void *),
      id_buf.data() + buffer_index * i_elements + m,
      w.data() + buffer_index * w_elements,
      c.data() + buffer_index * c_elements + m * group_output_channels,
      group_output_channels * sizeof(float), nr * sizeof(float), 0, z.data()
      // , &params
    );
      // std::cout << "C [" << buffer_index * c_elements + m * group_output_channels << " ] : " <<  *(c + buffer_index * c_elements + m * group_output_channels) << std::endl;
  }

  // copy back c_ret from c
  // this code should also be implemented with SIMD:(HW-specific)!
  scopy(c.data() + buffer_index * c_elements, c_ret, c_elements);
}
