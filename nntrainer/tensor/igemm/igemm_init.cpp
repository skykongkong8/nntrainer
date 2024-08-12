#include "igemm_init.h"
#include "fxdiv.h"
#include <math.h>

static size_t min(size_t a, size_t b) {
  return (b < a) ? b : a;
}

void xnn_indirection_init_conv2d(
  size_t output_tile_size,
  size_t output_start,
  size_t output_end,
  const void** indirection_buffer,
  const void* input,
  const void* zero_buffer,
  size_t input_pixel_stride,
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  size_t kernel_height,
  size_t kernel_width,
  size_t stride_height,
  size_t stride_width,
  size_t dilation_height,
  size_t dilation_width,
  size_t input_padding_top,
  size_t input_padding_left)
{
  const size_t output_size = output_height * output_width;
  const size_t kernel_size = kernel_height * kernel_width;

  const struct fxdiv_divisor_size_t output_width_divisor = fxdiv_init_size_t(output_width);

  for (size_t output_tile_start = output_start; output_tile_start < output_end; output_tile_start += output_tile_size) {
    for (size_t output_tile_offset = 0; output_tile_offset < output_tile_size; output_tile_offset++) {
      const size_t output_index = min(output_tile_start + output_tile_offset, output_size - 1);
      const struct fxdiv_result_size_t output_y_x = fxdiv_divide_size_t(output_index, output_width_divisor);
      const size_t output_x = output_y_x.remainder;
      const size_t output_y = output_y_x.quotient;
      for (size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
        const size_t input_y = output_y * stride_height + kernel_y * dilation_height - input_padding_top;
        if (input_y < input_height) {
          for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
            const size_t input_x = output_x * stride_width + kernel_x * dilation_width - input_padding_left;
            const size_t kernel_index = kernel_y * kernel_width + kernel_x;
            const size_t index = output_tile_start * kernel_size + kernel_index * output_tile_size + output_tile_offset;
            if (input_x < input_width) {
              indirection_buffer[index] = (const void*)
                ((uintptr_t) input + (input_y * input_width + input_x) * input_pixel_stride);
            } else {
              indirection_buffer[index] = zero_buffer;
            }
          }
        } else {
          for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
            const size_t kernel_index = kernel_y * kernel_width + kernel_x;
            const size_t index = output_tile_start * kernel_size + kernel_index * output_tile_size + output_tile_offset;
            indirection_buffer[index] = zero_buffer;
          }
        }
      }
    }
  }
}
