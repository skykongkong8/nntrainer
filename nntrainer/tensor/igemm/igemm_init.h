#include <stddef.h>
#include <stdint.h>

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
  size_t input_padding_left);
