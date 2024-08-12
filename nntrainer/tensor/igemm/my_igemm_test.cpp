#include "data_util.h"
#include "my_igemm.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#define TEST_EXTRA_BYTES 16

void my_igemm_test(size_t input_height, size_t input_width,
                   size_t input_pixel_stride, size_t group_output_channels,
                   size_t group_input_channels, size_t kernel_height,
                   size_t kernel_width, size_t padding_height,
                   size_t padding_width, size_t dilation, size_t subsampling,
                   unsigned int mr, unsigned int nr, unsigned int kr,
                   unsigned int sr) {
  // 1. generate input data
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng =
    std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), std::ref(rng));

  std::vector<float> a(input_height * input_width * input_pixel_stride +
                       TEST_EXTRA_BYTES / sizeof(float));
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<float> k(group_output_channels * kernel_height * kernel_width *
                       group_input_channels);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  std::vector<float> b(group_output_channels);
  std::generate(b.begin(), b.end(), std::ref(f32rng));

// output params : just for convenience, no need to be coded here
  const size_t output_pixel_stride = group_output_channels;
  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t output_height =
    (input_height + padding_height - effective_kernel_height) / subsampling + 1;
  const size_t output_width =
    (input_width + padding_width - effective_kernel_width) / subsampling + 1;
  const size_t c_elements = output_height * output_width * output_pixel_stride;

  std::vector<float> c_ret(c_elements);
  std::fill(c_ret.begin(), c_ret.end(), std::nanf(""));

  // DELETE THIS BLOCK!
  test_summary<float>(input_height, input_width, kernel_height, kernel_width,
                      output_height, output_width, group_input_channels,
                      group_output_channels);
  // DELETE THIS BLOCK!

  // 2. run igemm kernel
  igemm(k.data(), a.data(), b.data(), c_ret.data(), input_height, input_width,
        kernel_height, kernel_width, padding_height, padding_width, subsampling,
        dilation, group_input_channels, group_output_channels, mr, nr, kr, sr);

  // 3. run ground truth algorithm
  // ground_truth_conv

  // 4. compare time / accuracy
  // DELETE THIS BLOCK!
  print_matrix<float>(3, 3, output_width, c_ret.data());
  check_nan<float>(c_ret.data(), c_ret.size());
  // DELETE THIS BLOCK!

  // Note that igemm kernel should be fed from here with current design
}

int main() {
  //     /*********************** Conv 1 **********************/
  //   /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  //   b->Args({224, 224,  3,  3,  2,  2, 2, 1,    3,   24});
  my_igemm_test(/*IH*/ 224, /*IW*/ 224, /*S?*/ 2, /*GCout*/ 24, /*GCin*/ 3,
                /*KH*/ 3, /*KW*/ 3, /*PH*/ 2, /*PW*/ 2, /*D*/ 1,
                /*subsampling=pooling=stride_h,w*/ 1, /*mr*/ 1, /*nr*/ 8,
                /*kr*/ 1, /*sr*/ 1);

  std::cout << "[INFO] MY IGEMM TEST HAS SUCCESSFULLY FINISHED!\n";
  return 0;
}
