#include <iostream>
#include <random>
#include <cmath>

template <typename T>
void generate_matrix(unsigned int row, unsigned int col, unsigned int ldm,
                     T *mat, T min = 1, T max = 1) {
  std::uniform_real_distribution<double> dist(min, max);
  for (unsigned int i = 0; i < row; ++i) {
    for (unsigned int j = 0; j < col; ++j) {
      std::default_random_engine gen((i + 1) * (j + 42));
      T val = dist(gen);
      mat[i * ldm + j] = val;
    }
  }
}

template <typename T>
void print_matrix(unsigned int row, unsigned int col, unsigned int ldm,
                  T *mat) {
  for (unsigned int i = 0; i < row; ++i) {
    for (unsigned int j = 0; j < col; ++j) {
      std::cout << mat[i * ldm + j] << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <typename T>
void compare_matrix(unsigned int row, unsigned int col, unsigned int ldm, T *A,
                    T *B) {
  for (unsigned int i = 0; i < row; ++i) {
    for (unsigned int j = 0; j < col; ++j) {
      if (A[i * ldm + j] != B[i * ldm + j]) {
        std::cout << "Diff at : " << i << " , " << j << " | " << A[i * ldm + j]
                  << " VS " << B[i * ldm + j] << std::endl;
      }
    }
  }
}

template<typename T>
void test_summary(size_t input_height, size_t input_width, size_t kernel_height,
                  size_t kernel_width, size_t output_height,
                  size_t output_width, size_t group_input_channels, size_t group_output_channels) {
  std::cout <<"[ TEST SUMMARY ]\n";
  std::cout << "input_height : " << input_height << std::endl;
  std::cout << "input_width : " << input_width << std::endl;
  std::cout << "input_channel : " << group_input_channels << std::endl;
  std::cout << "kernel_height : " << kernel_height << std::endl;
  std::cout << "kernel_width : " << kernel_width << std::endl;
  std::cout << "output_height : " << output_height << std::endl;
  std::cout << "output_width : " << output_width << std::endl;
  std::cout << "output_channel : " << group_output_channels << std::endl;
  std::cout << std::endl;
}

template <typename T>
void check_nan(T* src, size_t length){
  for (int i = 0; i < length; ++i){
    if (std::isnan(src[i])){
      std::cout << "[ WARNING ] NaN value at the idx of : " << i << std::endl;
    }
  }
}
