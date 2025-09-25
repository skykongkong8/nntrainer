sqnbitgemm_headers = [
  'sqnbitgemm_interface.h',
  'mlas_gemm_postprocessor.h',
  'mlas_q4.h',
  'mlas_qnbit.h',
  'mlas.h',
  'mlasi.h',
  'q4common.h',
  'sqnbitgemm_q8_block.h',
  'sqnbitgemm.h',
  'test_util.h',
  'mlas_float16.h',
  'qgemm.h',
  'FgemmKernelCommon.h',
  'SgemmKernelCommon.h',
  'asmmacro.h'
]

sqnbitgemm_sources = [
  'sqnbitgemm_interface.cpp',
  'q4_dq.cpp',
  'sqnbitgemm.cpp',
  'threading.cpp',
  'sqnbitgemm_platform.cpp',
  'qgemm.cpp',
  'qgemm_kernel_default.cpp',
  'quantize.cpp',
  'nntr_sqnbitgemm_threadpool.cpp',
]

subdir('core')
nntrainer_inc += include_directories('core')
nntrainer_inc_abs += meson.current_source_dir() / 'core'

arch = host_machine.cpu_family()
if arch == 'arm' or arch == 'aarch64' or get_option('platform') == 'android'
  subdir('sqnbitgemm_neon_kernels')
  nntrainer_inc += include_directories('sqnbitgemm_neon_kernels')
  nntrainer_inc_abs += meson.current_source_dir() / 'sqnbitgemm_neon_kernels'
elif arch == 'x86_64' or arch == 'x86'
  subdir('sqnbitgemm_avx_kernels')
  nntrainer_inc += include_directories('sqnbitgemm_avx_kernels')
  nntrainer_inc_abs += meson.current_source_dir() / 'sqnbitgemm_avx_kernels'
endif

foreach h : sqnbitgemm_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

foreach s : sqnbitgemm_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach
