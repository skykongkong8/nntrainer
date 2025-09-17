sqnbitgemm_headers = [
  'sqnbitgemm_interface.h',
  'mlas_gemm_postprocessor.h',
  'mlas_q4.h',
  'mlas_qnbit.h',
  'mlas.h',
  'mlasi.h',
  'q4common.h',
  'sqnbitgemm_interface.h',
  'sqnbitgemm_q8_block.h',
  'sqnbitgemm.h',
  'test_util.h',
]

sqnbitgemm_sources = [
  'sqnbitgemm_interface.cpp',
  'q4_dq.cpp',
  'sqnbitgemm.cpp',
  'threading.cpp',
  'sqnbitgemm_platform.cpp',
]

subdir('core')
nntrainer_inc += include_directories('core')
nntrainer_inc_abs += meson.current_source_dir() / 'core'

arch = host_machine.cpu_family()
if arch == 'arm' or arch == 'aarch64' or get_option('platform') == 'android'
  # Add neon-related files
  sqnbitgemm_arm_sources = [
    'sqnbitgemm_kernel_neon_int8.cpp',
    'sqnbitgemm_kernel_neon.cpp',
  ]
  sqnbitgemm_arm_headers = [
    'sqnbitgemm_m1_sym_kernel_avx2_int8_blklen32.h',
    'sqnbitgemm_m1_sym_kernel_avx2_int8_blklen64.h',
  ]
  
  foreach h : sqnbitgemm_arm_headers
    nntrainer_headers += meson.current_source_dir() / h
  endforeach

  foreach s : sqnbitgemm_arm_sources
    nntrainer_sources += meson.current_source_dir() / s
  endforeach

elif arch == 'x86_64' or arch == 'x86'
  # Add avx2-related files
  sqnbitgemm_x86_sources = [
    'sqnbitgemm_kernel_avx2.cpp',
  ]

  sqnbitgemm_x86_headers = [
    'sqnbitgemm_kernel_neon.h',
    'sqnbitgemm_kernel_avx_common_fp32.h',
    'sqnbitgemm_kernel_avx_common_int8.h',
    'sqnbitgemm_kernel_avx_common.h',
    'sqnbitgemm_kernel_avx2_int8_blklen16.h',
    'sqnbitgemm_kernel_avx2_int8_blklen32.h',
    'sqnbitgemm_kernel_avx2_int8_blklen64.h',
  ]

  foreach h : sqnbitgemm_x86_headers
    nntrainer_headers += meson.current_source_dir() / h
  endforeach

  foreach s : sqnbitgemm_x86_sources
    nntrainer_sources += meson.current_source_dir() / s
  endforeach
endif

foreach h : sqnbitgemm_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

foreach s : sqnbitgemm_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach
