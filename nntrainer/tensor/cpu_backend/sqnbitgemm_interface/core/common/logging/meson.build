sqnbitgemm_core_logging_headers = [
  'capture.h',
  'isink.h',
  'logging.h',
  'macros.h',
  'severity.h',
  'sink_types.h',
]

subdir('sinks')
nntrainer_inc += include_directories('sinks')
nntrainer_inc_abs += meson.current_source_dir() / 'sinks'

foreach h : sqnbitgemm_core_logging_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
