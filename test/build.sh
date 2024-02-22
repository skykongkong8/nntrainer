cd jni;
ndk-build;
cd ../libs/arm64-v8a;
adb push . /data/local/tmp/tensor_datatype/