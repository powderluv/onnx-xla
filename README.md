# onnx-xla

TODO:

1. Fix bug in python setup.py develop (unable to copy module from build_ext)

2. Experiment with group convolution implementations to improve performance.

3. Use utility macros in onnx_xla and python_onnxifi to perform asserts

4. Clean up onnx_xla/backend.cc macros

5. Maybe move to one client (benchmark this?)

6. Add support for half in python interface to ONNXIFI

7. Add strided numpy array support ot python interface to ONNXIFI

8. Add weight descriptor support to the python interface to ONNXIFI

9. Benchmark two version of LRN(materializing square and not)


Steps to test:

0.01 Install bazel 0.11 
0.02 Checkout grpc referenced in third_party/tensorflow/tensorflow/workspace.bzl .
0.1 Reset the protobuf version in grpc to be same as workspace.bzl above.  
0.2 Build cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/local -DBUILD_SHARED_LIBS=ON && make install. 
0.3 Copy over the .so files since it doesn't install them. 
0.4 Install of the python module will fail. Manually mkdir build/lib.linux-x86_64-3.6 and re-run

1. Run "python setup.py install" or "python setup.py develop"

2. Start an XLA server with "./third_party/tensorflow/bazel-bin/tensorflow/compiler/xla/rpc/grpc_service_main_cpu --port=51000

3. To the backends ability to run a simple IR graph with a Relu operator, "cd build && ./tests"

4. To the backends ability to run a simple ModelProto graph with a Relu operator, "cd build && ./relu_model" 

5. To execute a test using the python wrapper of onnxifi, "python test.py"

6. To run unit tests of node translations, execute a "python onnx_xla_test.py"

