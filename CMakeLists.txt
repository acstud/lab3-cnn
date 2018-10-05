cmake_minimum_required(VERSION 3.10)

project(mnist_cnn)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")

add_executable(${PROJECT_NAME}
		src/cnn/cnn.h
		src/cnn/conv_layer_t.h
		src/cnn/fc_layer.h
		src/cnn/gradient_t.h
		src/cnn/layer_t.h
		src/cnn/optimization_method.h
		src/cnn/point_t.h
		src/cnn/pool_layer_t.h
		src/cnn/relu_layer_t.h
		src/cnn/tensor_t.h
		src/cnn/types.h
		src/utils/Timer.h
		src/main.cpp
		src/Train.cpp src/Train.h src/Infer.cpp src/Infer.h src/utils/io.cpp src/utils/io.h)
