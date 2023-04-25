NVCC = nvcc
CC = g++
CUDA_LINK_FLAGS =  -rdc=true -arch=compute_75 -code=sm_75
CUDA_COMPILE_FLAGS = -O2 --device-c -arch=compute_75 -code=sm_75
CC_COMPILE_FLAGS =  -I/usr/include/openmpi
CC_LINK_FLAGS = -lm -lstdc++ -lmpi -L/usr/lib -lpthread 

all :  aesCUDA aesCUDAStreams aesMPICPUWorker aesMPI

aesMPI : main.o rijndael_host.o rijndael_device.o
	${NVCC} ${CUDA_LINK_FLAGS} ${CC_LINK_FLAGS} $^ -o $@

aesMPICPUWorker : mainCPUWorker.o rijndael_host.o rijndael_device.o
	${NVCC} ${CUDA_LINK_FLAGS} ${CC_LINK_FLAGS} $^ -o $@

main.o : main.cpp rijndael.h
	${CC} ${CC_COMPILE_FLAGS} -c main.cpp 

mainCPUWorker.o : mainCPUWorker.cpp rijndael.h
	${CC} ${CC_COMPILE_FLAGS} -c mainCPUWorker.cpp 

rijndael_host.o : rijndael_host.cu rijndael.h rijndael_device.h
	${NVCC} ${CUDA_COMPILE_FLAGS} ${CC_COMPILE_FLAGS} -c rijndael_host.cu 

rijndael_device.o : rijndael_device.cu rijndael.h rijndael_device.h
	${NVCC} ${CUDA_COMPILE_FLAGS} ${CC_COMPILE_FLAGS} -c rijndael_device.cu 

aesCUDA : aesCUDA.o rijndael_host.o rijndael_device.o
	${NVCC} ${CUDA_LINK_FLAGS} ${CC_LINK_FLAGS} $^ -o $@

aesCUDA.o : aesCUDA.cu rijndael.h
	${NVCC} ${CUDA_COMPILE_FLAGS} ${CC_COMPILE_FLAGS} -c aesCUDA.cu 


aesCUDAStreams : aesCUDA.o rijndael_host_streams.o rijndael_device.o
	${NVCC} ${CUDA_LINK_FLAGS} ${CC_LINK_FLAGS} $^ -o $@

rijndael_host_streams.o : rijndael_host_streams.cu rijndael.h rijndael_device.h
	${NVCC} ${CUDA_COMPILE_FLAGS} ${CC_COMPILE_FLAGS} -c rijndael_host_streams.cu 

clean:
	rm *.o 
