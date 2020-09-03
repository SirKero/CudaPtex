#include <iostream>
#include <Ptexture.h>
#include <memory>
#include <vector>
#include <stdint.h>
#include "ptex.hpp"

#include "unique_device_ptr.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>


using namespace mufflon;


__global__ void cudaTest(float* res ,int faceId, int dimX, int dimY ,int numChannels,  const float* texArr, const uint32_t* texOffsetArr, const uint8_t* ResLog2U, const uint8_t* ResLog2V) {
	unsigned int x = threadIdx.x;
	unsigned int y = blockIdx.x;

	
	float u = __uint2float_rz(threadIdx.x);
	float v = __uint2float_rz(blockIdx.x);
	u /= __int2float_rz(dimX);
	v /= __int2float_rz(dimY);

	float tmpArr[3];
	PtexelFetch(tmpArr, faceId, u, v, numChannels, texArr, texOffsetArr,ResLog2U, ResLog2V);
	
	for (int i = 0; i < numChannels; i++) {
		res[numChannels * (x + dimX * y) + i] = tmpArr[i];
	}
}

__global__ void cudaTest2(float* res, int faceId, int dimX, int dimY, cudaPtexture tex) {
	unsigned int x = threadIdx.x;
	unsigned int y = blockIdx.x;


	float u = __uint2float_rz(threadIdx.x);
	float v = __uint2float_rz(blockIdx.x);
	u /= __int2float_rz(dimX);
	v /= __int2float_rz(dimY);

	float tmpArr[3];
	PtexelFetch(tmpArr, faceId, u, v, tex);

	for (int i = 0; i < tex.numChannels; i++) {
		res[tex.numChannels * (x + dimX * y) + i] = tmpArr[i];
	}
}

int main(){

	int face = 2;

	Ptex::PtexTexture* texture;
	Ptex::String ptex_error;
	cudaPtex pTexture;
	pTexture.loadFile("models/teapot/teapot.ptx", true);
	texture = Ptex::PtexTexture::open("models/teapot/teapot.ptx", ptex_error, true);

	if (texture == nullptr) {
		std::cout << "Error: Could not read ptex texture \n";
	}

	int texSize = texture->numChannels() * texture->getFaceInfo(face).res.size();
	int ResU = texture->getFaceInfo(face).res.u();
	int ResV = texture->getFaceInfo(face).res.v();

	int testFaceSize = pTexture.getNumChannels() * ResU * ResV;
	unique_device_ptr<Device::CUDA, float[]> testRes = make_udevptr_array <Device::CUDA, float, false>(testFaceSize);
	unique_device_ptr<Device::CPU, float[]> cpuRes = make_udevptr_array <Device::CPU, float, false>(testFaceSize);
	unique_device_ptr<Device::CPU, float[]> cpuRes2 = make_udevptr_array <Device::CPU, float, false>(testFaceSize);

	//cudaTest<<<ResV, ResU >>>(testRes.get(), face, ResU, ResV, pTexture.getNumChannels(), pTexture.getDataPointer(),
	//	pTexture.getOffsetPointer(), pTexture.getResLog2U(), pTexture.getResLog2V());

	cudaTest2 << <ResV, ResU >> > (testRes.get(), face, ResU, ResV, pTexture.getTexture());

	cudaMemcpy(cpuRes.get(), testRes.get(), testFaceSize * sizeof(float), cudaMemcpyDefault);

	std::cout << "Cuda Sampled: \n\n";

	for (int y = 0; y < ResV; y++) {
		for (int x = 0; x < ResU; x++) {
			std::cout << cpuRes[x + y * ResU] << ",";
		}
		std::cout << '\n';
	}

	std::cout << "\n\n Ptex Sampled \n\n";

	for (int y = 0; y < ResV; y++) {
		for (int x = 0; x < ResU; x++) {
			float result[1];
			texture->getPixel(face, x, y, result, 0, 1);
			std::cout << result[0]  << ",";
			cpuRes2[x + y * ResU] = result[0];
		}
		std::cout << '\n';
	}


	std::cout << "\n\n Diff \n\n";
	
	for (int y = 0; y < ResV; y++) {
		for (int x = 0; x < ResU; x++) {
			std::cout << std::abs(cpuRes[x + y * ResU] - cpuRes2[x + y * ResU])  << ",";
		}
		std::cout << '\n';
	}

	int numFaces = texture->numFaces();

	unique_device_ptr<Device::CPU, uint32_t[]> cpuOffsets = make_udevptr_array <Device::CPU, uint32_t, false>(numFaces);
	cudaMemcpy(cpuOffsets.get(), pTexture.getOffsetPointer(), numFaces * sizeof(uint32_t), cudaMemcpyDefault);

	std::cout << "\n\n Offsets \n\n";

	for (int i = 0; i < numFaces; i++) {
		std::cout << cpuOffsets[i] << ",";
		if (i % 5 == 0) {
			std::cout << '\n';
		}
	}


	texture->release();
}