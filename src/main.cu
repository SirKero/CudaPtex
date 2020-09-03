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


int main(){

	int face = 0;

	Ptex::PtexTexture* texture;
	Ptex::String ptex_error;
	texture = Ptex::PtexTexture::open("models/teapot/teapot.ptx",ptex_error, true);

	if (texture == nullptr) {
		std::cout << "Error: Could not read ptex texture \n";
	}

	uint16_t test = -1;
	std::cout << test << '\n' << '\n' << '\n';

	int texSize = texture->numChannels() * texture->getFaceInfo(face).res.size();
	int ResU = texture->getFaceInfo(face).res.u();
	int ResV = texture->getFaceInfo(face).res.v();
	//auto tex_ptr_raw = std::make_unique<unsigned char[]>(texSize);
	std::vector<uint8_t> tex_ptr_raw;
	tex_ptr_raw.resize(texSize);
	

	texture->getData(0, tex_ptr_raw.data(), 0);
	int count = 0;
	
	unique_device_ptr<Device::CPU, float[]> tex_ptr = make_udevptr_array < Device::CPU, float, false>(texSize);
	//convert char to float
	for (int i = 0; i < texSize; i++) {
		tex_ptr[i] = static_cast<float>(tex_ptr_raw[i]) / 255.0f;
	}


	std::cout << "\n \n So hier beginnt das convertierte array \n \n";
	count = 0;

	for (int i = 0; i < texSize; i++) {
		std::cout << +tex_ptr[i];
		if (count == 2) {
			count = 0;
			std::cout << "\n";
		}
		else {
			count++;
			std::cout << ",";
		}
	}



	texture->release();

	cudaPtex pTexture;
	pTexture.loadFile("models/teapot/teapot.ptx", true);

	/*
	unique_device_ptr<Device::CPU, float[]> test_ptr = make_udevptr_array < Device::CPU, float, false>(ptexReader.getTotalDataSize());
	cudaMemcpy(test_ptr.get(), ptexReader.getDataPointer(), ptexReader.getTotalDataSize() * sizeof(float), cudaMemcpyDefault);
	count = 0;
	std::cout << "\n\n\nTest Test Test Test Test\n\n\n";
	for (int i = 0; i < texSize; i++) {
		std::cout << test_ptr[i];
		if (count == 2) {
			count = 0;
			std::cout << "\n";
		}
		else {
			count++;
			std::cout << ",";
		}
	}
	*/

	std::cout << "\n\n New Test: \n\n";

	int testFaceSize = pTexture.getNumChannels() * ResU * ResV;
	unique_device_ptr<Device::CUDA, float[]> testRes = make_udevptr_array <Device::CUDA, float, false>(testFaceSize);
	unique_device_ptr<Device::CPU, float[]> cpuRes = make_udevptr_array <Device::CPU, float, false>(testFaceSize);

	cudaTest<<<ResV, ResU >>>(testRes.get(), face, ResU, ResV, pTexture.getNumChannels(), pTexture.getDataPointer(),
		pTexture.getOffsetPointer(), pTexture.getResLog2U(), pTexture.getResLog2V());

	cudaMemcpy(cpuRes.get(), testRes.get(), testFaceSize * sizeof(float), cudaMemcpyDefault);

	for (int y = 0; y < ResV; y++) {
		for (int x = 0; x < ResU; x++) {
			std::cout << cpuRes[x + y * ResU] << ",";
		}
		std::cout << '\n';
	}
	
	texture = Ptex::PtexTexture::open("models/teapot/teapot.ptx", ptex_error, true);

	if (texture == nullptr) {
		std::cout << "Error: Could not read ptex texture \n";
	}

	std::cout << "\n\n Tex2 \n\n";

	for (int y = 0; y < ResV; y++) {
		for (int x = 0; x < ResU; x++) {
			float result[1];
			texture->getPixel(face, x, y, result, 0, 1);
			std::cout << result[0]  << ",";
		}
		std::cout << '\n';
	}

	texture->release();
}