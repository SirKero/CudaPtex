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
	PtexelFetch(tmpArr, faceId, u, v, numChannels, texArr, texOffsetArr,ResLog2U, ResLog2V, false);
	
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

	int face = 10000;

	//std::string filepath = "models/teapot/teapot.ptx";
	std::string filepath = "models/bunny/bunny.ptx";
	//std::string filepath = "models/triangle/triangle.ptx";

	Ptex::PtexTexture* texture;
	Ptex::String ptex_error;
	Ptex::PtexFilter* filter;
	cudaPtex pTexture;
	pTexture.loadFile(filepath.c_str(), true);
	texture = Ptex::PtexTexture::open(filepath.c_str(), ptex_error, true);

	if (texture == nullptr) {
		std::cout << "Error: Could not read ptex texture \n";
	}

	Ptex::PtexFilter::FilterType filterType = Ptex::PtexFilter::FilterType::f_point;
	Ptex::PtexFilter::Options opts(filterType);
	filter = Ptex::PtexFilter::getFilter(texture, opts);

	

	int texSize = texture->numChannels() * texture->getFaceInfo(face).res.size();
	int ResU = texture->getFaceInfo(face).res.u();
	int ResV = texture->getFaceInfo(face).res.v();
	float ResUF = static_cast<float>(ResU);
	float ResVF = static_cast<float>(ResV);
	int numChannels = pTexture.getNumChannels();
	int testFaceSize = numChannels * ResU * ResV;
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
			for (int i = 0; i < numChannels; i++) {
				int idx = numChannels * (x + y * ResU) + i;
				std::cout << cpuRes[idx] << ",";
			}
			
		}
		std::cout << '\n';
	}

	std::cout << "\n\n Ptex Sampled \n\n";

	for (int y = 0; y < ResV; y++) {
		for (int x = 0; x < ResU; x++) {
			float result[3];
			filter->eval(result,0,numChannels,face,static_cast<float>(x)/ResUF, static_cast<float>(y)/ResVF, 0.1f,0.0f,0.0f,0.1f);
			for (int i = 0; i < numChannels; i++) {
				std::cout << result[i] << ",";
				int idx = numChannels * (x + y * ResU) + i;
				cpuRes2[idx] = result[i];
			}
			
		}
		std::cout << '\n';
	}


	std::cout << "\n\n Max Diff \n\n";
	float diff = 0.0;

	for (int y = 0; y < ResV; y++) {
		for (int x = 0; x < ResU; x++) {
			diff = std::max(diff,std::abs(cpuRes[x + y * ResU] - cpuRes2[x + y * ResU]));
		}
	}

	std::cout << diff << '\n';


	texture->release();
}