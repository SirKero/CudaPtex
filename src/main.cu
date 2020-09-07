#include <iostream>
#include <Ptexture.h>
#include <memory>
#include <vector>
#include <stdint.h>
#include "ptex.hpp"
#include <algorithm>
#include <type_traits>

#include "unique_device_ptr.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>


using namespace mufflon;

typedef uint8_t DATA_TYPE;

//Test Kernel. It returns an array with all read texels
__global__ void cudaTest2(DATA_TYPE* res, int faceId, int dimX, int dimY, cudaPtexture tex) {
	unsigned int x = threadIdx.x;
	unsigned int y = blockIdx.x;

	//Had to use specific conversion, if not result is wrong or NaN
	float u = __uint2float_rz(threadIdx.x);
	float v = __uint2float_rz(blockIdx.x);
	u /= __int2float_rz(dimX);
	v /= __int2float_rz(dimY);

	//Sample texel
	DATA_TYPE tmpArr[3];
	PtexelFetch(tmpArr, faceId, u, v, tex);

	//copy result to the returned array
	for (int i = 0; i < tex.numChannels; i++) {
		res[tex.numChannels * (x + dimX * y) + i] = tmpArr[i];
	}
}



int main(){

	int face = 176;	//FaceID, if bigger than numFaces, invalid results are printed


	//std::string filepath = "models/teapot/teapot.ptx";		//<<DT: uint8
	std::string filepath = "models/bunny/bunny.ptx";			//<<DT: uint8
	//std::string filepath = "models/triangle/triangle.ptx";	//<<DT: float

	//Fill the cuda Texture object
	cudaPtex pTexture;
	pTexture.loadFile(filepath.c_str(), cudaPtex::TextureType::dt_none, true);

	//Ptex texture as comparison
	Ptex::PtexTexture* texture;
	Ptex::String ptex_error;
	Ptex::PtexFilter* filter;
	texture = Ptex::PtexTexture::open(filepath.c_str(), ptex_error, true);
	if (texture == nullptr) {
		std::cout << "Error: Could not read ptex texture \n";
	}

	switch (texture->dataType())
	{
	case Ptex::DataType::dt_uint8:
		std::cout << "Ptex: Data Type is uint8 \n";
		break;
	case Ptex::DataType::dt_uint16:
		std::cout << "Ptex: Data Type is uint16 \n";
		break;
	case Ptex::DataType::dt_half:
		std::cout << "Ptex: Data Type is half \n";
		break;
	case Ptex::DataType::dt_float:
		std::cout << "Ptex: Data Type is float \n";
		break;
	}

	//Filter for correct u/v readings on triangles
	Ptex::PtexFilter::FilterType filterType = Ptex::PtexFilter::FilterType::f_point;
	Ptex::PtexFilter::Options opts(filterType);
	filter = Ptex::PtexFilter::getFilter(texture, opts);

	
	//Needed vars
	int texSize = texture->numChannels() * texture->getFaceInfo(face).res.size();
	int ResU = texture->getFaceInfo(face).res.u();
	int ResV = texture->getFaceInfo(face).res.v();
	float ResUF = static_cast<float>(ResU);
	float ResVF = static_cast<float>(ResV);
	int numChannels = pTexture.getNumChannels();
	int testFaceSize = numChannels * ResU * ResV;
	unique_device_ptr<Device::CUDA, DATA_TYPE[]> testRes = make_udevptr_array <Device::CUDA, DATA_TYPE, false>(testFaceSize);
	unique_device_ptr<Device::CPU, DATA_TYPE[]> cpuRes = make_udevptr_array <Device::CPU, DATA_TYPE, false>(testFaceSize);
	unique_device_ptr<Device::CPU, DATA_TYPE[]> cpuRes2 = make_udevptr_array <Device::CPU, DATA_TYPE, false>(testFaceSize);

	cudaTest2 << <ResV, ResU >> > (testRes.get(), face, ResU, ResV, pTexture.getTexture());

	cudaMemcpy(cpuRes.get(), testRes.get(), testFaceSize * sizeof(DATA_TYPE), cudaMemcpyDefault);

	//Prints the CUDA Samples. (0,0) is in the left top corner
	std::cout << "Cuda Sampled: \n\n";

	for (int y = 0; y < ResV; y++) {
		for (int x = 0; x < ResU; x++) {
			for (int i = 0; i < numChannels; i++) {
				int idx = numChannels * (x + y * ResU) + i;
				std::cout << +cpuRes[idx] << ",";
			}
		}
		std::cout << '\n';
	}

	//Prints the Ptex Samples. (0,0) is in the left top corner
	std::cout << "\n\n Ptex Sampled \n\n";

	for (int y = 0; y < ResV; y++) {
		for (int x = 0; x < ResU; x++) {
			float result[3];
			filter->eval(result,0,numChannels,face,static_cast<float>(x)/ResUF, static_cast<float>(y)/ResVF, 0.1f,0.0f,0.0f,0.1f);
			for (int i = 0; i < numChannels; i++) {
				DATA_TYPE resultU8;
				if (std::is_same<DATA_TYPE, float>::value) {
					resultU8 = static_cast<DATA_TYPE>(result[i]);
				}
				else if (std::is_same<DATA_TYPE, uint8_t>::value) {
					resultU8 = static_cast<DATA_TYPE>(result[i] * 255.0f);
				}
				std::cout << +resultU8 << ",";
				int idx = numChannels * (x + y * ResU) + i;
				cpuRes2[idx] = resultU8;
			}
			
		}
		std::cout << '\n';
	}

	//Prints the difference between both. Should be 0 or very small (float precision error)
	std::cout << "\n\n Max Diff \n\n";
	float diff = 0;

	for (int y = 0; y < ResV; y++) {
		for (int x = 0; x < ResU; x++) {
			diff = std::max(diff,static_cast<float>(std::abs(cpuRes[x + y * ResU] - cpuRes2[x + y * ResU])));
		}
	}

	std::cout << diff << '\n';


	texture->release();
}