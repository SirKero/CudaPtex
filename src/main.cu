#include <iostream>
#include <Ptexture.h>
#include <memory>
#include <vector>
#include "ptex.hpp"

#include "unique_device_ptr.hpp"

using namespace mufflon;

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
	//auto tex_ptr_raw = std::make_unique<unsigned char[]>(texSize);
	std::vector<uint8_t> tex_ptr_raw;
	tex_ptr_raw.resize(texSize);
	

	texture->getData(0, tex_ptr_raw.data(), 0);
	int count = 0;
	/*
	for (int i = 0; i < texSize; i++) {
		std::cout << +tex_ptr_raw[i];
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

	cudaPtex ptexReader;
	ptexReader.loadFile("models/teapot/teapot.ptx", true);

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
}