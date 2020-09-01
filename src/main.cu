#include <iostream>
#include <Ptexture.h>
#include <memory>

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

	int texSize = texture->numChannels() * texture->getFaceInfo(face).res.size();
	auto tex_ptr_raw = std::make_unique<unsigned char[]>(texSize);
	
	texture->getData(0, tex_ptr_raw.get(), 0);
	int count = 0;
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

	float testres[3];
	for (int i = 0; i < 3; i++) {
		testres[i] = 0.0f;
	}
	texture->getPixel(0, 0, 0, testres, 0, 1);

	std::cout << "\n" << testres[0] << "," << testres[1] << "," << testres[2] << "\n";
}