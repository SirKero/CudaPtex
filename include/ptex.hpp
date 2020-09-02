/*
 PTEX SOFTWARE
 Copyright 2014 Disney Enterprises, Inc.  All rights reserved
*/
#pragma once

#include "unique_device_ptr.hpp"
#include "Ptexture.h"

using namespace mufflon;

	
class cudaPtex {
public:
	//loads File into the cuda Array
	void loadFile(const char* filepath, bool premultiply = true);

	float* getDataPointer() {
		return m_data.get();
	}
	int getTotalDataSize() {
		return m_totalDataSize;
	}
	uint32_t* getOffsetPointer() {
		return m_offsets.get();
	}
private:
	bool m_loaded = false;
	unique_device_ptr<Device::CUDA, float[]> m_data;			//1D Array with all the colors
	unique_device_ptr < Device::CUDA, uint32_t[]> m_offsets;		//1D Array with the offsets. Length == numFaces
	unique_device_ptr < Device::CUDA, uint8_t[]> m_ResLog2U;		//1D Array with the res of the U. Length == numFaces
	unique_device_ptr < Device::CUDA, uint8_t[]> m_ResLog2V;		//1D Array with the res of the V. Length == numFaces
	uint8_t m_numChannels = 0;
	uint16_t m_numFaces = 0;
	int m_totalDataSize;

	
	template<typename T>
	void readPtexture(float* desArr, Ptex::PtexTexture (*texture));
};
// Can be deleted
/*
struct PtexFaceData
{
	uint32_t offset;		//offset in the array
	int32_t adjFaces[4];	//Info about adjacent Faces
	uint8_t resLog2U;		//Resolution in log2 of the u for the Face		
	uint8_t resLog2V;		//Resolution in log2 of the v for the Face
	uint8_t adjEdges;		//Info about adjacent Edges, 2bits per Edge (0 to 3)
};
*/


