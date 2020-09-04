/*
 PTEX SOFTWARE
 Copyright 2014 Disney Enterprises, Inc.  All rights reserved
*/
#pragma once

#include "unique_device_ptr.hpp"
#include "Ptexture.h"

using namespace mufflon;
	
struct cudaPtexture {
	float* data;
	uint32_t* offset;
	uint8_t* ResLog2U;
	uint8_t* ResLog2V;
	uint8_t numChannels;
	bool isTriangle;
};

class cudaPtex {
public:
	//Loading the Ptex File into the class intern Cuda Array(m_data)
	//it also fills the offset and u,v resolutions (in log2)
	__host__
	void loadFile(const char* filepath, bool premultiply = true);

	
	float* getDataPointer() const{
		return m_data.get();
	}

	
	int getTotalDataSize() const {
		return m_totalDataSize;
	}

	
	uint32_t* getOffsetPointer() const{
		return m_offsets.get();
	}

	
	uint8_t* getResLog2U() const{
		return m_ResLog2U.get();
	}

	
	uint8_t* getResLog2V() const {
		return m_ResLog2V.get();
	}

	
	uint8_t getNumChannels() const {
		return m_numChannels;
	}

	cudaPtexture getTexture() {
		return cudaPtexture{getDataPointer(), getOffsetPointer(), getResLog2U(), getResLog2V(), getNumChannels(), m_isTriangle};
	}

private:
	bool m_loaded = false;
	unique_device_ptr<Device::CUDA, float[]> m_data;			//1D Array with all the colors
	unique_device_ptr < Device::CUDA, uint32_t[]> m_offsets;		//1D Array with the offsets. Length == numFaces
	unique_device_ptr < Device::CUDA, uint8_t[]> m_ResLog2U;		//1D Array with the res of the U. Length == numFaces
	unique_device_ptr < Device::CUDA, uint8_t[]> m_ResLog2V;		//1D Array with the res of the V. Length == numFaces
	uint8_t m_numChannels = 0;
	uint32_t m_numFaces = 0;
	bool m_isTriangle = false;
	int m_totalDataSize;

	
	template<typename T>
	void readPtexture(float* desArr, Ptex::PtexTexture (*texture));
};



__device__ 
void PtexelFetch(float* res,int faceIdx, float u, float v,int numChannels , const float* texArr, const uint32_t* texOffsetArr, const uint8_t* ResLog2U, const uint8_t* ResLog2V, bool isTriangle);

__device__
void PtexelFetch(float* res, int faceIdx, float u, float v, cudaPtexture tex);


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


