/*
 PTEX SOFTWARE
 Copyright 2014 Disney Enterprises, Inc.  All rights reserved
*/
#pragma once

#include "unique_device_ptr.hpp"
#include "Ptexture.h"

using namespace mufflon;


//Struct which eases the use of the Cuda _device_ function
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
	void loadFile(const char* filepath, bool premultiply = true);

	//Getter functions
	float* getDataPointer() const{
		return m_data.get();
	}

	
	uint32_t getTotalDataSize() const {
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

	//Creates a cudaPtexture struct for ease of use
	cudaPtexture getTexture() {
		return cudaPtexture{getDataPointer(), getOffsetPointer(), getResLog2U(), getResLog2V(), getNumChannels(), m_isTriangle};
	}

private:
	bool m_loaded = false;
	unique_device_ptr < Device::CUDA, float[]   > m_data;			//1D Array with all the texel data.
	unique_device_ptr < Device::CUDA, uint32_t[]> m_offsets;		//1D Array with the offsets for the data array. Length == numFaces
	unique_device_ptr < Device::CUDA, uint8_t[] > m_ResLog2U;		//1D Array with the res of the U side in Log2. Length == numFaces
	unique_device_ptr < Device::CUDA, uint8_t[] > m_ResLog2V;		//1D Array with the res of the V side in Log2. Length == numFaces
	uint8_t m_numChannels = 0;										//Number of color channels in the texture
	uint32_t m_numFaces = 0;										//Number of faces
	bool m_isTriangle = false;										//The Indexes for triangles are calculated differently due to different save format
	uint32_t m_totalDataSize;

	//Inter function used in load file to read the texture face by face and save
	//the received data in the desArr. There are 4 possible types in which the Ptex texture was saved:
	//uint8_t -> conversion to float [0..1]
	//uint16_t -> conversion to float [0..1]
	//float -> just copy
	//half -> currently not supported
	template<typename T>
	void readPtexture(float* desArr, Ptex::PtexTexture (*texture));
};


//Fetches the texel for the faceIdx and the u and v coordinate [0..1].
//The size of res has to be at least as big as the number of channels
__device__ 
void PtexelFetch(float* res,int faceIdx, float u, float v,int numChannels , const float* texArr, const uint32_t* texOffsetArr, const uint8_t* ResLog2U, const uint8_t* ResLog2V, bool isTriangle);

//Fetches the texel for the faceIdx and the u and v coordinate [0..1].
//The size of res has to be at least as big as the number of channels
__device__
void PtexelFetch(float* res, int faceIdx, float u, float v, cudaPtexture tex);



