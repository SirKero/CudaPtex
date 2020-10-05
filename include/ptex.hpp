/*
 PTEX SOFTWARE
 Copyright 2014 Disney Enterprises, Inc.  All rights reserved
*/
#pragma once

#include "unique_device_ptr.hpp"
#include "Ptexture.h"

using namespace mufflon;


//Struct which eases the use of the Cuda _device_ function
struct cudaPtexture;

class cudaPtex {
public:
	//Possible Texture Data types
	enum class TextureType {
		dt_uint8,
		dt_uint16,
		dt_half,
		dt_float,
		dt_none
	};

	enum class FilterMode {
		POINT,
		BILINEAR
	};

	//(Filter only) Enum for edge Id. 
	enum class EdgeId {
		e_bottom,
		e_right,
		e_top,
		e_left
	};


	//Constructors
	cudaPtex() {}
	cudaPtex(const char* filepath, TextureType textureDataType = TextureType::dt_none, bool premultiply = true) {
		loadFile(filepath, textureDataType ,premultiply);
	}


	//Loading the Ptex File into the class intern Cuda Array
	//The Texure Type has to be the same as the ptex File or none (if dt_none is given, then the ptex Texture Type is used)
	void loadFile(const char* filepath, TextureType textureDataType = TextureType::dt_none, bool allowFilter = false , bool premultiply = true);

	void setFilter(FilterMode filter) {
		m_filterMode = filter;
	}

	//Getter functions
	void* getDataPointer() const{
		return m_dataArr.get();
	}
	uint32_t getTotalDataSize() const {
		return m_totalDataSize;
	}
	uint32_t* getOffsetPointer() const{
		return m_offsetPtr;
	}
	uint8_t* getResLog2U() const{
		return m_resLog2UPtr;
	}
	uint8_t* getResLog2V() const {
		return m_resLog2VPtr;
	}
	uint8_t getNumChannels() const {
		return m_numChannels;
	}

	//Creates a cudaPtexture struct for ease of use
	cudaPtexture getTexture();

private:
	bool m_loaded = false;
	unique_device_ptr < Device::CUDA, char[]> m_dataArr;		//1D Array with all the texel data and extra Data. Sequence is Data->Offset->ResLog2U->ResLog2V
	uint32_t* m_offsetPtr = nullptr;							//Pointer to the Start of the offsets in m_dataArr
	uint8_t* m_resLog2UPtr = nullptr;							//Pointer to the Start of resLog2U in m_dataArr
	uint8_t* m_resLog2VPtr = nullptr;							//Pointer to the Start of resLog2V in m_dataArr
	int32_t* m_adjFaces = nullptr;									//(Filter only) adjacent Faces
	uint8_t* m_adjEdges = nullptr;								//(Filter only) adjacent Edge

	TextureType m_DataType = TextureType::dt_none;				//Texture Type.
	FilterMode m_filterMode = FilterMode::POINT;				//Filter Mode
	uint8_t m_numChannels = 0;									//Number of color channels in the texture
	uint32_t m_numFaces = 0;									//Number of faces
	bool m_isTriangle = false;									//The Indexes for triangles are calculated differently due to different save format
	uint32_t m_totalDataSize = 0;

	//Inter function used in load file to read the texture face by face and copy the 
	//Data to the GPU(m_dataArr). There are 4 possible types in which the Ptex texture was saved:
	//uint8_t ; uint16_t ; float ; half
	template<typename T>
	void readPtexture(Ptex::PtexTexture (*texture), int totalDataSize, int extraBufferSize);
};

//Definition of the struct defined above
struct cudaPtexture {
	void* data;
	uint32_t* offset;
	uint8_t* ResLog2U;
	uint8_t* ResLog2V;
	int32_t* adjFaces;
	uint8_t* adjEdges;
	uint8_t numChannels;
	cudaPtex::TextureType texType;
	cudaPtex::FilterMode filterMode;
	bool isTriangle;
};

//Fetches the texel for the faceIdx and the u and v coordinate [0..1].
//The size of res has to be at least as big as the number of channels
__device__
void PtexelFetch(void* res, int faceIdx, float u, float v, cudaPtexture tex, float filterWidthU = 0.1f, float filterWidthV = 0.1f);


//Fetches the texel for the faceIdx and the u and v coordinate [0..1].
//The size of res has to be at least as big as the number of channels
__device__ 
void PtexelFetch(void* res, int faceIdx, float u, float v, int numChannels, void* texArr,
	const uint32_t* texOffsetArr, const uint8_t* ResLog2U, const uint8_t* ResLog2V, const int32_t* adjFaces,
	const uint8_t* adjEdges, cudaPtex::TextureType texType, cudaPtex::FilterMode filterMode, float filterWidthU, float filterWidthV, bool isTriangle);




