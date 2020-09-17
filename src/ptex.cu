/*
 PTEX SOFTWARE
 Copyright 2014 Disney Enterprises, Inc.  All rights reserved
 */

#include "ptex.hpp"
#include <iostream>
#include <math.h>
#include <memory>
#include <vector>
#include <type_traits>
#include <stdint.h>
#include <stdexcept>
//CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace cudaPtexHelper {
	// Calculates the power for Base 2.
__device__
unsigned int pow2(const uint8_t pow) {
	unsigned int result = 1 << pow;
	return result;
}

__device__ float mixf(float x, float y, float a) {
	return x * (1.0f - a) + y * a;
}

__device__ uint8_t mixUI(unsigned int x, unsigned int y, float a) {
	x = __float2uint_rz(__uint2float_rz(x) * (1.0f - a));
	y = __float2uint_rz(__uint2float_rz(y) * a);
	return x + y;
}


__device__
int getTexelIndex( float u, float v,const uint8_t &numChannels, const unsigned int &offset, const unsigned int& ResU, const unsigned int& ResV, const bool& isTriangle) {
	int index;
	u = fminf(fmaxf(u, 0.0f), 1.0f);	//clamp to 0..1
	v = fminf(fmaxf(v, 0.0f), 1.0f);	//clamp to 0..1
	if (!isTriangle) {
		index = offset + ResU * numChannels * __float2uint_rz(u + v * __uint2float_rz(ResV));
	}
	//For triangles: texture fetch after http://ptex.us/tritex.html
	else {
		float resf = __uint2float_rz(ResU);
		float ut = u * resf;
		float vt = v * resf;
		float uIdx = floorf(ut);
		float vIdx = floorf(vt);
		int tmpIndex;
		if ((ut - uIdx) + (vt - vIdx) <= 1.0f) {
			tmpIndex = __float2int_rz(uIdx + vIdx * resf);
		}
		else {
			tmpIndex = __float2int_rz((resf * resf - 1.0f) - (vIdx + uIdx * resf));
		}

		int iU = tmpIndex % ResU;
		int iV = tmpIndex / ResV;

		index = offset + numChannels * (iU + iV * ResU);
	}

	return index;
}

__device__
cudaPtex::EdgeId getAdjEdge(const int edgeID, const uint8_t adjEdge) {
	return cudaPtex::EdgeId((adjEdge >> (2 * edgeID)) & 3);
}

__device__
void PtexelFetchPoint(void* res,const int& faceIdx,const float& u,const float& v, const uint8_t& numChannels, void* texArr,
	const uint32_t* texOffsetArr, const uint8_t* ResLog2U, const uint8_t* ResLog2V, cudaPtex::TextureType texType, bool isTriangle) {
	//calc Res U and Res V from the log2 variants from the array
	unsigned int ResU = pow2(ResLog2U[faceIdx]);
	unsigned int ResV = pow2(ResLog2V[faceIdx]);

	unsigned int offset = texOffsetArr[faceIdx];
	unsigned int index = getTexelIndex(u, v, numChannels, offset, ResU, ResV, isTriangle);

	//Sample Data depending on type
	switch (texType)
	{
	case cudaPtex::TextureType::dt_uint8:
		for (int i = 0; i < numChannels; i++) {
			reinterpret_cast<uint8_t*>(res)[i] = reinterpret_cast<uint8_t*>(texArr)[index + i];
		}
		break;
	case cudaPtex::TextureType::dt_uint16:
		for (int i = 0; i < numChannels; i++) {
			reinterpret_cast<uint16_t*>(res)[i] = reinterpret_cast<uint16_t*>(texArr)[index + i];
		}
		break;
	case cudaPtex::TextureType::dt_half:
		for (int i = 0; i < numChannels; i++) {
			reinterpret_cast<half*>(res)[i] = reinterpret_cast<half*>(texArr)[index + i];
		}
		break;
	case cudaPtex::TextureType::dt_float:
		for (int i = 0; i < numChannels; i++) {
			reinterpret_cast<float*>(res)[i] = reinterpret_cast<float*>(texArr)[index + i];
		}
		break;
	}
}

__device__
void PtexelFetchBilinear(void* res, const int& faceIdx, const float& u, const float& v, const int& numChannels, void* texArr,
	const uint32_t* texOffsetArr, const uint8_t* ResLog2U, const uint8_t* ResLog2V, const int32_t* adjFaces, const uint8_t* adjEdges, 
	cudaPtex::TextureType texType, float filterWidthU, float filterWidthV, const bool& isTriangle) {
	
	//calc Res U and Res V from the log2 variants from the array
	unsigned int ResU = pow2(ResLog2U[faceIdx]);
	unsigned int ResV = pow2(ResLog2V[faceIdx]);

	unsigned int offset = texOffsetArr[faceIdx];
	
	//Clamp Filter between -1 and 1
	filterWidthU = fminf(fmaxf(filterWidthU, -1.0f), 1.0f);	
	filterWidthV = fminf(fmaxf(filterWidthV, -1.0f), 1.0f);

	//Landing point is always bottom left
	float resF[2] = { __uint2float_rz(ResU), __uint2float_rz(ResV) };
	float texelSize[2]{ 1.0f / resF[0], 1.0f / resF[1] };
	float fract[2] = { u * resF[0] - floorf(u * resF[0]), v * resF[1] - floorf(v * resF[1]) };
	float kernelU[4], kernelV[4];
	kernelU[0] = { u + (0.5f - fract[0]) * texelSize[0] }; kernelV[0] = {v + (0.5f - fract[1]) * texelSize[1]}; //move uv to pixel center
	kernelU[1] = { kernelU[0] + filterWidthU}; kernelV[1] = {kernelV[0] };		//right from point
	kernelU[2] = { kernelU[0] + filterWidthU }; kernelV[2] = { kernelV[0] + filterWidthV };	//up right from point
	kernelU[3] = { kernelU[0]}; kernelV[3] = { kernelV[0] + filterWidthV };	//up from point

	bool inOtherTex[3];
	//Check if Kernel is overstepping texture
	for (int i = 1; i < 4; i++) {
		inOtherTex[i - 1] = kernelU[i] > 1.0f || kernelU[i] < 0.0f || kernelV[i] > 1.0f || kernelV[i] < 0.0f ? true : false;
	}

	int faceIndices[4]; faceIndices[0] = faceIdx;
	for (int i = 1; i < 4; i++) {
		if (inOtherTex[i - 1]) {
			//Get side
			cudaPtex::EdgeId eid;
			if (kernelU[i] > 1.0f) { 
				eid = cudaPtex::EdgeId::e_right;
			}
			else if(kernelV[i] > 1.0f) {
				eid = cudaPtex::EdgeId::e_top;
			}
			else if(kernelV[i] < 0.0f) {
				eid = cudaPtex::EdgeId::e_bottom;
			}
			else {
				eid = cudaPtex::EdgeId::e_left;
			}

			faceIndices[i] = adjFaces[faceIdx * 4 + static_cast<int>(eid)];
			cudaPtex::EdgeId eiDir = getAdjEdge(static_cast<int>(eid), adjEdges[faceIdx]);

			float kernelNew = 0.0f;
			//Change UV depending on dir
			switch (eid)
			{
			case cudaPtex::EdgeId::e_right:
				kernelNew = kernelU[i] - 1.0f;
				switch (eiDir)	
				{
				case cudaPtex::EdgeId::e_bottom:
					kernelU[i] = 1.0f - kernelV[i];
					kernelV[i] = kernelNew;
					break;
				case cudaPtex::EdgeId::e_right:
					kernelV[i] = 1.0f - kernelV[i];
					kernelU[i] = 1.0f - kernelNew;
					break;
				case cudaPtex::EdgeId::e_top:
					kernelU[i] = kernelV[i];
					kernelV[i] = 1.0f - kernelNew;
					break;
				case cudaPtex::EdgeId::e_left:
					kernelU[i] = kernelNew;
					break;
				}
				break;
			case cudaPtex::EdgeId::e_top:
				kernelNew = kernelV[i] - 1.0f;
				switch (eiDir)
				{
				case cudaPtex::EdgeId::e_bottom:
					kernelV[i] = kernelNew;
					break;
				case cudaPtex::EdgeId::e_right:
					kernelV[i] = kernelU[i];
					kernelU[i] = 1.0f - kernelNew;
					break;
				case cudaPtex::EdgeId::e_top:
					kernelU[i] = 1.0f - kernelU[i];
					kernelV[i] = 1.0f - kernelNew;
					break;
				case cudaPtex::EdgeId::e_left:
					kernelV[i] = 1.0f - kernelU[i];
					kernelU[i] = kernelNew;
					break;
				}
				break;
			case cudaPtex::EdgeId::e_bottom:
				kernelNew = kernelV[i] + 1.0f;
				switch (eiDir)
				{
				case cudaPtex::EdgeId::e_bottom:
					kernelU[i] = 1.0f - kernelU[i];
					kernelV[i] = 1.0f - kernelNew;
					break;
				case cudaPtex::EdgeId::e_right:
					kernelV[i] = 1.0f - kernelU[i];
					kernelU[i] = kernelNew;
					break;
				case cudaPtex::EdgeId::e_top:
					kernelV[i] = kernelNew;
					break;
				case cudaPtex::EdgeId::e_left:
					kernelV[i] = kernelU[i];
					kernelU[i] = 1.0f - kernelNew;
					break;
				}
				break;
			case cudaPtex::EdgeId::e_left:
				kernelNew = kernelU[i] + 1.0f;
				switch (eiDir)
				{
				case cudaPtex::EdgeId::e_bottom:
					kernelU[i] = kernelV[i];
					kernelV[i] = 1.0f - kernelNew;
					break;
				case cudaPtex::EdgeId::e_right:
					kernelU[i] = kernelNew;
					break;
				case cudaPtex::EdgeId::e_top:
					kernelU[i] = 1.0f - kernelV[i];
					kernelV[i] = kernelNew;
					break;
				case cudaPtex::EdgeId::e_left:
					kernelU[i] = 1.0f - kernelNew;
					kernelV[i] = 1.0f - kernelV[i];
					break;
				}
				break;
			}

		}
		else {
			faceIndices[i] = faceIdx;
		}
	}

	//get index of all 4 points
	int index[4];
	index[0] = getTexelIndex(kernelU[0], kernelV[0], numChannels, offset, ResU, ResV, isTriangle);
	for (int i = 1; i < 4; i++) {
		if (faceIndices[i] != -1) {
			index[i] = getTexelIndex(kernelU[i], kernelV[i], numChannels, texOffsetArr[faceIndices[i]], ResU, ResV, isTriangle);
		}
		else {
			index[i] = index[0];
		}
	}

	//sample and interpolate the points depending on type
	switch (texType)
	{
	case cudaPtex::TextureType::dt_uint8: {
		for (int i = 0; i < numChannels; i++) {
			uint8_t mix1 = mixUI(reinterpret_cast<uint8_t*>(texArr)[index[0] + i], reinterpret_cast<uint8_t*>(texArr)[index[1] + i], fract[0]);
			uint8_t mix2 = mixUI(reinterpret_cast<uint8_t*>(texArr)[index[2] + i], reinterpret_cast<uint8_t*>(texArr)[index[3] + i], fract[0]);
			reinterpret_cast<uint8_t*>(res)[i] = mixUI(mix1, mix2, fract[1]);
		}
		break;
	}
		
	case cudaPtex::TextureType::dt_uint16: {
		for (int i = 0; i < numChannels; i++) {
			uint16_t mix1 = mixUI(reinterpret_cast<uint16_t*>(texArr)[index[0] + i], reinterpret_cast<uint16_t*>(texArr)[index[1] + i], fract[0]);
			uint16_t mix2 = mixUI(reinterpret_cast<uint16_t*>(texArr)[index[2] + i], reinterpret_cast<uint16_t*>(texArr)[index[3] + i], fract[0]);
			reinterpret_cast<uint16_t*>(res)[i] = mixUI(mix1, mix2, fract[1]);
		}
		break;
	}
	case cudaPtex::TextureType::dt_half: {
		for (int i = 0; i < numChannels; i++) {
			half mix1 = mixf(reinterpret_cast<half*>(texArr)[index[0] + i], reinterpret_cast<half*>(texArr)[index[1] + i], fract[0]);
			half mix2 = mixf(reinterpret_cast<half*>(texArr)[index[2] + i], reinterpret_cast<half*>(texArr)[index[3] + i], fract[0]);
			reinterpret_cast<half*>(res)[i] = mixf(mix1, mix2, fract[1]);
		}
		break;
	}
	case cudaPtex::TextureType::dt_float: {
		for (int i = 0; i < numChannels; i++) {
			float mix1 = mixf(reinterpret_cast<float*>(texArr)[index[0] + i], reinterpret_cast<float*>(texArr)[index[1] + i], fract[0]);
			float mix2 = mixf(reinterpret_cast<float*>(texArr)[index[2] + i], reinterpret_cast<float*>(texArr)[index[3] + i], fract[0]);
			reinterpret_cast<float*>(res)[i] = mixf(mix1, mix2, fract[1]);
		}
		break;
	}
	}//end switch

}

} // namespace cudaPtexHelper

using namespace cudaPtexHelper;
//Cuda functions
__device__ 
void PtexelFetch(void* res, int faceIdx, float u, float v, int numChannels, void* texArr,
	const uint32_t* texOffsetArr, const uint8_t* ResLog2U, const uint8_t* ResLog2V, const int32_t* adjFaces,
	const uint8_t* adjEdges, cudaPtex::TextureType texType, cudaPtex::FilterMode filterMode, float filterWidthU, float filterWidthV, bool isTriangle) {
	
	if (filterMode == cudaPtex::FilterMode::BILINEAR && !isTriangle) {
		PtexelFetchBilinear(res, faceIdx, u, v, numChannels, texArr, texOffsetArr, ResLog2U, ResLog2V, adjFaces, adjEdges , texType, 0.1f, 0.1f,isTriangle);
	}
	else {
		PtexelFetchPoint(res, faceIdx, u, v, numChannels, texArr, texOffsetArr, ResLog2U, ResLog2V, texType, isTriangle);
	}

}

__device__
void PtexelFetch(void* res, int faceIdx, float u, float v, cudaPtexture tex, float filterWidthU, float filterWidthV) {
	PtexelFetch(res, faceIdx, u, v,tex.numChannels, tex.data, tex.offset, tex.ResLog2U, tex.ResLog2V,tex.adjFaces,
				tex.adjEdges,tex.texType,tex.filterMode, filterWidthU, filterWidthV , tex.isTriangle);
}

void cudaPtex::loadFile(const char* filepath, TextureType textureDataType, bool allowFilter ,bool premultiply) {
	//release old texture Data if there is already a texture loaded in
	if (m_loaded) {
		m_dataArr.release();
		m_offsetPtr = nullptr;
		m_resLog2UPtr = nullptr;
		m_resLog2VPtr = nullptr;
		m_loaded = false;
	}

	//Load texture from file
	Ptex::PtexTexture* texture;
	Ptex::String ptexErr;
	texture = Ptex::PtexTexture::open(filepath, ptexErr, premultiply);
	//In experience this only triggers if the file is not a ptex file
	if (texture == nullptr) {
		std::cerr << "Ptex Error: " << ptexErr.c_str() << '\n';
		return;
	}

	//Get info about the texture
	m_numFaces = texture->numFaces();
	m_numChannels = texture->numChannels();
	
	if (texture->meshType() == Ptex::MeshType::mt_triangle) {
		m_isTriangle = true;
	}
	
	//Create CPU side Buffers
	uint32_t totalDataSize = 0;
	auto offsetBuf = std::make_unique<uint32_t[]>(m_numFaces);
	auto resUBuf = std::make_unique<uint8_t[]>(m_numFaces);
	auto resVBuf = std::make_unique<uint8_t[]>(m_numFaces);
	auto filterAdjFaces = std::make_unique<int32_t[]>(m_numFaces*4);	//(Filter only)
	auto filterAdjEdges = std::make_unique<uint8_t[]>(m_numFaces);		//(Filter only)

	//Fill helpers from the Ptex FaceInfo
	offsetBuf[0] = 0;	//first one has no offset
	for (unsigned int i = 0; i < m_numFaces; i++) {
		Ptex::FaceInfo faceInfo = texture->getFaceInfo(i);
		resUBuf[i] = faceInfo.res.ulog2;
		resVBuf[i] = faceInfo.res.vlog2;
		totalDataSize += faceInfo.res.size() * m_numChannels;
		if (i < m_numFaces - 1) 
			offsetBuf[i + 1] = totalDataSize;	//offset for the data

		//If filter is enabled also copy Filter stuff
		if (allowFilter) {
			for(int j=0; j<4;j++)
				filterAdjFaces[i * 4 + j] = faceInfo.adjfaces[j];
			filterAdjEdges[i] = faceInfo.adjedges;
		}

	}

	//Size of the extraBuffers
	uint32_t extraBufferSize = m_numFaces * sizeof(uint32_t) + 2 * m_numFaces * sizeof(uint8_t);

	//If Filter is allowed add extra needed space on top
	if (allowFilter)
		extraBufferSize += 4 * m_numFaces * sizeof(int32_t) + m_numFaces * sizeof(uint8_t);

	//DataSize in Bytes (is depending on the Texture Type)
	uint32_t totalDataByteSize = 0;	

	//Check if Data Types matches, throw runtimeError if not;
	//TODO: Add automatic type conversion
	Ptex::DataType ptexDataType = texture->dataType();
	if (textureDataType != TextureType::dt_none) {
		switch (ptexDataType)
		{
		case Ptex::DataType::dt_uint8:
			if (textureDataType != TextureType::dt_uint8)
				throw std::runtime_error("cudaPtex: Texture Type does not match! Ptex Texture data type is uint8");
			break;
		case Ptex::DataType::dt_uint16:
			if (textureDataType != TextureType::dt_uint16)
				throw std::runtime_error("cudaPtex: Texture Type does not match! Ptex Texture data type is uint16");
			break;
		case Ptex::DataType::dt_half:
			if (textureDataType != TextureType::dt_half)
				throw std::runtime_error("cudaPtex: Texture Type does not match! Ptex Texture data type is half");
			break;
		case Ptex::DataType::dt_float:
			if (textureDataType != TextureType::dt_float)
				throw std::runtime_error("cudaPtex: Texture Type does not match! Ptex Texture data type is float");
			break;
		}
	}

	//Check in which Data type the ptex file is in an read accordingly
	switch (ptexDataType) {
	case Ptex::DataType::dt_uint8: 
		readPtexture<uint8_t>(texture, totalDataSize, extraBufferSize);
		m_DataType = TextureType::dt_uint8;
		totalDataByteSize += totalDataSize * sizeof(uint8_t);
		break;

	case Ptex::DataType::dt_uint16: 
		readPtexture<uint16_t>(texture, totalDataSize, extraBufferSize);
		m_DataType = TextureType::dt_uint16;
		totalDataByteSize += totalDataSize * sizeof(uint16_t);
		break;

	case Ptex::DataType::dt_float: 
		readPtexture<float>(texture, totalDataSize, extraBufferSize);
		m_DataType = TextureType::dt_float;
		totalDataByteSize += totalDataSize * sizeof(float);
		break;

	case Ptex::DataType::dt_half: 
		readPtexture<half>(texture, totalDataSize, extraBufferSize);
		m_DataType = TextureType::dt_half;
		totalDataByteSize += totalDataSize * sizeof(half);
		break;
	}

	//Copy Extra Data to the GPU
	m_offsetPtr = reinterpret_cast<uint32_t*>(& m_dataArr[totalDataByteSize]);
	cudaMemcpy(m_offsetPtr, offsetBuf.get(), m_numFaces * sizeof(uint32_t), cudaMemcpyDefault);
	m_resLog2UPtr = reinterpret_cast<uint8_t*>(&m_dataArr[totalDataByteSize  + m_numFaces * sizeof(uint32_t)]);
	cudaMemcpy(m_resLog2UPtr, resUBuf.get(), m_numFaces * sizeof(uint8_t), cudaMemcpyDefault); 
	m_resLog2VPtr = reinterpret_cast<uint8_t*>(&m_dataArr[totalDataByteSize + m_numFaces * sizeof(uint32_t) + m_numFaces * sizeof(uint8_t)]);
	cudaMemcpy(m_resLog2VPtr, resVBuf.get(), m_numFaces * sizeof(uint8_t), cudaMemcpyDefault);
	if (allowFilter) {
		m_adjFaces = reinterpret_cast<int32_t*>(&m_dataArr[totalDataByteSize + m_numFaces * sizeof(uint32_t) + 2 * m_numFaces * sizeof(uint8_t)]);
		cudaMemcpy(m_adjFaces, filterAdjFaces.get(), 4 * m_numFaces * sizeof(uint32_t), cudaMemcpyDefault);
		m_adjEdges = reinterpret_cast<uint8_t*>(&m_dataArr[totalDataByteSize + 5 * m_numFaces * sizeof(uint32_t) + 2 * m_numFaces * sizeof(uint8_t)]);
		cudaMemcpy(m_adjEdges, filterAdjEdges.get(), m_numFaces * sizeof(uint8_t), cudaMemcpyDefault);
	}

	//Set some member vars
	m_totalDataSize = totalDataSize;
	m_loaded = true;

	//release Ptex texture, it is not needed anymore
	texture->release(); 
}

cudaPtexture cudaPtex::getTexture() {
	if (!m_loaded)
		throw std::runtime_error("cudaPtex: No Texture was loaded");
	return cudaPtexture{ getDataPointer(), m_offsetPtr, m_resLog2UPtr, m_resLog2VPtr,m_adjFaces, m_adjEdges , m_numChannels,m_DataType, m_filterMode, m_isTriangle };
}


template <typename T>
void cudaPtex::readPtexture<T>(Ptex::PtexTexture (*texture),int totalDataSize, int extraBufferSize) {
	static_assert(std::is_same<T,uint8_t>::value || std::is_same<T, uint16_t>::value || std::is_same<T, float>::value || std::is_same<T, half>::value, "Type is not supportet by Ptex");
	
	auto dataBuf = std::make_unique<T[]>(totalDataSize);
	uint64_t offset = 0; //Data offset for dataBuf
	for (int i = 0; i < texture->numFaces(); i++) {
		std::vector<T> faceDataBuffer;
		int texSize = texture->numChannels() * texture->getFaceInfo(i).res.size();
		faceDataBuffer.resize(texSize);
		texture->getData(i, faceDataBuffer.data(), 0);

		//copyData
		for (int j = 0; j < texSize; j++)
			dataBuf[j + offset] = faceDataBuffer[j];
		
		//add to offset
		offset += texSize;
	}

	//copy Data to GPU
	m_dataArr = make_udevptr_array<Device::CUDA, char, false>(totalDataSize * sizeof(T) + extraBufferSize);
	cudaMemcpy(m_dataArr.get(), dataBuf.get(), totalDataSize * sizeof(T), cudaMemcpyDefault);

}