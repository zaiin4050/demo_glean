#pragma once

#ifdef _WIN32
#ifdef __SOYNET__
#define SOYNET_DLL __declspec(dllexport)
#else
#define SOYNET_DLL __declspec(dllimport)
#endif
#else
#define SOYNET_DLL 
#endif

#ifdef __cplusplus
extern "C" {
#endif
	SOYNET_DLL void* initSoyNet(const char* cfg_file_name, const char* extend_param); // SoyNet handle¿ª return«—¥Ÿ.
	SOYNET_DLL void feedData(const void * SoyNetHandle, const void* data);
	SOYNET_DLL void feedDataDevice(const void * SoyNetHandle, const void* data);
	SOYNET_DLL void feedDataAux(const void * SoyNetHandle, const void* aux);
	SOYNET_DLL void inference(const void * SoyNetHandle);
	SOYNET_DLL void getOutput(const void * SoyNetHandle, void * output);
	SOYNET_DLL void inferSoyNet(const void * soynetHandle, void* input, int input_byte, void* output, int output_byte, char* infer_id, int id_byte, char* extend_param);
	SOYNET_DLL void freeSoyNet(const void* SoyNet);

	//kesco
	SOYNET_DLL void feedData_jpg(const void * SoyNetHandle, char* fn);
	SOYNET_DLL void feedData_bin(const void * SoyNetHandle, char* fn);
	SOYNET_DLL void getOutput_detectron2(const void * SoyNetHandle, char* fn0, char* fn1, char* fn2);
	SOYNET_DLL void getOutput_efficientnet(const void * SoyNetHandle, char* fn);

	//SOYNET_DLL void inference(const void * SoyNetHandle, void * output, const void* data);
#ifdef __cplusplus
}
#endif


#ifdef __SOYNET__

#include "jni.h"

#ifdef __cplusplus
extern "C" {
#endif

	JNIEXPORT jlong JNICALL Java_SoyNet_initSoyNet(JNIEnv *, jclass, jstring, jstring);
	JNIEXPORT void JNICALL Java_SoyNet_feedData(JNIEnv *, jclass, jlong, jbyteArray);
	JNIEXPORT void JNICALL Java_SoyNet_inference(JNIEnv *, jclass, jlong);
	JNIEXPORT void JNICALL Java_SoyNet_getOutput(JNIEnv *, jclass, jlong, jfloatArray);
	JNIEXPORT void JNICALL Java_SoyNet_freeSoyNet(JNIEnv *, jclass, jlong);

#ifdef __cplusplus
}
#endif
#endif