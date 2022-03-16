#include "SoyNet.h"
#include <string>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <random>

using namespace cv;
using namespace std;
using namespace chrono;

void Clamp(std::vector<float>& input_image, float min_value, float max_value)
{
	std::transform(input_image.begin(), input_image.end(), input_image.begin(), [min_value](float v) {return (v < min_value ? min_value : v); });
	std::transform(input_image.begin(), input_image.end(), input_image.begin(), [max_value](float v) {return (v > max_value ? max_value : v); });
}

// HTX
int glean_img(string model_name, string img_path, int engine_serialize, int display_flag, int save_flag) // generator
{

	// Load Image
	Mat img = cv::imread(img_path);
	Mat resize_img;

	int input_height = 32;
	int input_width = 32;

	int batch_size = 1;

	int model_height = 32;
	int model_width = 32;

	// Input Data memory alloc
	vector<uint8_t> input(batch_size*input_height*input_width * 3);
	memcpy(input.data(), img.data, batch_size*input_height*input_width * 3);

	vector<float>output(batch_size * (model_height * 8) * (model_width * 8) * 3 );
  Mat frame((model_height * 8), (model_width * 8), CV_32FC3, output.data());
  
	// SoyNet v4 extend_param 
	char cfg_file[] = "../mgmt/configs/glean.cfg";
	char license_file[] = "../mgmt/configs/license_trial.key";
	char extend_param[1000];
	char engine_file[] = "../mgmt/engines/glean.bin";
	char weight_file[] = "../mgmt/weights/glean.weights";

	char log_file[] = "../mgmt/logs/soynet_glean.log";

	sprintf(extend_param, "BATCH_SIZE=%d ENGINE_SERIALIZE=%d ENGINE_FILE=%s WEIGHT_FILE=%s LOG_FILE=%s LICENSE_FILE=%s",
		batch_size, engine_serialize, engine_file, weight_file, log_file, license_file);

	void* handle = initSoyNet(cfg_file, extend_param);


	uint64_t dur_microsec = 0;
	uint64_t count = 0;

	// warm-up
	inference(handle);
  
  // SoyNet Run
  //uint64_t start_microsec = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
	feedData(handle, input.data());
	inference(handle);
	getOutput(handle, output.data());
	//uint64_t end_microsec = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
	//uint64_t dur = end_microsec - start_microsec;
	//printf("dur = %.2f [ms]\n", (dur / 1000.));
  printf("Inference Done!\n");
  
	// tensor2img
	Clamp(output, 0.f, 1.f);

	// rgb -> bgr
	Mat display;
	cvtColor(frame, display, COLOR_RGB2BGR);
  
	if (save_flag) {
		Mat save_img;
		display.convertTo(save_img, CV_8UC3, 1.0 * 255.0);
		imwrite("glean_result_x8.jpg", save_img);
		printf("Save Done!\n");
	}
 
	if (display_flag) {
		imshow(model_name, display);
		waitKey(0);
	}

  
	freeSoyNet(handle);
 
	return 0;
}
int main() {

	string model_name = "glean";

	string img_path = "../data/bird_32x32.png";
	
  int engine_serialize = 0; // 1: Every time you run it, you create a new SoyNet engine. 0: If you don't have it, make it
  
  // flag
  int display_flag = 1;
	int save_flag = 1;
 
	glean_img(model_name, img_path, engine_serialize, display_flag, save_flag);


	return 0;
}