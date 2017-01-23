#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <float.h>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>
#include <set>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

using namespace std;

char * loadSource(char *filePathName, size_t *fileSize);
void randomMemInit(float* data, int size);
void printArray(float* array, int start, int end, int dimension);
float computeStress(float* highD, float* lowD, int num_of_points, int n_original_dims, int n_projection_dims, int s, int e);
float distance(float* a, float* b, int dim);
float distance(int i, int j, float* highD, int dimension);
bool testPivots(int num_of_points, float* highD, float* prevPivot, float* pivot, float* output, int dimension);
float avgNNDistance(int pointIdx, float* highD, float* pivot, int dimension);
int testForce(int num_of_points, float* highD, float* lowD, float* velocity, float* pivot, float* prevForce, float* refForce, float* highDistance );
bool testLowDistance(int pointIdx, float* lowD, float* pivot, float* output);
int testPosition(int num_of_points, float* prevVelocity, float* velocity, float* prevLowD, float* lowD, float* force);
int countNewPivots(int pointIdx, std::set<float> &currentPivots, float* pivot_indices);
void normalize(float* data, int size, int dimension);
void shuffle(float* data, int size, int dimension);
int myrand( );
float* loadCSV( const char *filename, int& num_of_points, int& n_original_dims);
bool terminate(int iteration, int stop_iteration, float* sstress);

void level_force_directed(
	float* highD, 
	float* lowD,
	cl_mem& d_lowD,
	float* pivot_indices,
	float* hd_distances,
	float* ld_distances,
	cl_mem& d_hd_distances,
	cl_mem& d_ld_distances,
	cl_mem& d_pivot_indices,
	int num_of_points,
	int n_original_dims,
	int n_projection_dims,
	int start_index,
	int end_index, 
	bool interpolate, 
	int near_set_size,
	int random_set_size,
	cl_command_queue& commands,
	cl_kernel& force_kernel,
	cl_kernel& position_kernel,
	cl_kernel& ld_kernel,
	cl_kernel& stress_kernel,
	float* resultN,
	float* resultD,
	cl_mem& d_resultN,
	cl_mem& d_resultD,
	int num_of_groups);
	
int fill_level_count( int input, int *h );

#define SKIP_LINES 2
#define COSCLEN			51		// length of cosc filter
#define EPS				1.e-5f	// termination threshold
float cosc[] = {0.f,  -0.00020937301404f,      -0.00083238644375f,      -0.00187445134867f,      -0.003352219513758f,     -0.005284158713234f,     -0.007680040381756f,     -0.010530536243981f,     -0.013798126870435f,     -0.017410416484704f,     -0.021256733995966f,     -0.025188599234624f,     -0.029024272810166f,     -0.032557220569071f,     -0.035567944643756f,     -0.037838297355557f,     -0.039167132882787f,     -0.039385989227318f,     -0.038373445436298f,     -0.036066871845685f,     -0.032470479106137f,     -0.027658859359265f,     -0.02177557557417f,      -0.015026761314847f,     -0.007670107630023f,     0.f,      0.007670107630023f,      0.015026761314847f,      0.02177557557417f,       0.027658859359265f,      0.032470479106137f,      0.036066871845685f,      0.038373445436298f,      0.039385989227318f,      0.039167132882787f,      0.037838297355557f,      0.035567944643756f,      0.032557220569071f,      0.029024272810166f,      0.025188599234624f,      0.021256733995966f,      0.017410416484704f,      0.013798126870435f,      0.010530536243981f,      0.007680040381756f,      0.005284158713234f,      0.003352219513758f,      0.00187445134867f,       0.00083238644375f,       0.00020937301404f,       0.f};
int g_heir[50];	

int main(int argc, char** argv)
{
	ofstream myfile;
	myfile.open("shuttle-output.csv");	
	struct timeval start, end;

	int near_set_size = 4;
	int random_set_size = 4;
	int num_of_points = 1024; 
	int n_original_dims = 10;
	int n_projection_dims = 2;

//	float* highD = loadCSV("breast-cancer-wisconsin.csv", num_of_points, n_original_dims);
	
	float* highD = loadCSV("shuttle_trn_corr.csv", num_of_points, n_original_dims);
//	num_of_points = 30000;
	int group_size = 128;
	int num_of_groups = ceil(num_of_points / (float)group_size );
	
	float* velocity = new float[num_of_points * n_projection_dims];  
	float* force = new float[num_of_points * n_projection_dims];
	float* prevForce = new float[num_of_points * n_projection_dims];
	int* seed_memory = new int[num_of_points];
	float* pivot_indices = new float[num_of_points * (near_set_size + random_set_size)];
	float* hd_distances = new float[num_of_points * (near_set_size + random_set_size)]();
	float* ld_distances = new float[num_of_points * (near_set_size + random_set_size)]();
	float* lowD = new float[num_of_points * n_projection_dims];
	float* metadata = new float[48];
	float* prevLowD = new float[num_of_points * n_projection_dims];
	float* prevVelocity = new float[num_of_points * n_projection_dims];
	float* prevPivot = new float[num_of_points * (near_set_size + random_set_size)];
	float* resultN = new float[num_of_groups]();
	float* resultD = new float[num_of_groups]();

	metadata[0] = 0;  
	
	std::cout << "data size: " << num_of_points << "X" << n_original_dims << std::endl;
	int abc;
	//std::cin >> abc;
	float spring_force = .7f; 
	float damping = .3f; 
	float delta_time = 0.3f; 
	float freeness = .85f;
	float size_factor = 1.f / ((float) (near_set_size + random_set_size));

	int err;  
	srand(2016);


    normalize(highD, num_of_points, n_original_dims);
	std::cout << "here" << std::endl;

	//printArray(highD, 0, 10, n_original_dims);

	float high = 0.f;

	for(int i = 0; i < num_of_points * n_projection_dims; i++)
	{
		lowD[i] = ((float)(rand()%10000)/10000.f)-0.5f;
	}
	//printArray(lowD, 0, 10, n_projection_dims);

	//std::cout << "working on seed" << std::endl;
	for(int i = 0; i < num_of_points; i++)
	{
		seed_memory[i] = rand() % 2000;
	}
	//std::cout << "working on velocity " << std::endl;

	for(int i = 0; i < num_of_points * n_projection_dims; i++)
	{
		velocity[i] = 0.f;
		force[i] = 0.f;
	}

	//std::cout << "working on pivots" << std::endl;

	for(int i = 0;  i < num_of_points * (near_set_size + random_set_size); i++)
	{
		pivot_indices[i] = floor(rand() % num_of_points);
		hd_distances[i] = 1.3f;
		ld_distances[i] = 1.3f;
	}

	unsigned int correct;               // number of correct results returned

	size_t global;                      // global domain size for our calculation
	size_t local;                       // local domain size for our calculation

	cl_platform_id platform_ids[2];
	cl_device_id device_id;             // compute device id 
	cl_device_id devices[2];
	cl_context context;                 // compute context
	cl_command_queue commands;          // compute command queue
	cl_program program;                 // compute program
	cl_kernel force_kernel;                   // compute kernel
	cl_kernel position_kernel;
	cl_kernel ld_kernel;
	cl_kernel stress_kernel;


	cl_mem d_highD;
	cl_mem d_lowD;
	cl_mem d_velocity;
	cl_mem d_force;
	cl_mem d_seed_memory;
	cl_mem d_pivot_indices;
	cl_mem d_hd_distances;
	cl_mem d_ld_distances;
	cl_mem d_metadata;
	cl_mem d_resultN;
	cl_mem d_resultD;


	int gpu = 1;
	cl_uint numPlatforms;
	cl_int status;

	err = clGetPlatformIDs(2, platform_ids, &numPlatforms);
	std::cout << "number of platforms: " << numPlatforms << std::endl; 
	if (err != CL_SUCCESS)
	{
		printf("Error in clGetPlatformID, %d\n", err);
	}

	char buffer[10240];
	clGetPlatformInfo(platform_ids[0], CL_PLATFORM_NAME, 10240, buffer, NULL);
	std::cout << "Platform[1] Name: " << buffer << std::endl;

	clGetPlatformInfo(platform_ids[1], CL_PLATFORM_NAME, 10240, buffer, NULL);
    	std::cout << "Platform[2] Name: " << buffer << std::endl;
	
//	std::cout << "Running forward MDS on GPU " << std::endl;

cl_uint num_devices;
//cl_uint devices[2];
	err = clGetDeviceIDs(platform_ids[0] , CL_DEVICE_TYPE_ALL, 1, devices, &num_devices);
std::cout << "num of devices: " << num_devices << std::endl;

clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 10240, buffer, NULL);
std::cout << "first device name: " << buffer << std::endl;

//clGetDeviceInfo(devices[1], CL_DEVICE_NAME, 10240, buffer, NULL);
//std::cout << "second device name: " << buffer << std::endl;


	err = clGetDeviceIDs(platform_ids[0] , CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}

	// Create a compute context 
	//
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	// Create a command commands
	//

	commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n");
		return EXIT_FAILURE;
	}

	// Create the compute program from the source buffer
	//
	size_t sourceFileSize;
	char kernel_file[] = "glimmer_kernel.cl";
	char *cSourceCL = loadSource(kernel_file, &sourceFileSize);
	program = clCreateProgramWithSource(context, 1, (const char **) & cSourceCL, &sourceFileSize, &err);

	if (!program)
	{
		printf("Error: Failed to create compute program!\n");
		return EXIT_FAILURE;
	}

	// Build the program executable
	//
//	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
//	if(err == CL_INVALID_PROGRAM)std::cout << "invalid program" << std::endl;

        cl_int ret = clBuildProgram(program, 1, &device_id, "-D __OPENCLCC__ -I . -D FOO", NULL, NULL);
        if (ret != CL_SUCCESS) {fprintf(stderr, "Error in clBuildProgram: %d!\n", ret); ret = CL_SUCCESS; }
        size_t logsize = 0;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
        char * log = (char *) malloc (sizeof(char) *(logsize+1));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, logsize, log, NULL);
        fprintf(stderr, "CL_PROGRAM_BUILD_LOG:\n%s", log);
        free(log);

	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];

		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	// Create the compute kernel in the program we wish to run
	//
	force_kernel = clCreateKernel(program, "relax_a_point", &err);
    	position_kernel = clCreateKernel(program, "updatePosition", &err);
	ld_kernel = clCreateKernel(program, "computeLdDistance", &err);
	stress_kernel = clCreateKernel(program, "computeStress", &err);
	if (!force_kernel || !position_kernel || !ld_kernel || !stress_kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	std::cout << "assigning device buffers " << std::endl;
	// Create the input and output arrays in device memory for our calculation

	d_highD =  clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * num_of_points * n_original_dims, NULL, &err);
	d_lowD = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_points * n_projection_dims, NULL, &err);
	d_velocity = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_points * n_projection_dims, NULL, &err);
	d_force = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_points * n_projection_dims, NULL, &err);
	d_seed_memory = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * num_of_points, NULL, &err);
	d_pivot_indices = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_points * (near_set_size + random_set_size), NULL, &err);
	d_hd_distances = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_points * (near_set_size + random_set_size), NULL, &err);
	d_ld_distances = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_points * (near_set_size + random_set_size), NULL, &err);
	d_metadata = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 48, NULL, &err);
	d_resultN = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_groups, NULL, &err);
	d_resultD = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_groups, NULL, &err);


	if (!d_highD || !d_lowD || !d_velocity || !d_force || !d_pivot_indices || !d_seed_memory || !d_hd_distances || !d_ld_distances || !d_metadata)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    


	//Launch OpenCL kernel
	size_t localWorkSize[3] = {512, 0, 0}, globalWorkSize[3] = {num_of_points, 0, 0};

	int iteration = 0;
	int start_index = 0;
	int end_index = num_of_points;
	
	err = clSetKernelArg(force_kernel, 0, sizeof(cl_mem), (void *)&d_highD);
	err |= clSetKernelArg(force_kernel, 1, sizeof(cl_mem), (void *)&d_lowD);
	err |= clSetKernelArg(force_kernel, 2, sizeof(cl_mem), (void *)&d_velocity);
	err |= clSetKernelArg(force_kernel, 3, sizeof(cl_mem), (void *)&d_force);
	err |= clSetKernelArg(force_kernel, 4, sizeof(cl_mem), (void *)&d_seed_memory);
	err |= clSetKernelArg(force_kernel, 5, sizeof(cl_mem), (void *)&d_pivot_indices);
	err |= clSetKernelArg(force_kernel, 6, sizeof(int), (void *)&start_index);
	err |= clSetKernelArg(force_kernel, 7, sizeof(int), (void *)&end_index);
	err |= clSetKernelArg(force_kernel, 8, sizeof(int), (void *)&n_original_dims);
	err |= clSetKernelArg(force_kernel, 9, sizeof(float), (void *)&n_projection_dims);
	err |= clSetKernelArg(force_kernel, 10, sizeof(float), (void *)&near_set_size);
	err |= clSetKernelArg(force_kernel, 11, sizeof(float), (void *)&random_set_size);
	err |= clSetKernelArg(force_kernel, 12, sizeof(cl_mem), (void*)&d_hd_distances);
	err |= clSetKernelArg(force_kernel, 13, sizeof(cl_mem), (void*)&d_metadata);
    
        err |= clSetKernelArg(position_kernel, 0, sizeof(cl_mem), (void*)&d_velocity);
   	err |= clSetKernelArg(position_kernel, 1, sizeof(cl_mem), (void*)&d_lowD);
        err |= clSetKernelArg(position_kernel, 2, sizeof(cl_mem), (void*)&d_force);
        err |= clSetKernelArg(position_kernel, 3, sizeof(int), (void*)&start_index);
	err |= clSetKernelArg(position_kernel, 4, sizeof(int), (void*)&end_index);
	err |= clSetKernelArg(position_kernel, 5, sizeof(int), (void*)&n_projection_dims);
        err |= clSetKernelArg(position_kernel, 6, sizeof(float), (void*)&delta_time);
        err |= clSetKernelArg(position_kernel, 7, sizeof(float), (void*)&freeness);
	
	err |= clSetKernelArg(ld_kernel, 0, sizeof(cl_mem), (void*)&d_lowD);
   	err |= clSetKernelArg(ld_kernel, 1, sizeof(cl_mem), (void*)&d_pivot_indices);
        err |= clSetKernelArg(ld_kernel, 2, sizeof(cl_mem), (void*)&d_ld_distances);
        err |= clSetKernelArg(ld_kernel, 3, sizeof(int), (void*)&start_index);
	err |= clSetKernelArg(ld_kernel, 4, sizeof(int), (void*)&end_index);
	err |= clSetKernelArg(ld_kernel, 5, sizeof(int), (void*)&n_projection_dims);
        err |= clSetKernelArg(ld_kernel, 6, sizeof(int), (void*)&near_set_size);
        err |= clSetKernelArg(ld_kernel, 7, sizeof(float), (void*)&random_set_size);


	std::cout << "Initial number of groups: " << num_of_groups << std::endl;


	err |= clSetKernelArg(stress_kernel, 0, sizeof(cl_mem), (void *)&d_hd_distances);
        err |= clSetKernelArg(stress_kernel, 1, sizeof(cl_mem), (void *)&d_ld_distances);               
        err |= clSetKernelArg(stress_kernel, 2, sizeof(cl_float) * group_size, NULL);    
        err |= clSetKernelArg(stress_kernel, 3, sizeof(cl_float) * group_size, NULL);
        err |= clSetKernelArg(stress_kernel, 4, sizeof(int), (void *)&num_of_points);
        err |= clSetKernelArg(stress_kernel, 5, sizeof(cl_mem), (void *)&d_resultN);
        err |= clSetKernelArg(stress_kernel, 6, sizeof(cl_mem), (void *)&d_resultD);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	err &= clEnqueueWriteBuffer(commands, d_pivot_indices, CL_TRUE, 0, sizeof(float) * num_of_points * (near_set_size + random_set_size), pivot_indices, 0, NULL, NULL);
	err &= clEnqueueWriteBuffer(commands, d_highD, CL_TRUE, 0, sizeof(float) * num_of_points * n_original_dims, highD, 0, NULL, NULL);
	err &= clEnqueueWriteBuffer(commands, d_lowD, CL_TRUE, 0, sizeof(float) * num_of_points * n_projection_dims, lowD, 0, NULL, NULL);
	err &= clEnqueueWriteBuffer(commands, d_velocity, CL_TRUE, 0, sizeof(float) * num_of_points * n_projection_dims, velocity, 0, NULL, NULL);
	err &= clEnqueueWriteBuffer(commands, d_seed_memory, CL_TRUE, 0, sizeof(float) * num_of_points, seed_memory, 0, NULL, NULL);
	err &= clEnqueueWriteBuffer(commands, d_metadata, CL_TRUE, 0, sizeof(float) * 48, metadata, 0, NULL, NULL);
	err &= clEnqueueWriteBuffer(commands, d_force, CL_TRUE, 0, sizeof(float) * num_of_points * n_projection_dims, force, 0, NULL, NULL); 
	err &= clEnqueueWriteBuffer(commands, d_hd_distances, CL_TRUE, 0, sizeof(float) * num_of_points * (near_set_size + random_set_size), hd_distances, 0, NULL, NULL);

	//Multi-level code
	bool g_done = false;
	bool g_interpolating = false;
//	int iteration = 0;
	int stop_iteration = 0;
	int g_levels = fill_level_count( num_of_points, g_heir );
	int g_current_level = g_levels-1;

//printArray(pivot_indices, 0, 10, 8);
long st = time(NULL);
gettimeofday(&start, NULL);
	while( !g_done ) {
		if( g_interpolating ) // interpolate
		{
			cout << "Interpolating " << g_heir[g_current_level] << " and " << g_heir[g_current_level + 1] << endl;
			num_of_groups = ceil(g_heir[g_current_level] / (float)group_size);
			std::cout << "number of groups : " << num_of_groups << std::endl;
			level_force_directed(highD, lowD, d_lowD, pivot_indices, hd_distances, ld_distances, d_hd_distances, d_ld_distances, d_pivot_indices, num_of_points,
			n_original_dims, n_projection_dims, g_heir[ g_current_level+1 ], g_heir[g_current_level], g_interpolating,  near_set_size,
			random_set_size, commands, force_kernel, position_kernel, ld_kernel, stress_kernel, resultN, resultD, d_resultN, d_resultD, num_of_groups);


		}
		else
		{
			cout << "Relaxing " << g_heir[g_current_level] << " and 0" << endl; 
			num_of_groups = ceil(g_heir[g_current_level] / (float)group_size);
			std::cout << "number of groups : " << num_of_groups << std::endl;	
			level_force_directed(highD, lowD, d_lowD, pivot_indices, hd_distances, ld_distances, d_hd_distances, d_ld_distances, d_pivot_indices, num_of_points,
			n_original_dims, n_projection_dims, 0, g_heir[g_current_level], g_interpolating,  near_set_size,
			random_set_size, commands, force_kernel, position_kernel, ld_kernel, stress_kernel, resultN, resultD, d_resultN, d_resultD, num_of_groups);


		}
		if( true ) {

			if( g_interpolating ) {
//				cout << "setting interpolate to false, activating relaxing" << endl;
				g_interpolating = false;
			}
			else {
//				cout << "setting interpolating to true, activate interpolate" << endl;
				g_current_level--; // move to the next level down
				g_interpolating = true;
//				cout << "current level " << g_current_level << endl;
				//stop_iteration = iteration + 1;

				if( g_current_level < 0 ) {
					cout << "done " << endl;
					g_done = true;
				}
			}
		}

		iteration++;	// increment the current iteration count			
	}
	gettimeofday(&end, NULL);
	err &= clEnqueueReadBuffer(commands, d_lowD, CL_TRUE, 0, sizeof(float) * num_of_points * n_projection_dims, lowD, 0, NULL, NULL);
	long et = time(NULL);
	std::cout << "elapsed time: " << (et - st) << std::endl;
	//Multi-level code
	  double elapsed = (end.tv_sec - start.tv_sec) * 1000 +
              ((end.tv_usec - start.tv_usec)/1000.0);
        std::cout << "elapsed time: " << elapsed << " ms " << std::endl;
	/*bool interpolate = false;
	level_force_directed(highD, lowD, d_lowD, pivot_indices, d_pivot_indices, num_of_points,
	n_original_dims, n_projection_dims, 0, 6000, false,  near_set_size,
	random_set_size, commands, force_kernel, position_kernel);
	
	int dd;
	cin >> dd;
	level_force_directed(highD, lowD, d_lowD, pivot_indices, d_pivot_indices, num_of_points,
        n_original_dims, n_projection_dims, 6000, num_of_points, true,  near_set_size,
        random_set_size, commands, force_kernel, position_kernel);
	cin >> dd;
	level_force_directed(highD, lowD, d_lowD, pivot_indices, d_pivot_indices, num_of_points,
        n_original_dims, n_projection_dims, 0, num_of_points, false,  near_set_size,
        random_set_size, commands, force_kernel, position_kernel);
	cin >> dd;
	*/
	
	for(int i = 0; i < num_of_points; i++)
	{
		//for(int j = 0; j < n_projection_dims; j++)
		//{
			myfile << lowD[i * n_projection_dims + 0] << "," << lowD[i * n_projection_dims + 1] << "\n";
		//}
	}
	myfile.close();


//	long elapsed = etime - stime;
//	std::cout<< "Elapsed time " << elapsed << std::endl;

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}



	printf("Matrix multiplication completed...\n"); 

	//Shutdown and cleanup
	delete[] highD;
	delete[] lowD;
	delete[] velocity;
	delete[] prevVelocity;
	delete[] force;
	delete[] seed_memory;
	delete[] pivot_indices;
	delete[] hd_distances;
	delete[] ld_distances;
	delete[] metadata;
	delete[] prevLowD;
	delete[] prevForce;
	
	clReleaseMemObject(d_highD);
	clReleaseMemObject(d_lowD);
	clReleaseMemObject(d_velocity);
	clReleaseMemObject(d_force);
	clReleaseMemObject(d_seed_memory);
	clReleaseMemObject(d_pivot_indices);
	clReleaseMemObject(d_hd_distances);
	clReleaseMemObject(d_ld_distances);
	clReleaseMemObject(d_metadata);
	clReleaseProgram(program);
	clReleaseKernel(force_kernel);
    	clReleaseKernel(position_kernel);
	clReleaseKernel(ld_kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return 0;
}

char * loadSource(char *filePathName, size_t *fileSize)
{
	FILE *pfile;
	size_t tmpFileSize;
	char *fileBuffer;
	pfile = fopen(filePathName, "rb");

	if (pfile == NULL)
	{
		printf("Open file %s open error!\n", filePathName);
		return NULL;
	}

	fseek(pfile, 0, SEEK_END);
	tmpFileSize = ftell(pfile);

	fileBuffer = (char *)malloc(tmpFileSize);

	fseek(pfile, 0, SEEK_SET);
	fread(fileBuffer, sizeof(char), tmpFileSize, pfile);

	fclose(pfile);

	//debug================================
	//for (int i = 0; i < tmpFileSize; i++)
	//{
	//	printf("%c", fileBuffer[i]);
	//}
	//=====================================

	*fileSize = tmpFileSize;
	return fileBuffer;
}

// Allocates a matrix with random float entries.
void randomMemInit(float* data, int size)
{
	int i;

	for (i = 0; i < size; ++i)
		data[i] = rand() % 10; // / RAND_MAX;
}


void printArray(float* array, int start, int end, int dimension)
{
	for(int i = start; i < end; i++)
	{
		for(int d = 0; d < dimension; d++)
		{
			if(array[i * dimension + d] >= 0)continue;
			std::cout << array[i * dimension + d] << ", ";
		}
//		std::cout << std::endl;
	}
}

float computeStress(float* highD, float* lowD, int num_of_points, int n_original_dims, int n_projection_dims, int s, int e) {
	float stress = 0.f;
	for(int i = s; i < e; i++)
	{
		for(int j = i + 1; j < e; j++)
		{
			//float hd = distance(highD + i * n_original_dims, highD + j * n_original_dims, n_original_dims);
			//float ld = distance(lowD + i * n_projection_dims, lowD + j * n_projection_dims, n_projection_dims);
			float hd = distance(i, j, highD, n_original_dims);
			float ld = distance(i, j, lowD, n_projection_dims);
			float delta =  fabs(hd - ld);
			stress += delta;
		}
	}

	return stress;
}

float computeSparseStress(float* highD, float* lowD, float* pivot_indices, int num_of_points, int near_set_size, int random_set_size, int n_original_dims, int n_projection_dims, int s, int e) {
	float numerator = 0.f;
	float denominator = 0.f;
	int pivot_size = near_set_size + random_set_size;
	for(int i = 0; i < e; i++)
	{
		for(int j = 0; j < pivot_size; j++)
		{
			float hd = distance(i, (int)pivot_indices[i * pivot_size + j], highD, n_original_dims);
			float ld = distance(i, (int)pivot_indices[i * pivot_size + j], lowD, n_projection_dims);
			float delta =  (hd - ld);
			numerator += (delta * delta);
			denominator += hd * hd;
		}
	}
	return numerator / denominator;
}

float computeSparseStress(float* highD, float* lowD, float *hd_distances, float* ld_distances, float* pivot_indices, int num_of_points, int near_set_size, int random_set_size, int n_original_dims, int n_projection_dims, int s, int e) {
	float numerator = 0.f;
	float denominator = 0.f;
	int pivot_size = near_set_size + random_set_size;
	for(int i = s; i < e; i++)
	{
		for(int j = 0; j < pivot_size; j++)
		{
			//float hd = distance(i, (int)pivot_indices[i * pivot_size + j], highD, n_original_dims);
			
			//float hd = distance(i, (int)pivot_indices[i * pivot_size + j], highD, n_original_dims);
			float hd = hd_distances[i * pivot_size + j];
			float ld = ld_distances[i * pivot_size + j];
				
				
			//float ld = distance(i, (int)pivot_indices[i * pivot_size + j], lowD, n_projection_dims);
			float delta =  (hd - ld);
			numerator += (delta * delta);
			denominator += hd * hd;
		}
	}
	return numerator / denominator;
}

float computeAlternateStress(float* hd_distances, float* ld_distances, float* pivot_indices, int num_of_points, int near_set_size, int random_set_size, int n_original_dims, int n_projection_dims, int s, int e) {
	float numerator = 0.f;
	float denominator = 0.f;
	int pivot_size = near_set_size + random_set_size;
	
	for(int i = 0; i < e; i++)
	{
		for(int j = 0; j < pivot_size; j++)
		{
			float hd = hd_distances[i * pivot_size + j];
			if(hd == 1000.f)hd = 0.f;
			float ld = ld_distances[i * pivot_size + j];
			float delta =  (hd - ld);
			numerator += (delta * delta);
			denominator += hd * hd;
		}
	}
	return numerator / denominator;
}

float distance(float* a, float* b, int dim)
{
	float dist = 0.f;
	for(int d = 0; d < dim; d++)
	{
		dist += (a[d] - b[d]) * (a[d] - b[d]);
	}
	return sqrt(dist);
}

bool testLowDistance(int pointIdx, float* lowD, float* pivot, float* output)
{
	for(int i = 0; i < 24; i++)
	{
		float dist  = distance(pointIdx, (int)pivot[pointIdx * 24 + i], lowD, 2);
		std::cout << dist << " <> " << output[pointIdx * 24 + i] << std::endl;
	}
}

bool testPivots(int num_of_points, float* highD, float* prevPivot, float* pivot, float* output, int dimension)
{
bool sorted = true;
bool preserveNear = true;
//num_of_points = 1;
for(int pointIdx = 0; pointIdx < num_of_points; pointIdx++){
	float distances[24];
	float prev_dist = 0.f;

//	bool sorted = true;
	for(int i = 0; i < 24; i++)
	{
		distances[i] = distance(pointIdx, (int)pivot[pointIdx * 24 + i], highD, dimension);
//		std::cout << pivot[pointIdx * 24 + i] << ":" <<  distances[i] << " <>  " << output[pointIdx * 24 + i] << std::endl;
		if(prev_dist > distances[i])
		{
			std::cout << "reversal " << pointIdx << ":" << prev_dist << " <> " << distances[i]  << std::endl;
//			sorted = false;
			return false;
		}
		prev_dist = distances[i];
	}

	if(!sorted)std::cout << "Not sorted " << std::endl;
	preserveNear = false;
	int j = 0;
	int count = 0;
	for(int i = 0; i < 14; i++)
	{
		for(j = 0; j < 24; j++)
		{
			if(prevPivot[pointIdx * 24 + i] == pivot[pointIdx * 24 + j])count++;
		}
	}

	if(count >= 14)preserveNear = true;
	if(!preserveNear){ std::cout << pointIdx << ":near set not preserved " << count << std::endl; return false;} //std::cout << "Near set is not preserved " << std::endl;
}
	return sorted && preserveNear;
}


// highD is array of size 1024 x dimension
float distance(int i, int j, float* data, int dimension)
{
	//std::cout << "Distance between " << i << " and " << j << ": ";
	float norm = 0.f;
	for(int d = 0; d < dimension; d++)
	{
		float diff = (data[i * dimension + d] - data[j * dimension + d]);
		norm += diff * diff;
	}

	//std::cout << norm << std::endl;
	return (float)sqrt(norm);
}

float avgNNDistance(int pointIdx, float* highD, float* pivot, int dimension)
{

	float distances[10];
	float sum = 0.f;
	for(int i = 0; i < 10; i++)
	{
		distances[i] = distance(pointIdx, (int)pivot[i], highD, dimension);

		sum += distances[i];
	}      
	return sum / 10.f;
}

int testForce(int num_of_points, float* highD, float* lowD, float* velocity, float* pivot, float* prevForce, float* refForce, float* output )
{
int count = 0;
for(int pointIdx = 0; pointIdx < num_of_points; pointIdx++){
	float dir_vector[2];
	float pivot_distances_low[24];
	float pivot_distances_high[24];
	float rel_velocity[2];
	float force[2];

	float spring_force = .7f; 
	float damping = .3f; 
	float delta_time = 0.3f; 
	float freeness = .85f;
	float size_factor = 1.f / 24.f;
    
    for(int i = 0; i < 2; i++)
    {
        force[i] = prevForce[pointIdx * 2 + i];
    }

	// Compute high distances
	for(int i = 0; i < 24; i++)
	{
		float hi = 0.f;
		for( int k = 0; k < 10; k++ ) {

			float norm = (highD[pointIdx * 10 + k] - highD[(int)pivot[pointIdx * 24 + i] * 10 + k]);
			hi += norm * norm;
		}
		pivot_distances_high[i] = sqrt(hi);
//		std::cout << pivot_distances_high[i] << " <> " << output[pointIdx * 24 + i] << std::endl;
	}

//	int count = 0;
	// Compute low distances
	for(int i = 0; i < 24; i++)
	{              
		float norm = 0.f;
		for(int k = 0; k < 2; k++) {
			dir_vector[k] = lowD[(int)pivot[pointIdx * 24 + i] * 2 + k] - lowD[pointIdx * 2 + k];
			norm += dir_vector[k] * dir_vector[k];
		}
		norm = sqrt(norm);
		pivot_distances_low[i] = norm;
//		std::cout << pivot_distances_low[i] << " <> " << output[pointIdx * 24 + i] << std::endl;
		if(norm > 1.e-6 && pivot_distances_high[i] != 1000.f) {
			for(int k = 0; k < 2; k++) {
				dir_vector[k] /= norm;
			}
		}

		// relative velocity
		for(int k = 0; k < 2; k++) {
			rel_velocity[k] = velocity[(int)pivot[pointIdx * 24 + i] * 2 + k] - velocity[pointIdx * 2 + k];
//			std::cout << rel_velocity[k] << std::endl;
//			std::cout << velocity[(int)pivot[i] * 2 + k] << " - " << velocity[pointIdx * 2 + k] << " = " << rel_velocity[k] << " <> " << output[pointIdx * 48 + i * 2 + k] << std::endl;
//                        if(rel_velocity[k] == output[pointIdx * 48 + i * 2 + k])count++;
		}

		// calculate difference
		float delta_distance = (pivot_distances_high[i] - pivot_distances_low[i]) * spring_force;

		// compute damping value
		norm = 0.f;
		for(int k = 0; k < 2; k++) {
			norm += dir_vector[k] * rel_velocity[k];
		}
		delta_distance += norm * damping;
	//	std::cout << delta_distance << "  == " <<  output[pointIdx * 24 + i] << std::endl;
		// accumulate the force
	
		for(int k = 0; k < 2; k++) {
            force[k] += dir_vector[k] * delta_distance;
			//std::cout << velocity[(int)pivot[i]] rel_velocity[k] << " <> " << output[i * 2 + k] << std::endl;
			//if(rel_velocity[k] == output[i * 2 + k])count++;
		}	
	}
    
    for(int k = 0; k < 2; k++) {
        force[k] *= size_factor;
    }


//	count = 0;
	for(int k = 0; k < 2; k++)
	{
//		std::cout << force[k] << " <> " << refForce[pointIdx * 2 + k] << std::endl;
		if(force[k] == refForce[pointIdx * 2 + k])count++;
	}
}

	return count;
}

int testPosition(int num_of_points, float* prevVelocity, float* velocity, float* prevLowD, float* lowD, float* force)
{
int count = 0;
for(int gid = 0; gid < num_of_points; gid++){
    
    //int gid = pointIdx;
    float delta_time = 0.3f;
    float freeness = 0.85f;
    float vel[2];
    float position[2];
    
    for(int k = 0; k < 2; k++)
    {
        vel[k] = prevVelocity[gid * 2 + k];
        position[k] = prevLowD[gid * 2 + k];
    }
    
    // update new velocity
    // v = v0 + at
    for(int k = 0; k < 2; k++) {
        float v0 = vel[k];
        float v = v0 + force[gid * 2 + k] * delta_time;
        v *= freeness;
        vel[k] = max(min(v, 2.f), -2.f);
        
    }
    
    // update new positions
    // x = x0 + vt
    for(int k = 0; k < 2; k++) {
        position[k] += vel[k] * delta_time;
    }

    //int count = 0;
    for(int k = 0; k < 2; k++)
    {
//        std::cout << position[k] << " <> " << lowD[gid * 2 + k] << std::endl;
        if(position[k] == lowD[gid * 2 + k])count++;
    }
}
//	if(count == 2 * num_of_points)std::cout << "Passed Position Test " << std::endl;
    return count;
    
}


int countNewPivots(int pointIdx, std::set<float>& currentPivots, float* pivot_indices)
{
	int count = 0;
	for(int i = 0; i < 24; i++)
	{
		if(currentPivots.count(pivot_indices[pointIdx * 24 + i]) == 0)
		{
			count++;
			currentPivots.insert(pivot_indices[pointIdx * 24 + i]);
		}
	}
	return count;
}


void normalize(float* data, int size, int dimension)
{
	std::cout << "IN " << size << "--" << dimension << std::endl;
    float* max_vals = new float[dimension];
    float* min_vals = new float[dimension]; 
    for( int i = 0; i < dimension; i++ ) {
        max_vals[ i ] = 0.f;
        min_vals[ i ] = 10000.0f;
    }

	int dum;
//	cin >> dum;
    int k = 0;
    for( int i = 0; i < size; i++ ) {        
        for( int j = 0; j < dimension; j++ ) {
            if( data[i * (dimension) + j] > max_vals[j] ) {
                max_vals[j] = data[i * (dimension) +j];
            }
            if( data[i*(dimension)+j] < min_vals[j] ) {
                min_vals[j] = data[i*(dimension)+j];                    
            }
        }
    }
//	cin >> dum;
    for( int i = 0; i < dimension; i++ ) {
        max_vals[ i ] -= min_vals[ i ];
    }

    for( int i = 0; i < size; i++ ) {        
        for( int j = 0; j < dimension; j++ ) {
            if( (max_vals[j] - min_vals[j]) < 0.0001f ) {
                data[i*(dimension)+j] = 0.f;
            }
            else {
                data[i*(dimension)+j] = 
                    (data[i*(dimension)+j] - min_vals[j])/max_vals[j];
                if(  data[i*(dimension)+j] >= 1000.f || data[i * dimension + j] <= -1000  ) 
                    data[i*(dimension)+j] = 0.f;
            }
        }
    }
//cin >> dum;
    delete max_vals ;
    delete min_vals;
	std::cout << "OUT" << std::endl;
}

void shuffle(float* data, int size, int dimension)
{
    float *shuffle_temp = new float[dimension];
    int shuffle_idx = 0;
    for( int i = 0; i < size * dimension; i += dimension ) {

        shuffle_idx = i + ( myrand() % (size - (i / dimension)) ) * dimension;
        for( int j = 0; j < dimension; j++ ) {    // swap

            shuffle_temp[j]=data[i+j];
            data[i+j] = data[shuffle_idx+j];
            data[shuffle_idx+j] = shuffle_temp[j];
        }        
    }
    delete shuffle_temp;
}

int myrand( ) {

    unsigned int n = (unsigned int)rand();
    unsigned int m = (unsigned int)rand();
	std::cout << "n and m: " << n << ", " << m << ":" << (int)((n << 16) + m) << std::endl;
	return 5;
//    return ((int)((n << 16) + m));
}


float* loadCSV( const char *filename, int& num_of_points, int& n_original_dims ) {

    char line[65536];    // line of input buffer
    char item[512];        // single number string
    float *data = NULL;    // output data

    // open the file 
    ifstream fp;
    fp.open( filename);


    // get dataset statistics
    int N = 0;
    n_original_dims = 0;

    while( fp.getline( line, 65535) != NULL && N < 43502) {

        // count the number of points (for every line)
        N++;

        // count the number of dimensions (once)
        if( n_original_dims == 0 && N > SKIP_LINES) {
            int i = 0;
            while( line[i] != '\0' ) {
                if( line[i] == ',' ) {
                    n_original_dims++;
                }
                i++;
            }
            n_original_dims++;
        }
    }
    fp.close();
    std::cout << "number of data points " << N << " and " << n_original_dims;
    N -= SKIP_LINES;

    // allocate our data buffer    
    data = (float*)malloc(sizeof(float)*N*n_original_dims);

    // read the data into the buffer
    fp.open(filename);
    int skip = 0;
    int k = 0;
	int c = 0;
    while( fp.getline( line, 65535) != NULL && c < 43502 ) {

        int done = 0;
        int i = 0;
        int j = 0;
        while( !done ) {

            // skip the introductory lines
            if( skip++ < SKIP_LINES ) {

                done = 1;
            }
            else {

                // parse character data
                if( line[i] == ',' ) {

                    item[j] = '\0';
                    data[k++] = (float) atof( item );
                    j = 0;
                }
                else if( line[i] == '\n' || line[i] == '\0' ) {

                    item[j] = '\0';
                    data[k++] = (float) atof( item );
                    done++;
                }
                else if( line[i] != ' ' ) {

                    item[j++] = line[i];
                }
                i++;
            }
        }
	c++;
    }
    num_of_points = N;
    return data;
}

int verify_hd_distances(float* highD, float* pivot_indices, float* hd_distances, 
int s, int e, int n_original_dims, int near_set_size, int random_set_size)
{
	int count = 0;
	int pivot_size = near_set_size + random_set_size;
	for(int i = s; i < e; i++)
	{
		for(int j = 0; j < pivot_size; j++)
		{
			float hd = distance(i, (int)pivot_indices[i * pivot_size + j], highD, n_original_dims);
			if(hd != hd_distances[i * pivot_size + j])
			{
				count++;
				std::cout << hd_distances[i * pivot_size + j] << std::endl;
			}
		}
	}
	return count;
}

void level_force_directed(
	float* highD, 
	float* lowD,
	cl_mem& d_lowD,
	float* pivot_indices,
	float* hd_distances,
	float* ld_distances,
	cl_mem& d_hd_distances,
	cl_mem& d_ld_distances,
	cl_mem& d_pivot_indices,
	int num_of_points,
	int n_original_dims,
	int n_projection_dims,
	int start_index,
	int end_index, 
	bool interpolate, 
	int near_set_size,
	int random_set_size,
	cl_command_queue& commands,
	cl_kernel& force_kernel,
	cl_kernel& position_kernel,
	cl_kernel& ld_kernel,
	cl_kernel& stress_kernel,
	float* resultN,
	float* resultD,
	cl_mem& d_resultN,
	cl_mem& d_resultD,
	int num_of_groups)
{
	ofstream fout;
	fout.open("stress.csv", std::ofstream::out | std::ofstream::app);	
	// Initialize near sets using random values
	int modular_operand = interpolate ? start_index : end_index;
	for(int i = 0; i < end_index; i++)
	{       
		for(int j = 0; j < near_set_size; j++)
		{       
			pivot_indices[i * (near_set_size + random_set_size) + j] = floor(rand() % modular_operand);
		}
	}
	
	int group_size = 128;
	size_t localWorkSize[3] = {group_size, 0, 0};
	size_t globalWorkSize[3] = {(int)(ceil((end_index - start_index) / group_size) * group_size), 0, 0};

	int err = 0;
	//int err = clSetKernelArg(force_kernel, 5, sizeof(cl_mem), (void *)&d_pivot_indices);
    	err |= clSetKernelArg(force_kernel, 6, sizeof(int), (void *)&start_index);
    	err |= clSetKernelArg(force_kernel, 7, sizeof(int), (void *)&end_index);

	err |= clSetKernelArg(position_kernel, 3, sizeof(int), (void*)&start_index);
    	err |= clSetKernelArg(position_kernel, 4, sizeof(int), (void*)&end_index);
	
	err |= clSetKernelArg(ld_kernel, 3, sizeof(int), (void*)&start_index);
    	err |= clSetKernelArg(ld_kernel, 4, sizeof(int), (void*)&end_index);
	
	err |= clEnqueueWriteBuffer(commands, d_pivot_indices, CL_TRUE, 0, sizeof(float) * num_of_points * (near_set_size + random_set_size), pivot_indices, 0, NULL, NULL);
	float* sstress = new float[end_index - start_index];

	int length = end_index * 8;
	err |= clSetKernelArg(stress_kernel, 4, sizeof(int), (void *)&length);
	

	size_t stress_g_size[3] = {(int)(ceil((length / 8.f) /(float) group_size) * group_size), 0, 0};

	for(int iteration = start_index; iteration < end_index; iteration++)
	{
		// launch kernel
		err = clEnqueueNDRangeKernel(commands, force_kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	  	err = clEnqueueNDRangeKernel(commands, position_kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		err = clEnqueueNDRangeKernel(commands, ld_kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		err = clEnqueueNDRangeKernel(commands, stress_kernel, 1, NULL, stress_g_size, localWorkSize, 0, NULL, NULL);	

		
		// read data to compute stress : lowD and pivot_indices
//		err &= clEnqueueReadBuffer(commands, d_pivot_indices, CL_TRUE, 0, sizeof(float) * num_of_points * (near_set_size + random_set_size), pivot_indices, 0, NULL, NULL);
//		err &= clEnqueueReadBuffer(commands, d_hd_distances, CL_TRUE, 0, sizeof(float) * num_of_points * (near_set_size + random_set_size), hd_distances, 0, NULL, NULL);
//		err &= clEnqueueReadBuffer(commands, d_ld_distances, CL_TRUE, 0, sizeof(float) * num_of_points * (near_set_size + random_set_size), ld_distances, 0, NULL, NULL);
//		err &= clEnqueueReadBuffer(commands, d_lowD, CL_TRUE, 0, sizeof(float) * num_of_points * n_projection_dims, lowD, 0, NULL, NULL);
		err &= clEnqueueReadBuffer(commands, d_resultN, CL_TRUE, 0, sizeof(float) * num_of_groups, resultN, 0, NULL, NULL);
		err &= clEnqueueReadBuffer(commands, d_resultD, CL_TRUE, 0, sizeof(float) * num_of_groups, resultD, 0, NULL, NULL);

		// Test if distances are computed correctly
//		int countH = 0, countL = 0;
//		for(int i = 0; i < length; i++)
//		{
//			if(isinf(hd_distances[i]) || isnan(hd_distances[i])) countH++;
//			if(isinf(ld_distances[i]) || isnan(ld_distances[i])) countL++;
//		}		
//		if(countH > 0 || countL > 0)
//{std::cout << "length, invalid values: " << length << ", " << countH << ", " << countL << std::endl;} 
		if(err == CL_SUCCESS)
		{
		//	if(iteration == start_index)printArray(pivot_indices, 0, end_index, 8);
		//	std::cout << "hd" << std::endl;
		//	printArray(hd_distances, 0, end_index, 8);
		//	std::cout << "ld" << std::endl;
		//	printArray(ld_distances, 0, end_index, 8);

//			float sparse_stress = computeAlternateStress(hd_distances, ld_distances, pivot_indices, num_of_points, near_set_size, random_set_size, n_original_dims, n_projection_dims, 0, end_index);
			

			float d = 0.f;
			float n = 0.f;
			for(int k = 0; k < num_of_groups; k++)
			{
//				std::cout << resultN[k] << std::endl;
				n += resultN[k];
				d += resultD[k];
			}
//			std::cout << "n, d: " << n << ", " << d << std::endl;  
			float otherStress = n / d;
//			sstress[iteration - start_index] = sparse_stress;
			fout << otherStress << "\n";
//			cout << otherStress << "\n";
			sstress[iteration - start_index] = otherStress;
//			std::cout << "stresses: "  << sparse_stress << " <> " << otherStress << std::endl;
			if(terminate(iteration, start_index, sstress ))
			{
				std::cout << "Stopping at iteration: " << iteration - start_index << std::endl;
				break;
			}
		}
	}
	fout.close();
	delete[] sstress;
}

bool terminate(int iteration, int stop_iteration, float* sstress)
{
	if(iteration - stop_iteration >= 1000) return true;
	float signal = 0.f;
	if( iteration - stop_iteration > COSCLEN ) {

		for( int i = 0; i < COSCLEN; i++ ) {

			signal += sstress[ (iteration - COSCLEN)+i ] * cosc[ i ];
		}

		if( fabs( signal ) < EPS ) {
			return true;
		}
	}

	return false;
}

int fill_level_count( int input, int *h ) {
	static int levels = 0;
	printf("h[%d]=%d\n",levels,input);
	h[levels]=input;
	levels++;
	if( input <= 1000 )
		return levels;
	return fill_level_count( input / 8, h );
}
