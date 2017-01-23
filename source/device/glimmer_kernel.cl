int rand(int* seed);
void swap (__private float* a,__private float* b );
int partition (__private float array[],__private float array1[], int l, int h);
void quickSortIterative (__private float array[],__private float array1[], int l, int h);

__kernel void relax_a_point(
	__global float* highD, 
	__global float* lowD, 
	__global float* velocity, 
	__global float* force, 
	__global int* seed_memory,
	__global float* pivot_indices, 
	const int start_index,
	const int end_index,
	const int n_original_dims,
	const int n_projection_dims,
	const int near_set_size,
	const int random_set_size,
    __global float* hd_distances,
	__global float* metadata) 
{ 
	float spring_force = 0.7f;
	float damping = 0.3f;
	float delta_time = 0.3f;
	float freeness = 0.85f;
	float size_factor = (1.f / ((float)(near_set_size + random_set_size)));
	int mod_op = start_index;
	if(start_index == 0)mod_op = end_index;
	
	int gid = get_global_id(0) + start_index;
	if(gid >= end_index)return;
	__private float my_pivot_indices[24];
	int pivot_size = near_set_size + random_set_size;
	
	for(int i = 0; i < near_set_size; i++)
	{
		my_pivot_indices[i] = pivot_indices[gid * pivot_size + i];
	}
     	
	float pivot_distances_high[24];
	float pivot_distances_low[24];
	float dir_vector[2];
	float rel_velocity[2];

	int seed = seed_memory[gid];

//	int start_index = near_set_size;

	for(int i = near_set_size; i < pivot_size; i++)
	{
		seed = seed + i;
		int random_number = rand(&seed);
		if(random_number < 0) random_number = -random_number;
		if(seed < 0) seed = -seed;
		my_pivot_indices[i] = random_number % mod_op; 
	}

	seed_memory[gid] = seed; 

	for(int i = 0; i < pivot_size; i++)
	{
		float hi = 0.f;
		for( int k = 0; k < n_original_dims; k++ ) {

			float norm = (highD[gid * n_original_dims + k] - highD[(int)my_pivot_indices[i] * n_original_dims + k]);
			hi += norm * norm;
		}
		pivot_distances_high[i] = sqrt((float)hi);
	}

	quickSortIterative( my_pivot_indices, pivot_distances_high, 0, pivot_size - 1);


	// mark duplicates with 1000
	for(int i = 1; i < pivot_size; i++) {
		if((int)my_pivot_indices[i] == (int)my_pivot_indices[i - 1]) {
			pivot_distances_high[i] = 1000.f;
		}
	}

	// TODO: sort pivot_distances and pivot_indices
	quickSortIterative( pivot_distances_high, my_pivot_indices, 0, pivot_size - 1);

	for(int i = 0; i < pivot_size; i++)
	{
		hd_distances[gid * pivot_size + i] = pivot_distances_high[i];
	}
	// Move the point
	for(int i = 0; i < pivot_size; i++) {
		int idx = (int)my_pivot_indices[i];
		float norm = 0.f;
		for(int k = 0; k < n_projection_dims; k++) {
			dir_vector[k] = lowD[idx * n_projection_dims + k] - lowD[gid * n_projection_dims + k];
			norm += dir_vector[k] * dir_vector[k];
		}
		
		norm = sqrt(norm);
		pivot_distances_low[i] = norm;

		if(norm > 1.e-6 && pivot_distances_high[i] != 1000.f)
		{
			for(int k = 0; k < n_projection_dims; k++) {
				dir_vector[k] /= norm;
			}
		
	
			// relative velocity
			for(int k = 0; k < n_projection_dims; k++) {
				rel_velocity[k] = velocity[idx * n_projection_dims + k] - velocity[gid * n_projection_dims + k];
			}

			// calculate difference
			float delta_distance = (pivot_distances_low[i] - pivot_distances_high[i]) * spring_force;
			// compute damping value
			norm = 0.f;
			for(int k = 0; k < n_projection_dims; k++) {
				norm += dir_vector[k] * rel_velocity[k];
			}
			delta_distance += norm * damping;
			
			// accumulate the force
			for(int k = 0; k < n_projection_dims; k++) {
	            		force[gid * n_projection_dims + k] += dir_vector[k] * delta_distance;
			}
		}	
	}

	// scale the force by size factor
	for(int k = 0; k < n_projection_dims; k++) {
		force[gid * n_projection_dims + k] *= size_factor;
	}

	for(int i = 0; i < pivot_size; i++)
	{
		pivot_indices[gid * pivot_size + i] = my_pivot_indices[i];
	}
}

__kernel void updatePosition(
                        __global float* velocity,
                        __global float* lowD,
                        __global float* force,
						const int start_index,
						const int end_index,
                        const int n_projection_dims,
                        const float delta_time,
                        const float freeness
)
{
    
    int gid = get_global_id(0) + start_index;
    if(gid >= end_index)return;
    // update new velocity
    // v = v0 + at
    for(int k = 0; k < n_projection_dims; k++) {
        float v0 = velocity[gid * n_projection_dims + k];
        float v = v0 + force[gid * n_projection_dims + k] * delta_time;
        v *= freeness;
        velocity[gid * n_projection_dims + k] = max(min(v, 2.f), -2.f);

    }
    
     // update new positions
     // x = x0 + vt
     for(int k = 0; k < n_projection_dims; k++) {
         lowD[gid * n_projection_dims + k] += velocity[gid * n_projection_dims + k] * delta_time;
     }
     //barrier(CLK_GLOBAL_MEM_FENCE);

}

__kernel void computeLdDistance(
                        __global float* lowD,
						__global float* pivot_indices,
						__global float* ld_distances,
						const int start_index,
						const int end_index,
                        const int n_projection_dims,
						const int near_set_size,
						const int random_set_size
)
{    
    int gid = get_global_id(0) + start_index;
    if(gid >= end_index)return;
	int pivot_size = near_set_size + random_set_size;
	float dir_vector[2];
	// compute ld_distances
	for(int i = 0; i < pivot_size; i++)
	{
		int idx = (int)pivot_indices[gid * pivot_size + i];
		float norm = 0.f;
		for(int k = 0; k < n_projection_dims; k++) {
			dir_vector[k] = lowD[idx * n_projection_dims + k] - lowD[gid * n_projection_dims + k];
			norm += dir_vector[k] * dir_vector[k];
		}
		ld_distances[gid * pivot_size + i] = sqrt(norm);
	}
}

__kernel
void computeStress(__global float* hd_distances,
		__global float* ld_distances,		  
            __local float* scratchN,
	    __local float* scratchD,
            __const int length,
            __global float* resultN,
	    __global float* resultD) {

  int global_index = get_global_id(0);
  float numerator = 0.f;
  float denominator = 0.f;
  
  // Loop sequentially over chunks of input vector
  while (global_index < length) {
    float hd = hd_distances[global_index] == 1000.f ? 0.f : hd_distances[global_index];
    float ld = ld_distances[global_index];
	if(isinf(hd) || isnan(ld))hd = 0.f;
	if(isinf(ld) || isnan(ld))ld = 0.f;
    float tempN = hd - ld;
    numerator += (tempN * tempN);
    //float tempD = hd_distances[global_index];
    denominator += (hd * hd);
    global_index += get_global_size(0);
  }

  // Perform parallel reduction
  int local_index = get_local_id(0);
  scratchN[local_index] = numerator;
  scratchD[local_index] = denominator;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset = offset / 2) {
    if (local_index < offset) {
      float mineN = scratchN[local_index];
      float mineD = scratchD[local_index];
      float otherN = scratchN[local_index + offset];
      float otherD = scratchD[local_index + offset];
if(isinf(mineN) || isnan(mineN))mineN = 0.f;
if(isinf(mineD) || isnan(mineD))mineD = 0.f;
if(isinf(otherN) || isnan(otherN))otherN = 0.f;
if(isinf(otherD) || isnan(otherD))otherD = 0.f;

      scratchN[local_index] = mineN + otherN;
      scratchD[local_index] = mineD + otherD;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
	if(isinf(scratchN[0]) || isnan(scratchN[0]))scratchN[0] = 0.f;
if(isinf(scratchD[0]) || isnan(scratchD[0]))scratchD[0] = 0.f;
    resultN[get_group_id(0)] = scratchN[0];
    resultD[get_group_id(0)] = scratchD[0];
  }
}


int rand(int* seed) // 1 <= *seed < m
{
    long const a = 16807; //ie 7**5
    long const m = 2147483647; //ie 2**31-1
    int s = *seed;	
    long temp  = (s * a) % m;
    *seed = temp;
    return(*seed);
}

// A utility function to swap two elements
void swap ( __private float* a, __private float* b )
{
    float t = *a;
    *a = *b;
    *b = t;
}


/* This function is same in both iterative and recursive*/
int partition (__private float array[], __private float array1[], int l, int h)
{
    float x = array[h];
    int i = (l - 1);

    for (int j = l; j <= h- 1; j++)
    {
        if (array[j] <= x)
        {
            i++;
            swap (&array[i], &array[j]);
            swap (&array1[i], &array1[j]);
        }
    }
    swap (&array[i + 1], &array[h]);
    swap (&array1[i + 1], &array1[h]);
    return (i + 1);
}




/* A[] --> Array to be sorted,
   l  --> Starting index,
   h  --> Ending index */
void quickSortIterative (__private float array[], __private float array1[], int h, int l)
{
    // Create an auxiliary stack
    int stack[24];

    // initialize top of stack
    int top = -1;

    // push initial values of l and h to stack
    stack[ ++top ] = l;
    stack[ ++top ] = h;

    // Keep popping from stack while is not empty
    while ( top >= 0 )
    {
        // Pop h and l
        h = stack[ top-- ];
        l = stack[ top-- ];

        // Set pivot element at its correct position
        // in sorted array
        int p = partition( array, array1, l, h );

        // If there are elements on left side of pivot,
        // then push left side to stack
        if ( p-1 > l )
        {
            stack[ ++top ] = l;
            stack[ ++top ] = p - 1;
        }

        // If there are elements on right side of pivot,
        // then push right side to stack
        if ( p+1 < h )
        {
            stack[ ++top ] = p + 1;
            stack[ ++top ] = h;
        }
    }
}

