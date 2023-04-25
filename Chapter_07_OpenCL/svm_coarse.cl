kernel void initKernel (global int *data, int N)
{
  int ID = get_global_id (0);
  if(ID<N)
     data[ID] = ID;
}

