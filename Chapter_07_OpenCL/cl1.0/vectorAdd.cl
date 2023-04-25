__kernel void vecAdd(__global int *a, __global int *b, __global int *c)
{                     
   size_t ID = get_global_id(0);
   printf("%i\n",c[ID]);
   if(ID < get_global_size(0))
      c[ID] = a[ID] + b[ID];
}
