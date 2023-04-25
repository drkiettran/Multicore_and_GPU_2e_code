kernel void hello()
{                     
   size_t ID = get_global_id(0); 
   printf("Work item %i says hello!\n", ID); 
}
