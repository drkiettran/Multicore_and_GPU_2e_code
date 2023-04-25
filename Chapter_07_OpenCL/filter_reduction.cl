inline bool predicate(int i)
{
    return i % 2;
}

kernel void countOdds(global int *data, int N, global int *groupCount)
{
  int ID = get_global_id (0);
  int haveOdd = predicate(data[ID]); // no check for ID vs N, implies OpenCL 2.0
  
  int totalForThisGroup = work_group_reduce_add(haveOdd);
  if(get_local_id(0)==0)
      groupCount[get_group_id(0)+1] = totalForThisGroup;
}


kernel void moveOdds(global int *src, int N, global int *dest, global int *groupOffsets)
{
  int ID = get_global_id (0);
  int haveOdd = predicate(src[ID]); // no check for ID vs N, implies OpenCL 2.0
  int localOffset = work_group_scan_exclusive_add(haveOdd); 
  
  if(haveOdd)
      dest[ localOffset + groupOffsets[get_group_id(0)] ] = src[ID];
}
