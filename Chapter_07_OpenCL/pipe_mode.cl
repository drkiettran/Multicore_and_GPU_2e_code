kernel void freqCount (global int *data, int N, write_only pipe int2 countsPipe, int st, int end, local int *counts)
{
  int ID = get_global_id (0);
  int totalWorkItems = get_global_size (0);
  int localID = get_local_id (0);
  int groupSize = get_local_size (0);

  // initialize local counts
  int totalCounts = end - st + 1;
  for (int i = localID; i < totalCounts; i += groupSize)
    counts[i] = 0;
  work_group_barrier (CLK_LOCAL_MEM_FENCE);

  for (int i = ID; i < N; i += totalWorkItems)
    {
      int v = data[i] - st;
      atom_add (counts + v, 1);
    }

  // wait for all counts to be done
  work_group_barrier (CLK_LOCAL_MEM_FENCE);

  for (int i = localID; i < totalCounts; i += groupSize)
    {
      if (counts[i] > 0)
        {
          int2 pair = { i, counts[i] };
          write_pipe (countsPipe, &pair);
        }
    }
}

//====================================================================
// Assumes a 256-work item group
kernel void modeFind (read_only pipe int2 countsPipe, global int *res, int N, int st, int end, local int *counts)
{
  int localID = get_local_id (0);
  int groupSize = get_local_size (0);   // this should be 256
  local int bestIdx[256];
  local int bestCount[256];
  local int total;
  if (localID == 0)
    total = 0;

  // initialize local counts
  int totalCounts = end - st + 1;
  for (int i = localID; i < totalCounts; i += groupSize)
    counts[i] = 0;

  work_group_barrier (CLK_LOCAL_MEM_FENCE);

  // start collecting partial results from the pipe
  int2 pipeItem;
  while (total != N)
    {
      int res = read_pipe (countsPipe, &pipeItem);
      if (res == 0)
        {
          atomic_add (counts + pipeItem.x, pipeItem.y);
          atomic_add (&total, pipeItem.y);
        }
    }

  // find the mode
  if (localID < totalCounts)
    {
      bestIdx[localID] = localID;
      bestCount[localID] = counts[localID];
    }
  else
    {
      bestIdx[localID] = 0;
      bestCount[localID] = 0;
    }

  work_group_barrier (CLK_LOCAL_MEM_FENCE);

  // find local maximum from counts
  for (int i = localID + groupSize; i < totalCounts; i += groupSize)
    if (bestCount[localID] < counts[i])
      {
        bestIdx[localID] = i;
        bestCount[localID] = counts[i];
      }

  work_group_barrier (CLK_LOCAL_MEM_FENCE);

  // reduce mode
  int step = 1;
  bool cont = true;
  while (step < groupSize)
    {
      int otherID = localID | step;
      if (localID & step != 0)
        cont = false;
      if (cont && otherID < groupSize)
        {
          if (bestCount[localID] < bestCount[otherID])
            {
              bestIdx[localID] = bestIdx[otherID];
              bestCount[localID] = bestCount[otherID];
            }
        }
      step *= 2;

      work_group_barrier (CLK_LOCAL_MEM_FENCE);
    }

  if (localID == 0)
    {
      res[0] = bestIdx[0] + st;
      res[1] = bestCount[0];
    }
}
