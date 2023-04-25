// const sampler_t s = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

kernel void imageConv (read_only image2d_t in, int w, int h, constant float *filter, int fdim, write_only image2d_t out, sampler_t s)
{
  int xID = get_global_id (0);
  int yID = get_global_id (1);
  // check if work item is corresponding to an actual pixel
  if (xID < w && yID < h)
    {
      int filtCenter = fdim / 2;
      int filtIdx = 0;
      float newPixel = 0;
      int2 coord = { xID, yID };
      for (int yOff = -filtCenter; yOff <= filtCenter; yOff++)
        {
          coord.y = yID + yOff;
          coord.x = xID - filtCenter;
          for (int xOff = -filtCenter; xOff <= filtCenter; xOff++)
            {
              int4 oldP = (int4) read_imagei (in, s, coord);
              newPixel += oldP.x * filter[filtIdx];
              filtIdx++;        // works only if the inner loop is for x-axis iteration           
              coord.x++;
            }
        }

      coord.x = xID;
      coord.y = yID;
      int4 newP = { (int) newPixel, 0, 0, 0 };
      write_imagei (out, coord, newP);
    }
}
