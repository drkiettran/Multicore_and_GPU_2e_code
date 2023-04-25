//***********************************************************
// CPU based calculation
// The probabilities must be calculated before hand
float mutual (unsigned char *img, unsigned char *img2, int numPixels, int levels, float *pi, float *pj, float *pij)
{
  float mutual = 0;
  for (int i = 0; i < levels; i++)
    for (int j = 0; j < levels; j++)
      {
        int idx = i * levels + j;
        if (pi[i] != 0 && pj[j] != 0 && pij[idx] != 0)
          mutual += pij[idx] * log (pij[idx] / (pi[i] * pj[j]));
      }
  return mutual / log (2);
}

//***********************************************************
// Calculates D1 and D2 concurrently
void Diff (float *pij, float *diff, int levels, float *d1, float *d2)
{
  *d1 = *d2 = 0;
  for (int i = 0; i < levels; i++)
    for (int j = 0; j < levels; j++)
      {
        int idx = i * levels + j;
        float p = pij[idx];
        float temp = diff[idx] * p;
        float temp2 = temp * diff[idx];
        *d1 += temp;
        *d2 += temp2;
      }
}

//***********************************************************
void readImage (Image & img, char *fname)
{
  FILE *fin;
  char buffer[1000];
  fin = fopen (fname, "rb");

//   fscanf (fin, "%*s%i%i%i", &(img.width), &(img.height), &(img.levels));
  fscanf (fin, "%*s%*c");
  fgets(buffer, 1000, fin);
  if(buffer[0] == '#')
     fscanf(fin, "%i%i%i", &(img.width), &(img.height), &(img.levels));
  else
    {
      sscanf(buffer, "%i%i", &(img.width), &(img.height));
      fscanf(fin,"%i", &(img.levels));
    }

  assert (MAXLEVELS >= img.levels);
  assert (MAXPIXELMEM >= img.width * img.height);

  memset ((void *) img.pixel, 0, img.width * img.height * sizeof (unsigned char));
  if (img.levels < 256)
    for (int i = 0; i < img.width * img.height; i++)
      fread ((void *) &(img.pixel[i]), sizeof (unsigned char), 1, fin);
  else if (img.levels < 65536)
    for (int i = 0; i < img.width * img.height; i++)
      fread ((void *) &(img.pixel[i]), sizeof (unsigned short), 1, fin);
  else
    fread (img.pixel, sizeof (unsigned int), img.width * img.height, fin);
  img.levels++;
  fclose (fin);

//cout << &(img.pixel[0])- &(img.pixel[1])<< endl;
}

//***********************************************************
void CalcProb (unsigned char *img, int numPixels, int levels, float *p, ImgRegResults *res)
{
  memset ((void *) p, 0, levels * sizeof (float));
  for (int i = 0; i < numPixels; i++)
    p[img[i]]++;

  for (int i = 0; i < levels; i++)
    p[i] /= numPixels;


  float x2=0, avg=0;
  for (int i = 0; i < levels; i++)
    {
      float temp = i*p[i]; 
      avg += temp;
      x2 += temp*i;
    }
  res -> x2 = x2;
  res -> avg = avg;
}

//***********************************************************
void CalcJointProb (unsigned char *img, unsigned char *img2, int numPixels, int levels, float *pij)
{
  memset ((void *) pij, 0, levels * levels * sizeof (float));
  for (int i = 0; i < numPixels; i++)
    pij[img[i] * levels + img2[i]]++;

  for (int i = 0; i < levels * levels; i++)
    pij[i] /= numPixels;
}

//***********************************************************
// OBSOLETE: Calculation is incorpotated in CalcProb
void Variance (unsigned char *a, int numPixels, float *avg, float *avg_x2)
{
  *avg = *avg_x2 = 0;
  for (int i = 0; i < numPixels; i++)
    {
      unsigned int x = a[i];
      *avg += x;
      *avg_x2 += (1.0 * x) * x;
    }
  *avg /= (1.0 * numPixels);
  *avg_x2 /= (1.0 * numPixels);
}

//***********************************************************
// A customized Covariance, where the average and mean x2 of one image are passed as parameters
float Covariance (unsigned char *a, unsigned char *b, int numPixels, float avga, float x2)
{
  float avgb, y2, xy;

  avgb = 0;
  y2 = 0;
  for (int i = 0; i < numPixels; i++)
    {
      avgb += b[i];
      y2 += (1.0 * b[i]) * b[i];
    }
  avgb /= (1.0 * numPixels);
  y2 /= (1.0 * numPixels);

//printf("---      %f %f %f %f \n", avga, x2, avgb, y2);

  xy = 0;
  for (int i = 0; i < numPixels; i++)
    xy += (1.0 * a[i]) * b[i];
  xy /= numPixels;

  //return covab*covab / (covaa*covbb);
  return (xy - avga * avgb) / sqrt ((x2 - avga * avga) * (y2 - avgb * avgb));
}

