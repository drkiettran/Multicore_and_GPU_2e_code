#ifndef PGM_H
#define PGM_H

class PGMImage
{
 public:
   PGMImage(char *);
   PGMImage(int x=100, int y=100, int col=16);
   ~PGMImage();
   bool write(char *);
		   
   int x_dim;
   int y_dim;
   int num_colors;
   unsigned char *pixels;
};

#endif
