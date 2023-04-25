/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0.2
 Last modified : November 2019
 License       : Released under the GNU GPL 3.0
 Description   : OpenMP+MPI Solution. Has to be compiled with -fopenmp
                 Decryption key is 107,481,429
 To build use  : mpic++ -fopenmp decrypt2_omp.cpp -o decrypt2_omp
 ============================================================================
 */
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <rpc/des_crypt.h>
#include <mpi.h>
#include <omp.h>

int BLOCK = 1000000;

unsigned char cipher[] = {251, 37, 207, 196, 160, 168, 80, 50, 26, 17, 17, 62, 244, 249, 150, 70, 175, 116, 11, 115, 159, 5, 164, 229, 149, 101, 172, 233, 244, 20, 86, 136, 70, 130, 170, 94, 205, 228, 149, 182, 15, 243, 245, 55, 26, 33, 67, 88, 216, 164, 81, 109, 107, 32, 236, 24, 220, 162, 199, 88, 240, 34, 247, 151, 132, 253, 31, 42, 202, 77, 42, 108, 27, 131, 101, 160, 82, 164, 181, 170, 0};

char search[] = " live ";
//----------------------------------------------------------
void decrypt (long key, char *ciph, int len)
{
  // prepare key for the parity calculation. Least significant bit in all bytes should be empty
  long k = 0;
  for (int i = 0; i < 8; i++)
    {
      key <<= 1;
      k += (key & (0xFE << i * 8));
    }

  // Decrypt ciphertext
  des_setparity ((char *) &k);
  ecb_crypt ((char *) &k, (char *) ciph, len, DES_DECRYPT);
}

//----------------------------------------------------------
// Returns true if the plaintext produced containes the "search" string
bool tryKey (long key, char *ciph, int len)
{
  char temp[len + 1];
  memcpy (temp, ciph, len);
  temp[len] = 0;
  decrypt (key, temp, len);
  return strstr ((char *) temp, search) != NULL;
}

//----------------------------------------------------------
int main (int argc, char **argv)
{
  int N, id;
  long upper = (1L << 56);
  long found = -1;
  MPI_Status st;
  MPI_Request req;
  int flag = 0;
  int ciphLen = strlen ((char *) cipher);

  BLOCK = atoi(argv[1]);

  MPI_Init (&argc, &argv);
  double start = MPI_Wtime ();
  MPI_Comm_rank (MPI_COMM_WORLD, &id);
  MPI_Comm_size (MPI_COMM_WORLD, &N);
  MPI_Irecv ((void *) &found, 1, MPI_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &req);
  int iterCount = 0;

  long idx = 0;
  while (idx < upper && found < 0)
    {
        
#pragma omp parallel for default(none) shared(cipher, ciphLen, found, idx, id, N, BLOCK)
      for (long i = idx + id; i < idx + N * BLOCK; i += N)
        {
          if (tryKey (i, (char *) cipher, ciphLen))
            {
#pragma omp critical
              found = i;
            }
        }

      if (found >= 0)
        {
          for (int node = 0; node < N; node++)
            MPI_Send ((void *) &found, 1, MPI_LONG, node, 0, MPI_COMM_WORLD);
        }

      idx += N * BLOCK;
    }

  if (id == 0)
    {
      MPI_Wait (&req, &st);     // in case process 0 finishes before the key is found     
      decrypt (found, (char *) cipher, ciphLen);
      printf ("%i nodes in %lf sec : %li %s\n", N, MPI_Wtime () - start, found, cipher);
    }
  MPI_Finalize ();
}
