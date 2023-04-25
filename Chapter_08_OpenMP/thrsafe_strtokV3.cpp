/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : July 2020
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ -fopenmp thrsafe_strtokV3.cpp -o thrsafe_strtokV3
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <set>
#include <mutex>
#include <omp.h>
#include <unistd.h>

using namespace std;

char *strtokV3 (char *s, const char *delim, char **aux)
{
  static set < char *>inProc;
  static mutex l;

  // add s to the set of strings being processed or return NULL on failure
  unique_lock < mutex > ul (l);
  if (*aux == NULL && inProc.insert (s).second == false)
    return NULL;
  ul.unlock ();

  int idx1 = 0, idx2 = -1;
  char needle[2] = { 0 };
  char *token = NULL;
  int i;
  char *temp = s;
  if (*aux != NULL)
    {
      temp = *aux;
      s[0] = 0;
    }

  // iterate over all characters of the input string
  for (i = 0; temp[i]; i++)
    {
      // printf("%i %i %c\n", i, omp_get_thread_num (), temp[i]);
      needle[0] = temp[i];
      // check if a character matches a delimiter
      if (strstr (delim, needle) != NULL)       // strstr is reentrant
        {
          idx1 = idx2 + 1;      // get the index boundaries of the token
          idx2 = i;
          if (idx1 != idx2)     // is it a token or a delimiter following another?
            {
              temp[i] = 0;
              *aux = temp + i + 1;
              token = temp + idx1;
              break;
            }
        }
    }

  // repeat checks for the token preceding the end of the string
  if (!token)
    {
      idx1 = idx2 + 1;
      idx2 = i;
      if (idx1 != idx2)
        {
          *aux = temp + i;
          token = temp + idx1;
        }
    }
  ul.lock ();
  if (token == NULL)
    inProc.erase (s);

  return token;
}

//---------------------------------------------
void threadSafeParse (char *s, const char *delim)
{
  char *state = NULL;
  char *tok;

  tok = strtokV3 (s, delim, &state);
  while (tok)
    {
      printf ("Thread %i : %s\n", omp_get_thread_num (), tok);
      usleep (1);
      tok = strtokV3 (s, delim, &state);
    }
}

//---------------------------------------------
int main (int argc, char *argv[])
{
  if (argc != 4)
    {
      fprintf (stderr, "Usage: %s string1 string2 delim\n", argv[0]);
      exit (EXIT_FAILURE);
    }
  char *str1 = argv[1], *str2 = argv[2], *delim = argv[3];

#pragma omp parallel
  {
#pragma omp taskgroup
#pragma omp single
    {
      for (int i = 0; i < 100; i++)
        if (i % 2)
#pragma omp task
          threadSafeParse (str1, delim);
        else
#pragma omp task
          threadSafeParse (str2, delim);

    }
  }

  exit (EXIT_SUCCESS);
}
