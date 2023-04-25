/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : Generic semaphore implementation based on condition variables
 To compile    : 
 ============================================================================
 */
#include "semaphore.h"

//---------------------------------
semaphore::semaphore (int i)
{
  value = i;
}

//---------------------------------
void semaphore::acquire ()
{
  std::unique_lock < std::mutex > ul (l);
  value--;

  if (value < 0)
    {
      block.wait (ul);
    }
}

//---------------------------------
void semaphore::acquire (unsigned int i)
{
  std::unique_lock < std::mutex > ul (l);
  block.wait( ul, [&](){ return value >= i;} );
  value -= i;
}
// void semaphore::acquire (unsigned int i)
// {
//   while (i--)
//     this->acquire ();
// }

//---------------------------------
void semaphore::release (unsigned int i)
{
  std::lock_guard < std::mutex > guard (l);
  while (i--)
    {
      value++;
      block.notify_one ();
    }
}

//---------------------------------
int semaphore::available ()
{
  std::lock_guard < std::mutex > guard (l);
  return value;
}

//---------------------------------
bool semaphore::try_acquire (unsigned int i)
{
  std::lock_guard < std::mutex > guard (l);
  bool res = false;
  if (value >= (int) i)
    {
      value -= i;
      res = true;
    }
  return res;
}
