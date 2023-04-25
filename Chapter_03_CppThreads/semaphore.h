#include <mutex>
#include <condition_variable>

class semaphore
{
private:
  int value = 0;
    std::mutex l;
    std::condition_variable block;

public:
    semaphore (int i = 0);
    semaphore (const semaphore &) = delete;
    semaphore (const semaphore &&) = delete;
    semaphore & operator= (const semaphore &) = delete;
    semaphore & operator= (const semaphore &&) = delete;
    void acquire ();
    void acquire (unsigned int i);
    void release (unsigned int i = 1);
    int available ();
    bool try_acquire (unsigned int i = 1);
};
