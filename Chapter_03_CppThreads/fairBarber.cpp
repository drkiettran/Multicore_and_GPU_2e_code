/*
 ============================================================================
 Author        : G. Barlas
 Version       : 2.0
 Last modified : August 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake fairBarber.pro; make
 ============================================================================
 */
#include <thread>
#include <mutex>
#include <iostream>
#include <unistd.h>
#include <memory>
#include <stdio.h>
#include <vector>
#include "semaphore.h"
#include <boost/make_shared.hpp>

using namespace std;

const int NUMCHAIRS=10;
//---------------------------------------

void concurPrint(int cID, int bID) {
    static mutex l;
    l.lock();
    cout << "Customer " << cID << " is being serviced by barber " << bID << endl;
    l.unlock();
}
//---------------------------------------

class Barber {
private:
    int ID;
    static semaphore *barberReady;
//     static boost::shared_ptr<semaphore[]> customerReady;
    static boost::shared_ptr<semaphore[]> customerReady;
    static boost::shared_ptr<semaphore[]> barberDone;

    static mutex l1;
    static semaphore customersLeft;
    static boost::shared_ptr<int[]> buffer;
    static int in;
    static int numBarbers;
public:
    static void initClass(int numB, int numC, semaphore *r, boost::shared_ptr<semaphore[]> &c, boost::shared_ptr<semaphore[]> &d, boost::shared_ptr<int[]> &b);

    Barber(int i) : ID(i) {}
    void operator()();
};
//---------------------------------------
semaphore *Barber::barberReady;
boost::shared_ptr<semaphore[]> Barber::customerReady;
boost::shared_ptr<semaphore[]> Barber::barberDone;
mutex Barber::l1;
semaphore Barber::customersLeft;
boost::shared_ptr<int[]> Barber::buffer;
int Barber::in = 0;
int Barber::numBarbers;
//---------------------------------------

void Barber::initClass(int numB, int numC, semaphore *r, boost::shared_ptr<semaphore[]> &c,  boost::shared_ptr<semaphore[]> &d, boost::shared_ptr<int[]> &b) {
    customersLeft.release(numC);
    barberReady = r;
    customerReady = c;
    barberDone = d;
    buffer = b;
    numBarbers = numB;
}
//---------------------------------------  

void Barber::operator()() {
    while (customersLeft.try_acquire()) {
        l1.lock();
        buffer[in] = ID;
        in = (in + 1) % numBarbers;
        l1.unlock();
        barberReady->release(); // signal availability
        customerReady[ID].acquire(); // wait for customer to be sitted
        barberDone[ID].release(); // signal that hair is done
    }
}
//---------------------------------------  

class Customer {
private:
    int ID;
    static semaphore *barberReady;
    static boost::shared_ptr<semaphore[]> customerReady;
    static boost::shared_ptr<semaphore[]> barberDone;
    static semaphore waitChair;
    static semaphore barberChair;
    static mutex l2;
    static boost::shared_ptr<int[]> buffer;
    static int out;
    static int numBarbers;
    static semaphore numProducts;
public:
    static void initClass(int numB, semaphore *r, boost::shared_ptr<semaphore[]> &c, boost::shared_ptr<semaphore[]> &d, boost::shared_ptr<int[]> &b);

    Customer(int i) : ID(i) {}
    void operator()();
};
//---------------------------------------
semaphore *Customer::barberReady;
boost::shared_ptr<semaphore[]> Customer::customerReady;
boost::shared_ptr<semaphore[]> Customer::barberDone;
semaphore Customer::waitChair(NUMCHAIRS);
semaphore Customer::barberChair;
mutex Customer::l2;
boost::shared_ptr<int[]> Customer::buffer;
int Customer::out = 0;
int Customer::numBarbers;

//---------------------------------------

void Customer::initClass(int numB, semaphore *r, boost::shared_ptr<semaphore[]> &c, boost::shared_ptr<semaphore[]> &d, boost::shared_ptr<int[]> &b) {
    barberReady = r;
    customerReady=c;
    barberDone = d;
    buffer = b;
    numBarbers = numB;
    barberChair.release(numB);
}
//---------------------------------------

void Customer::operator()() {
    waitChair.acquire(); // wait for a chair
    barberReady->acquire(); // wait for a barber to be ready
    l2.lock();
    int bID = buffer[out];
    out = (out + 1) % numBarbers;
    l2.unlock();
    waitChair.release(); // get up from the chair
    barberChair.acquire(); // wait for an available barber chair
    customerReady[bID].release(); // signal that customer is ready
    concurPrint(ID, bID);
    barberDone[bID].acquire(); // wait for barber to finish haircut
    barberChair.release(); // get up from barber's chair
}
//---------------------------------------

int main(int argc, char *argv[]) {
    if (argc == 1) {
        cerr << "Usage " << argv[0] << " #barbers #customers\n";
        exit(1);
    }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    boost::shared_ptr<int[]> buffer(new int[N]);

    semaphore barberReady;
    boost::shared_ptr<semaphore[]> customerReady (new semaphore[N]);
    boost::shared_ptr<semaphore[]> barberDone (new semaphore[N]);

    Barber::initClass(N, M, &barberReady, customerReady, barberDone, buffer);
    Customer::initClass(N, &barberReady, customerReady, barberDone, buffer);

    unique_ptr<thread > thr[N+M]; 

    shared_ptr<Barber> b[N];
    shared_ptr<Customer> c[M];
    for (int i = 0; i < N; i++) {
        b[i] = make_shared<Barber>(i);
        thr[i] = make_unique<thread>(ref(*b[i]));
    }
    for (int i = 0; i < M; i++) {
        c[i] = make_shared<Customer>(i);
        thr[i+N] = make_unique<thread>(ref(*c[i]));
    }

    for (int i = 0; i < N+M; i++)
        thr[i]->join();

    return 0;
}
