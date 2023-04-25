/*
 ============================================================================
 Author        : G. Barlas
 Version       : 2.0
 Last modified : November 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : qmake; make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <memory>
#include <thread>
#include <mpi.h>
#include <QImage>
#include <QRgb>
#include <sharedQue.h>
using namespace std;

//************************************************************
// Communication tags
#define CORESAVAILTAG 0
#define RESULTTAG     1
#define WORKITEMTAG   2
#define ENDTAG        3

//************************************************************
class MandelWorkItem
{
public:
    double upperX, upperY, lowerX, lowerY;
    int imageX, imageY, pixelsX, pixelsY;
    static MPI_Datatype type;

    static void initType();
};
MPI_Datatype MandelWorkItem::type;

//************************************************************
void MandelWorkItem::initType()
{
    MandelWorkItem sample;

    int blklen[2];
    MPI_Aint displ[2], off, base;
    MPI_Datatype types[2];

    blklen[0] = 4;
    blklen[1] = 4;

    types[0] = MPI_DOUBLE;
    types[1] = MPI_INT;

    displ[0] = 0;
    MPI_Get_address (&(sample.upperX), &base);
    MPI_Get_address (&(sample.imageX), &off);
    displ[1] = off - base;

    MPI_Type_create_struct (2, blklen, displ, types, &type);
    MPI_Type_commit (&type);
}
//************************************************************
class MandelResult
{
public:  // public access is not recommended but it is used here to shorten the code
    unique_ptr<int[]> imgPart=nullptr;
    // the needed imageX and imageY parameters, i.e. the location of computed image in bigger picture, are placed at the end of the pixel buffer
    static int numItems;       // not part of the communicated data
    static MPI_Datatype type;

    MandelResult(){};
    MandelResult(int , int );
    void init(int , int );
    int getResultSize() {return 1;}
    static void initType(int blkSize);
};
MPI_Datatype MandelResult::type;
int MandelResult::numItems;
//-------------------------------------------------------
MandelResult::MandelResult(int iX, int iY)
{
    imgPart = make_unique<int[]>(numItems+2); // +2 to hold imageX and imageY
    imgPart[numItems] = iX;
    imgPart[numItems+1] = iY;
}
//-------------------------------------------------------
void MandelResult::init(int iX, int iY)
{
    imgPart = make_unique<int[]>(numItems+2); // +2 to hold imageX and imageY
    imgPart[numItems] = iX;
    imgPart[numItems+1] = iY;
}
//-------------------------------------------------------
void MandelResult::initType(int s)
{
    numItems = s;
    int blklen;
    MPI_Aint displ;
    MPI_Datatype types;

    blklen = numItems + 2;

    types = MPI_INT;

    displ = 0;

    MPI_Type_create_struct (1, &blklen, &displ, &types, &type);
    MPI_Type_commit (&type);
}

//************************************************************
// Class for computing a fractal set part
class MandelCompute
{
private:
    double upperX, upperY, lowerX, lowerY;
    int pixelsX, pixelsY, imageX, imageY;
    MandelResult *res;

    static int MAXITER;
    int diverge (double cx, double cy);

public:
    void compute();
    void init(MandelWorkItem *, MandelResult *);
    MandelResult* getResult();
};
int MandelCompute::MAXITER = 255;

//--------------------------------------

int MandelCompute::diverge (double cx, double cy)
{
    int iter = 0;
    double vx = cx, vy = cy, tx, ty;
    while (iter < MAXITER && (vx * vx + vy * vy) < 4)
    {
        tx = vx * vx - vy * vy + cx;
        ty = 2 * vx * vy + cy;
        vx = tx;
        vy = ty;
        iter++;
    }
    return iter;
}

//--------------------------------------
void MandelCompute::init (MandelWorkItem* wi, MandelResult *r)
{
    upperX = wi->upperX;   //local copies are used to speed up computation
    upperY = wi->upperY;
    lowerX = wi->lowerX;
    lowerY = wi->lowerY;
    imageX = wi->imageX;
    imageY = wi->imageY;

    res = r;

    pixelsX = wi->pixelsX;
    pixelsY = wi->pixelsY;
}

//--------------------------------------

void MandelCompute::compute ()
{
    double stepx = (lowerX - upperX) / pixelsX;
    double stepy = (upperY - lowerY) / pixelsY;

    int *img = res->imgPart.get();  // shortcut
    for (int i = 0; i < pixelsX; i++)
        for (int j = 0; j < pixelsY; j++)
        {
            double tempx, tempy;
            tempx = upperX + i * stepx;
            tempy = upperY - j * stepy;
            img[j * pixelsX + i] = diverge (tempx, tempy);
        }
    img[pixelsX * pixelsY] = imageX;
    img[pixelsX * pixelsY + 1] = imageY;
}
//--------------------------------------
MandelResult*  MandelCompute::getResult()
{
    return this->res;
}

//************************************************************
// Function object for use by the threads
class WorkerThread 
{
private:
    int ID;
    int runs;
    shared_ptr<QueueMonitor<MandelWorkItem *> > in;
    shared_ptr< QueueMonitor<MandelResult *> > out;
//     QueueMonitor<MandelWorkItem *> *in;
//     QueueMonitor<MandelResult *> *out;

public:
    WorkerThread(int i, shared_ptr<QueueMonitor<MandelWorkItem *> > &, shared_ptr< QueueMonitor<MandelResult *> > &);
    void operator()();
};
//--------------------------------------
WorkerThread::WorkerThread(int id, shared_ptr<QueueMonitor<MandelWorkItem *> > &i, shared_ptr< QueueMonitor<MandelResult *> > &o)
{
    ID = id;
    in = i;
    out = o;
    runs=0;
}

//--------------------------------------
void WorkerThread::operator()()
{
    unique_ptr<MandelCompute> c = make_unique<MandelCompute>();
    while(1)
    {
        MandelWorkItem * work = in->deque();
        if(work==NULL) break;

        MandelResult *res = out->reserve();
        c->init (work, res);
        in->release(work);

        c->compute ();

        out->enque(res);
    }
    out->enque(NULL);
}
//************************************************************
// Uses the divergence iterations to pseudocolor the fractal set
void savePixels (QImage * img, int *imgPart, int imageX, int imageY, int height, int width)
{
    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++)
        {
            int color = imgPart[j * width + i];
            img->setPixel (imageX + i, imageY + j, qRgb (256 - color, 256 - color, 256 - color));
        }
}

//************************************************************
int main (int argc, char *argv[])
{
    int N, rank, numCores;
    double start_time, end_time;
    MPI_Status status;
    MPI_Request request;


    numCores = sysconf(_SC_NPROCESSORS_ONLN);

    // init MPI and check thread support
    int provided;
    MPI_Init_thread (&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    assert(provided >= MPI_THREAD_FUNNELED);
    start_time = MPI_Wtime ();

    MPI_Comm_size (MPI_COMM_WORLD, &N);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    // check that all parameters are supplied
    if (rank == 0  && (argc < 6))
    {
        cerr << argv[0] << " upperCornerX upperCornerY lowerCornerX lowerCornerY workItemPixelsPerSide\n";
        MPI_Abort (MPI_COMM_WORLD, 1);
    }

    int workItemPixelsPerSide = atoi (argv[5]);

    // create samples of the work item and result classes and register their types with MPI
    MandelWorkItem::initType();
    MandelResult::initType(workItemPixelsPerSide * workItemPixelsPerSide);

    if (rank == 0)   // master code
    {
        double upperCornerX, upperCornerY;
        double lowerCornerX, lowerCornerY;
        double partXSpan, partYSpan;
        int Xparts, Yparts;
        int imgX = 4096, imgY = 2160;

        upperCornerX = atof (argv[1]);
        upperCornerY = atof (argv[2]);
        lowerCornerX = atof (argv[3]);
        lowerCornerY = atof (argv[4]);

        // make sure that the image size is evenly divided in work items
        Xparts = (int) ceil (imgX * 1.0 / workItemPixelsPerSide);
        Yparts = (int) ceil (imgY * 1.0 / workItemPixelsPerSide);
        imgX = Xparts * workItemPixelsPerSide;
        imgY = Yparts * workItemPixelsPerSide;

        partXSpan = (lowerCornerX - upperCornerX) / Xparts;
        partYSpan = (upperCornerY - lowerCornerY) / Yparts;
        unique_ptr<QImage> img = make_unique<QImage> (imgX, imgY, QImage::Format_RGB32);

        // prepare the work items in individual structures
        unique_ptr<MandelWorkItem[]> w = make_unique<MandelWorkItem[]>(Xparts * Yparts);
        for (int i = 0; i < Xparts; i++)
            for (int j = 0; j < Yparts; j++)
            {
                int idx = j * Xparts + i;

                w[idx].upperX = upperCornerX + i * partXSpan;
                w[idx].upperY = upperCornerY - j * partYSpan;
                w[idx].lowerX = upperCornerX + (i + 1) * partXSpan;
                w[idx].lowerY = upperCornerY - (j + 1) * partYSpan;

                w[idx].imageX = i * workItemPixelsPerSide;
                w[idx].imageY = j * workItemPixelsPerSide;
                w[idx].pixelsX = workItemPixelsPerSide;
                w[idx].pixelsY = workItemPixelsPerSide;
            }

        // now distribute the work item to the worker nodes
        unique_ptr<int[]> nodeCores = make_unique<int[]>(N);
        unique_ptr<int[]> workItemsAssignedToNode = make_unique<int[]>(N);
        unique_ptr<MandelResult> res = make_unique<MandelResult>(0, 0);
        for (int i = 0; i < Xparts * Yparts; i++)
        {
            MPI_Recv (res->imgPart.get(), res->getResultSize(), MandelResult::type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int workerID = status.MPI_SOURCE;
            int tag = status.MPI_TAG;
            if(tag == CORESAVAILTAG)
            {
                nodeCores[workerID] = res->imgPart[0];
                workItemsAssignedToNode[workerID]=0;
            }
            else if (tag == RESULTTAG)
            {
                workItemsAssignedToNode[workerID]--;
                int idx = res->numItems;
                int imageX = res->imgPart[idx];  // extract location of image part
                int imageY = res->imgPart[idx+1];
                savePixels (img.get(), res->imgPart.get(), imageX, imageY, workItemPixelsPerSide, workItemPixelsPerSide);
            }

            while(workItemsAssignedToNode[workerID] != 2*nodeCores[workerID]  && i != Xparts*Yparts)
            {
                MPI_Isend (&(w[i]), 1, MandelWorkItem::type, workerID, WORKITEMTAG, MPI_COMM_WORLD, &request);
                i++;
                workItemsAssignedToNode[workerID]++;
            }
            i--;
        }

        // now send termination messages
        int busyNodes=0;
        for(int i=1;i<N;i++)
            if(workItemsAssignedToNode[i]!=0) busyNodes++;

        while(busyNodes != 0)
        {
            MPI_Recv (res->imgPart.get(),1,MandelResult::type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            int workerID = status.MPI_SOURCE;
            int tag = status.MPI_TAG;

            if (tag == RESULTTAG)
            {
                int idx = res->numItems;
                int imageX = res->imgPart[idx];
                int imageY = res->imgPart[idx+1];
                savePixels (img.get(), res->imgPart.get(), imageX, imageY, workItemPixelsPerSide, workItemPixelsPerSide);
                workItemsAssignedToNode[workerID]--;
                if(workItemsAssignedToNode[workerID]==0) busyNodes--;
            }
            MPI_Isend (NULL, 0, MandelWorkItem::type, workerID, ENDTAG, MPI_COMM_WORLD, &request);
        }

        img->save ("mandel.png", "PNG", 0); // save the resulting image

        end_time = MPI_Wtime ();
        cout << "Total time : " << end_time - start_time << endl;
    }
    else      // worker code
    {
        MPI_Send (&numCores, 1, MPI_INT, 0, CORESAVAILTAG, MPI_COMM_WORLD);    // publish available cores to master

        unique_ptr<MandelWorkItem[]> w = make_unique<MandelWorkItem[]>(2*numCores); // preallocated work items
        unique_ptr<MandelResult[]> r = make_unique<MandelResult[]>(2*numCores);     // preallocated result items
        for(int i=0;i<2*numCores;i++)
            r[i].init(0,0);

        shared_ptr< QueueMonitor<MandelWorkItem *> > inque = make_shared< QueueMonitor<MandelWorkItem *> >(2*numCores, w.get());
        shared_ptr<QueueMonitor<MandelResult *>> outque = make_shared<QueueMonitor<MandelResult *>>(2*numCores,r.get());

        unique_ptr < thread > thr[numCores];
        unique_ptr<WorkerThread> workFunc[numCores];
        for(int i=0;i<numCores;i++)
        {
            workFunc[i] = make_unique<WorkerThread>(i, inque, outque);          
            thr[i] = make_unique<thread>(ref(*workFunc[i]));
        }

        // one loop for sending and recv messages
        bool endOfWork=false;
        int numWorkerThreads = numCores;
        int assigned=0;
        while (1)
        {
            // receiving part
            if(!endOfWork  && assigned != numCores )
            {
                MandelWorkItem *w = inque->reserve();
                MPI_Recv (w, 1, MandelWorkItem::type, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // get a new work item
                int tag = status.MPI_TAG;
                if (tag == ENDTAG)
                {
                    for(int i=0;i<numCores;i++)
                        inque->enque(NULL);
                    endOfWork=true;
                }
                else
                {
                    inque->enque(w);
                    assigned++;
                }
            }

            // sending part
            MandelResult *res;
            if(outque->availItems() >0)
            {
                res = outque->deque();
                if(res == NULL)
                {
                    numWorkerThreads--;
                }
                else
                {
                    MPI_Request r;
                    MPI_Status s;

                    MPI_Issend (res->imgPart.get(), res->getResultSize(), MandelResult::type, 0, RESULTTAG, MPI_COMM_WORLD, &r); // return the results
                    MPI_Wait(&r, &s);

                    outque->release(res);
                    assigned--;
                }
            }

            if(!numWorkerThreads) // terminate the loop
                break;
        }

        for(int i=0;i<numCores;i++)
            thr[i]->join();
    }
    MPI_Finalize ();
    return 0;
}
