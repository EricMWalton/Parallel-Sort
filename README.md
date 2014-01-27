Parallel-LSD-Sort
=================

This program uses an LSD radix sort and the CUDA framework to sort a series of unsigned integers.  This was an assignment on Udacity's [Parallel Programming](https://www.udacity.com/course/cs344) course.

I experimented quite a bit for this one.  The course expects you to use an LSD radix sort, but initially I built a sorting network (for individual thread blocks), then built a solution using Mergesort.  The LSD version listed here is quite a bit faster.

Usage
-----

Simply copy and paste the contents of "fullProgram.cpp" into the Udacity IDE.  PLEASE do not use this if you are taking the class.  I benefitted a great deal from trying many solutions, and I don't want to encourage others to skip those important steps.

Speed
-----

This is not quite in the ballpark of the fastest solutions on Udacity--various commenters report times of around 1 millisecond.  My version runs in about 6 milliseconds.

Acknowledgements
----------------

I took the basic outline of the algorithm from [this paper](http://mgarland.org/files/papers/nvr-2008-001.pdf).  
