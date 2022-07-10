This is a Mandelbrot set explorer written in CUDA and using VTK for
graphics and mouse interactions.  The code is targeted to run on my
"personal supercomputer", which is a Dell Precision 7810 tower with an
Nvidia Tesla K80 GPGPU as the compute engine.

This Mandelbrot explorer iterates the logistic map,

$$
x_{n+1} = \lambda x_n (1 - x_n)
$$

instead of the usual Mandelbrot map

$$
z_{n+1} = z_n^2 + C
$$

My explorer shows the escape time vs the parameter $\lambda$ in the
complex plane.  I take the sqrt() of the escape times prior to
plotting and also fiddle with the color mapping in order to get pretty
images.  Some screenshots are below. 

<p float="left">
<img src="Mandelbrot1.png" height="300" width="300"/>
<img src="Mandelbrot2.png" height="300" width="300"/>
</p>
<p float="left">
<img src="Mandelbrot3.png" height="300" width="300"/>
<img src="Mandelbrot4.png" height="300" width="300"/>
</p>

--------------------------------------------------------------
To run the explorer, just type "Mandelbrot" at the command line.  The
middle mouse button pans the image and the mouse wheel will zoom in
and out.  The other mouse buttons are disabled.  The
following command line args are optional:

-N Number of iterations to use.  Default is 8, but higher numbers (in
 the thousands) produce better Mandelbrot sets.

These are useful if you want to start the program displaying a
particular area in the complex plane:

-x Starting x    
-y Starting y    
-w Starting width to display    
-h Starting height to display    

--------------------------------------------------------------
To build the program from scratch:    

mkdir build    
cd build    
cmake ..    
make    
./Mandelbrot    

Ignore any warnings about thrust you get.

Stuart Brorson, July 2022.


