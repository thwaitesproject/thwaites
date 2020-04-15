H = 100.;
x1 = 5*H + H/2;
x2 = 30*H - H/2;
dx1 = H;

// Outer box
Point(1) = {-x1, -10*H, 0, dx1};
Point(2) = {x2, -10*H, 0, dx1};
Point(3) = {x2, 10*H, 0, dx1};
Point(4) = {-x1, 10*H, 0, dx1};


Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Line Loop(5) = {1,2,3,4};


Plane Surface(11) = {5};

V[]=Extrude{0,0,5*H}{Surface{11};};

// Top at y = h
Physical Surface(3) = {11, 20, 28, 33};
// Side along x = -x1 (INLET)
Physical Surface(1) = {32};
// Side along x = H+x2 (OUTLET)
Physical Surface(2) = {24};

// This is just to ensure all the interior
// elements get written out. 
Physical Volume(1) = {V[1]};

