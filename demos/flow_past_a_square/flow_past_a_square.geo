H = 0.04;
x1 = 5*H + H/2;
x2 = 20*H - H/2;
dx = H/200;
dx1 = H/4;

// Outer box
Point(1) = {-x1, -7*H, 0, dx1};
Point(2) = {x2, -7*H, 0, dx1};
Point(3) = {x2, 7*H, 0, dx1};
Point(4) = {-x1, 7*H, 0, dx1};

// Inner box
Point(5) = {-H/2, -H/2, 0, dx};
Point(6) = {H/2, -H/2, 0, dx};
Point(7) = {H/2, H/2, 0, dx};
Point(8) = {-H/2, H/2, 0, dx};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Line Loop(5) = {1,2,3,4};

Line(6) = {5,6};
Line(7) = {6,7};
Line(8) = {7,8};
Line(9) = {8,5};
Line Loop(10) = {6,7,8,9};

Plane Surface(11) = {5,10};

// Top at y = h
Physical Line(3) = {3};
// Side along x = -x1 (INLET)
Physical Line(1) = {4};
// Side along x = H+x2 (OUTLET)
Physical Line(2) = {2};
// Bottom at y = 0
Physical Line(4) = {1};

// Inner cube
Physical Line(5) = {6,7,8,9};

// This is just to ensure all the interior
// elements get written out. 
Physical Surface(1) = {11};

