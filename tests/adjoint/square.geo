Point(1) = {0,0,0,0.005};
Point(2) = {0,1,0,0.05};
Point(3) = {1,1,0,0.05};
Point(4) = {1,0,0,0.005};
Line(1) = {1,2};
Line(2) = {4,3};
Line(3) = {1,4};
Line(4) = {2,3};
Line Loop(5) = {1,4,-2,-3};
Plane Surface(6) = {5};
Physical Surface(1) = {6};
Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};
