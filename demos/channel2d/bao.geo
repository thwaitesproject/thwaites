H1 = 1.0;
H2 = 0.6*H1;
s = 1.0;
s2 = 0.01;

Point(1) = {0,0,0,s};
Point(2) = {0, 21*H1, 0,s};
Point(3) = {33*H1, 21*H1, 0,s};
Point(4) = {33*H1, 0, 0,s};

Point(5) = {8*H1, 10*H1, 0, s2};
Point(6) = {8*H1, 11*H1, 0, s2};
Point(7) = {9*H1, 10.5*H1+0.5*H2, 0, s2};
Point(8) = {9*H1, 10.5*H1-0.5*H2, 0, s2};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,5};

Line Loop(1) = {1,2,3,4};
Line Loop(2) = {5,6,7,8};

Plane Surface(1) = {1,2};

Physical Line(1) = 1;
Physical Line(2) = 3;
Physical Line(3) = 4;
Physical Line(4) = 2;
Physical Line(5) = {5,6,7,8};

Physical Surface(1) = 1;
