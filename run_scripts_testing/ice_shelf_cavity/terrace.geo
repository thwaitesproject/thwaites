//                              _105
//                          104|    |104
//                 ______105___|    |____105_____
//------------->   |                             | 
//Flow          103|                             |102
//                 |_____________________________|
//                               101


Point(1) = {0, -400, 0, 25.0};
Point(2) = {0, -300, 0, 25.0};

Point(3) = {1000, -300, 0, 5.0};


Point(4) = {1000, -250, 0, 5.0};

Point(5) = {2000, -250, 0, 5.0};
Point(6) = {2000, -400, 0, 25.0};



Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};

Line Loop(13) = {6, 1, 2, 3, 4, 5};
Plane Surface(14) = {13};
//Bottom
Physical Line(1) = {6};
//North
Physical Line(2) = {5};
//South
Physical Line(3) = {1};
//Ice sides
//Physical Line(4) = {3,5};
//Ice top
Physical Line(4) = {2, 3, 4};

Plane Surface(107) = {13};
Physical Surface(108) = {14};
