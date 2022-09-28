//                              _105
//                          104|    |104
//                 ______105___|    |____105_____
//------------->   |                             | 
//Flow          103|                             |102
//                 |_____________________________|
//                               101


Point(1) = {0, -600, 0, 10.0};
Point(2) = {0, -500, 0, 10.0};

Point(3) = {900, -500, 0, 5.0};


Point(4) = {975, -450, 0, 5.0};

Point(5) = {1025, -450, 0, 5.0};
Point(6) = {1100, -500, 0, 5.0};

Point(7) = {2000, -500, 0, 10.0};
Point(8) = {2000,-600, 0, 10.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};

Line Loop(13) = {8, 1, 2, 3, 4, 5, 6, 7};
Plane Surface(14) = {13};
//Bottom
Physical Line(1) = {8};
//North
Physical Line(2) = {7};
//South
Physical Line(3) = {1};
//Ice sides
//Physical Line(4) = {3,5};
//Ice top
Physical Line(4) = {2, 3, 4, 5, 6};

Plane Surface(107) = {13};
Physical Surface(108) = {14};
