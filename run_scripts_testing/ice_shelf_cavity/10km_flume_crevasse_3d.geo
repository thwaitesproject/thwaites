//                              _105
//                          104|    |104
//                 ______105___|    |____105_____
//------------->   |                             | 
//Flow          103|                             |102
//                 |_____________________________|
//                               101

// top surface outer wall
km = 1000;
Point(1) = {0, 0, -500, 50};
Point(2) = {0, 10*km, -500, 50};
Point(3) = {10*km, 10*km, -500, 50};
Point(4) = {10*km, 0, -500, 50};

// top surface bottom of crevasse
Point(5) = {1*km, 4.9*km, -500, 20};
Point(6) = {1*km, 5.1*km, -500, 20};
Point(7) = {9*km, 5.1*km, -500, 20};
Point(8) = {9*km, 4.9*km, -500, 20};

// top surface top of crevasse
Point(9) = {1.1*km, 4.975*km, -450, 20};
Point(10) = {1.1*km, 5.025*km, -450, 20};
Point(11) = {8.9*km, 5.025*km, -450, 20};
Point(12) = {8.9*km, 4.975*km, -450, 20};

// Bottom corners
Point(13) = {0, 0, -600, 50};
Point(14) = {0, 10*km, -600, 50};
Point(15) = {10*km, 10*km, -600, 50};
Point(16) = {10*km, 0, -600, 50};


// near side surface lines y = 0
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line(9) = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 9};

// bottom 
Line(13) = {13, 14};
Line(14) = {14, 15};
Line(15) = {15, 16};
Line(16) = {16, 13};

// top of crevasse to bottom of crevsase
Line(17) = {5, 9};
Line(18) = {6, 10};
Line(19) = {7, 11};
Line(20) = {8, 12};

// corner of domain to bottom of crevsase
Line(21) = {1, 5};
Line(22) = {2, 6};
Line(23) = {3, 7};
Line(24) = {4, 8};

// bottom to top corner
Line(25) = {13, 1};
Line(26) = {14, 2};
Line(27) = {15, 3};
Line(28) = {16, 4};

// bottom corners
Line(29) = {13, 14};
Line(30) = {14, 15};
Line(31) = {15, 16};
Line(32) = {16, 13};


// top of crevasse 
Line Loop(1) = {9, 10, 11, 12};

// x < 1.1km 
Line Loop(2) = {5,18,-9,-17};

// y > 5.025km
Line Loop(3) = {18,10,-19,-6};


// x > 8.9km
Line Loop(4) = {19,11,-20,-7};


// y < 4.975km
Line Loop(5) = {20,12,-17,-8};


// x < 1km
Line Loop(6) = {1,22,-5, -21};

// y > 9km
Line Loop(7) = {22,6,-23, -2};

// x > 9km
Line Loop(8) = {23, 7,-24, -3};

// y < 1km
Line Loop(9) = {24, 8,-21, -4};

// x = 0
Line Loop(10) = {29, 26, -1, -25};

// y = 10km
Line Loop(11) = {26, 2, -27, -30};

// x = 10km
Line Loop(12) = {27, 3, -28, -31};

// y = 0km
Line Loop(13) = {25, -4, -28, 32};

// bottom
Line Loop(14) = {29, 30, 31, 32};



Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};
Plane Surface(6) = {6};
Plane Surface(7) = {7};
Plane Surface(8) = {8};
Plane Surface(9) = {9};
Plane Surface(10) = {10};
Plane Surface(11) = {11};
Plane Surface(12) = {12};
Plane Surface(13) = {13};
Plane Surface(14) = {14};


//Bottom
Physical Surface(1) = {14};
//North (x)
Physical Surface(2) = {12};
//South
Physical Surface(3) = {10};
//Ice top
Physical Surface(4) = {1,2,3,4,5,6,7,8,9};
//East (y)
Physical Surface(5) = {13};
//West (y)
Physical Surface(6) = {11};

Surface Loop(1) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14};

Volume(1) = {1};
Physical Volume(1) = 1;
