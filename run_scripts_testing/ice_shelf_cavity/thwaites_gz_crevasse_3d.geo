//                              _105
//                          104|    |104
//                 ______105___|    |____105_____
//------------->   |                             | 
//Flow          103|                             |102
//                 |_____________________________|
//                               101

// nearside y = 0 plane
Point(1) = {0, 0, -600, 25.0};
Point(2) = {0, 0, -500, 25.0};

Point(3) = {900, 0, -500, 5.0};


Point(4) = {975, 0, -450, 5.0};

Point(5) = {1025, 0, -450, 5.0};
Point(6) = {1100, 0, -500, 5.0};

Point(7) = {2000, 0, -500, 25.0};
Point(8) = {2000, 0, -600, 25.0};

// farside y = 50 plane
Point(9) = {0, 2000, -600, 25.0};
Point(10) = {0, 2000, -500, 25.0};

Point(11) = {900, 2000, -500, 10.0};

Point(12) = {975, 2000, -450, 10.0};

Point(13) = {1025, 2000, -450, 10.0};
Point(14) = {1100, 2000, -500, 10.0};

Point(15) = {2000, 2000, -500, 25.0};
Point(16) = {2000, 2000, -600, 25.0};

// near side surface lines y = 0
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};


// far side surface lines y = 50
Line(9) = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 13};
Line(13) = {13, 14};
Line(14) = {14, 15};
Line(15) = {15, 16};
Line(16) = {16, 9};

//  lines on x = 0
Line(17) = {2, 10};
Line(18) = {9, 1};
// lines on x = 50
Line(19) = {15, 7};
Line(20) = {8, 16};

// lines on x = 2330 (walls of cavity)
Line(21) = {3, 11}; 
Line(22) = {12, 4}; 

// lines on x = 2670 (walls of cavity)
Line(23) = {13, 5}; 
Line(24) = {6, 14}; 

// nearside y = 0
Line Loop(1) = {8, 1, 2, 3, 4, 5, 6, 7};

// farside y = 50
Line Loop(2) = {16, 9, 10, 11, 12, 13, 14, 15};

// x = 0
Line Loop(3) = {1, 17, -9, 18};

// x = 5000
Line Loop(4) = {7, 20, -15, 19};

// bottom  
Line Loop(5) = {18, -8, 20, 16};

// top x < 2330 
Line Loop(6) = {17, 10, -21, -2};

// cavity side x = 2330 
Line Loop(7) = {21, 11, 22, -3};

// top cavity  
Line Loop(8) = {22, 4, -23, -12};

// cavity side x = 2670 
Line Loop(9) = {24, -13, 23, 5};


// top x > 2670 
Line Loop(10) = {24, 14, 19, -6};


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


//Bottom
Physical Surface(1) = {5};
//North (x)
Physical Surface(2) = {4};
//South
Physical Surface(3) = {3};
//Ice top
Physical Surface(4) = {6, 7, 8, 9, 10};
//East (y)
Physical Surface(5) = {1};
//West (y)
Physical Surface(6) = {2};

Surface Loop(1) = {1,2,3,4,5,6,7,8,9,10};

Volume(1) = {1};
Physical Volume(1) = 1;
