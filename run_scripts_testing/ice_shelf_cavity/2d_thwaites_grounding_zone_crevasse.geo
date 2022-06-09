km = 1000;

// Cavity geometry

Point(1) = {0, -520, 0, 5.0};
Point(2) = {0, -620, 0, 5.0};
Point(3) = {2800,-490, 0, 5.0};
Point(4) = {2800, -480, 0, 5.0};

// Crevasse
Point(5) = {1400, -500, 0, 5.0};
Point(6) = {1400, -480, 0, 5.0};
Point(7) = {1200, -480, 0, 5.0};
Point(8) = {1200, -500, 0, 5.0};

// Connect up ocean cavity points
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};


// Assign Physical groups
Physical Line(1) = {3};
Physical Line(2) = {1};
Physical Line(3) = {2};
Physical Line(4) = {4, 5, 6, 7, 8};

Line Loop(11) = {8, 1, 2, 3, 4, 5, 6, 7};
Plane Surface(12) = {11};

Physical Surface(1) = {12};

