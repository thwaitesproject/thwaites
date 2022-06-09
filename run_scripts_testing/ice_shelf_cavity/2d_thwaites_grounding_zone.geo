km = 1000;

// Cavity geometry

Point(1) = {0, -520, 0, 5.0};
Point(2) = {0, -620, 0, 5.0};
Point(3) = {2800,-490, 0, 5.0};
Point(4) = {2800, -480, 0, 5.0};

// Connect up ocean cavity points
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};



// Assign Physical groups
// Grounding line
Physical Line(1) = {3};
// Open ocean
Physical Line(2) = {1};
// Seabed
Physical Line(3) = {2};
// ice-ocean
Physical Line(4) = {4};

Line Loop(11) = {4, 1, 2, 3};
Plane Surface(12) = {11};

Physical Surface(1) = {12};

