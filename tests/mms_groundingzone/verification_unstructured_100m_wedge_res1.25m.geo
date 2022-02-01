
// verification rectangle for MMS

Point(1) = {0, 0, 0, 1.25};
Point(2) = {0, 50, 0, 1.25};
Point(3) = {100, 100, 0, 1.25};
Point(4) = {100, 0, 0, 1.25};

// Connect up ocean cavity points
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};



// Assign Physical groups
// Grounding line
Physical Line(1) = {1};
// Open ocean
Physical Line(2) = {3};
// Seabed
Physical Line(3) = {4};
// ice-ocean
Physical Line(4) = {2};

Line Loop(11) = {4, 1, 2, 3};
Plane Surface(12) = {11};

Physical Surface(1) = {12};

