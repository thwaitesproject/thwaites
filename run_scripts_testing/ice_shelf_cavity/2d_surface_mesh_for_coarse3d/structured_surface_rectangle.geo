km = 1000;

// Cavity geometry
cavity_xlength = 5000;
cavity_ylength = 10000;

dx = 500;
east_nodes = (cavity_xlength / dx) + 1;
north_nodes = (cavity_ylength / dx) + 1;

Point(1) = {0, 0, 0, dx};
Point(2) = {cavity_xlength, 0, 0, dx};
Point(3) = {cavity_xlength, cavity_ylength, 0, dx};
Point(4) = {0, cavity_ylength, 0, dx};

// Connect up ocean cavity points
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Assign Physical groups
// Physical id 3 and 5 for bottom and top
Physical Line(1) = {1}; // grounding line wall
Physical Line(5) = {2};  
Physical Line(2) = {3}; // open ocean
Physical Line(6) = {4}; 


// layered mesh in cavity 
Transfinite Line {4,-2} = north_nodes Using Progression 1;
Transfinite Line {1,-3} = east_nodes Using Progression 1;


Line Loop(11) = {4, 1, 2, 3};
Plane Surface(12) = {11};

Transfinite Surface {12};

Physical Surface(1) = {12};


