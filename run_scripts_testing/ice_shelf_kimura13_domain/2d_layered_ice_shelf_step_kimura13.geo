km = 1000;

// Cavity geometry
cavity_length = 50 * km; 
open_ocean_length = 50 * km;
H1 = 100;
H2 = H1 + 800;
H3 = H2 + 200;

// Set horizontal resolution
dx = 1 * km;
cavity_horizontal_nodes = (cavity_length / dx) + 1;
ocean_horizontal_nodes = (open_ocean_length / dx) + 1;

// Set vertical resolution
cavity_layers = 9;
ice_front_layers = 2;
cavity_vertical_nodes = cavity_layers + 1;
ice_front_vertical_nodes =  ice_front_layers + 1; 

// Points in ice shelf cavity
Point(1) = {0, 0, 0, dx};
Point(2) = {cavity_length, 0, 0, dx};
Point(3) = {cavity_length, H2, 0, dx};
Point(4) = {0, H1, 0, dx};

// Points outside ice shelf cavity
Point(5) = {cavity_length + open_ocean_length, 0, 0, dx};
Point(6) = {cavity_length + open_ocean_length, H2, 0, dx};
Point(7) = {cavity_length + open_ocean_length, H3, 0, dx};
Point(8) = {cavity_length, H3, 0, dx};

// Connect up ice shelf cavity points 
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Connect up open ocean points
Line(5) = {2, 5};
Line(6) = {5, 6};
Line(7) = {6, 7};
Line(8) = {7, 8};
Line(9) = {8, 3};

// Add extra line for transfinite surface
Line(10) = {3, 6};


// Assign Physical groups
Physical Line("seabed") = {1, 5};
Physical Line("ice_base") = {3};
Physical Line("grounding_line") = {4};
Physical Line("open_ocean") = {6, 7};
Physical Line("ocean_surface") = {8};
Physical Line("ice_front") = {9};

// layered mesh in cavity 
Transfinite Line {4,-2} = cavity_vertical_nodes Using Progression 1;
Transfinite Line {1,-3} = cavity_horizontal_nodes Using Progression 1;

// layered mesh ocean outside cavity 
Transfinite Line {2, 6} = cavity_vertical_nodes Using Progression 1;
Transfinite Line {9, -7} = ice_front_vertical_nodes Using Progression 1;

Transfinite Line {5, 10} = ocean_horizontal_nodes Using Progression 1;
Transfinite Line {10, -8} = ocean_horizontal_nodes Using Progression 1;

Line Loop(11) = {4, 1, 2, 3};
Line Loop(12) = {5, 6, -10, -2};
Line Loop(13) = {10, 7, 8, 9};

Plane Surface(1) = {11};
Plane Surface(2) = {12};
Plane Surface(3) = {13};

Transfinite Surface {1,2,3};



