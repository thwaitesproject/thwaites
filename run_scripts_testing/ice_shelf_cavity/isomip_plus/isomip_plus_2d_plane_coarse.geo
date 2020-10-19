// This is based on the fluidity 3d ice shelf
// test. 

// It needs to be squashed to create a layered mesh. 

// This is half the resolution required for isomip+
// i.e dx = dy = 4km and 18 layers

// In xy-plane, z unused
// Define km
km=1000.0;

// Ice shelf front depth
iceshelffrontdepth=120.0;

// Ice shelf depth below front (draft + additional depth below max draft)
oceandepthminusfront=600.0;

// Shelf length
shelflength=320*km;

// Open ocean length
oceanlength=160*km;

//oceanwidth
oceanwidth=80*km;

// Southern boundary
Point(1) = {0,0,0};
Extrude {0, oceandepthminusfront, 0} {
Point{1}; Layers{15};
}

// Extrude north in latitude
//Line(1) = {1,2};
Extrude {shelflength,0,0} {
Line{1}; Layers{80};
}
// Extrude north in latitude again
//Line(2) = {3,4};
//Line(3) = {1,2};
//Line(4) = {2,3};
Extrude {oceanlength,0,0} {
Line{2}; Layers{40};
}
// Extrude up vertically, creating ice shelf front
//Line(6) = {5,6};
//Line(7) = {3,5};
//Line(8) = {4,6};
Extrude {0,iceshelffrontdepth, 0} {
Line{8}; Layers{3};
}
//Line(10) = {4,7};
//Line(11) = {6,8};
//Line(12) = {7,8};

//for 3D
/*Extrude {0,oceanwidth,0} {*/
/*Surface{5,9,13}; Layers{10};*/
/*}*/
//Extrude {0,0.5*oceanwidth,0} {
//Surface{5,9,13}; Layers{10};
//}

//Extrude {0,-0.5*oceanwidth,0} {
//Surface{5,9,13}; Layers{10};
//}

/*Extrude {0,oceanwidth,0} {*/
/*Surface{9}; Layers{10};*/
/*}*/
/*Extrude {0,oceanwidth,0} {*/
/*Surface{13}; Layers{10};*/
/*}*/

// Boundaries
// Ocean free surface
Physical Line(101) = {10};
// Bottom
Physical Line(102) = {3,7};
// South
Physical Line(103) = {1};
// North
Physical Line(104) = {6,12};
// Ice front
Physical Line(105) = {11};
// Ice slope
Physical Line(106) = {4};

Physical Surface(50) = {5,9,13};

