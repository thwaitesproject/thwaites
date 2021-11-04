import pytest
from firedrake_adjoint import *


def test_nothing():
    # REMOVE ME: placeholder to ensure we have at least one test
    pass

# Run test 3 times for no timesteps, 1 timestep and 10 timesteps
@pytest.mark.parameterize("T", [(10.), (900.), (9000.))])
def test_2d_isomip_cavity_salfunctional(T):

                    
    # ISOMIP+ setup 2d slice with extuded mesh
    #Buoyancy driven overturning circulation
    # beneath ice shelf.
    from thwaites import *
    from thwaites.utility import get_top_surface, cavity_thickness, CombinedSurfaceMeasure, ExtrudedFunction
    from thwaites.utility import offset_backward_step_approx
    from firedrake.petsc import PETSc
    from firedrake import FacetNormal
    import pandas as pd
    import argparse
    import numpy as np
    from math import ceil
    from pyop2.profiling import timed_stage
    import rasterio
    from thwaites.interpolate import interpolate as interpolate_data

    ##########
    ADJOINT = True

    if ADJOINT:
        from firedrake_adjoint import *
        from thwaites.diagnostic_block import DiagnosticBlock

    ##########
    PETSc.Sys.popErrorHandler()

    restoring_time = Constant(0.1*86400.)
    ##########

    #  Generate mesh
    L = 800E3
    grounding_line = 460E3 # Ice domain from isomip+ offsets ocean domain
    domain_length = L - grounding_line
    shelf_length = 640E3
    H1 = 130.
    H2 = 600.
    H3 = 720.
    dy = 4000. 
    ny = round(domain_length/dy)
    nz = 15.
    dz = H2/nz #40.0
    water_depth = H3

    # create mesh
    grounding_line = 460E3 # Ice domain from isomip+ offsets ocean domain
    base_mesh = IntervalMesh(ny, grounding_line, L)
    layers = []
    cell = 0
    yr = 0
    min_dz = 0.5*dz # if top cell is thinner than this, merge with cell below
    tiny_dz = 0.01*dz # workaround zero measure facet issue (fd issue #1858)

    x_base = SpatialCoordinate(base_mesh)

    P1 = FunctionSpace(base_mesh, "CG", 1)
    ocean_thickness = Function(P1)
    ocean_thickness.interpolate(conditional(x_base[0] + 0.5*dy < shelf_length, H2, H3))

    PETSc.Sys.Print(len(ocean_thickness.dat.data[:]))

    def extruded_cavity_mesh(base_mesh, ocean_thickness):
        P0dg = FunctionSpace(base_mesh, "DG", 0)
        P0dg_cells = Function(P0dg)
        tmp = ocean_thickness.copy(deepcopy=True)
        P0dg_cells.assign(np.finfo(0.).min)
        par_loop("""for (int i=0; i<bathy.dofs; i++) {
                bathy_max[0] = fmax(bathy[i], bathy_max[0]);
                }""",
                dx, {'bathy_max': (P0dg_cells, RW), 'bathy': (tmp, READ)})

        P0dg_cells /= dz

        P0dg_cells_array = P0dg_cells.dat.data_ro_with_halos[:]

        for i in P0dg_cells_array:
            layers.append([0, i])

        mesh = ExtrudedMesh(base_mesh, layers, layer_height=dz)
        return mesh 

    mesh = extruded_cavity_mesh(base_mesh, ocean_thickness)
    x, z = SpatialCoordinate(mesh)

    P0_extruded = FunctionSpace(mesh, 'DG', 0)
    p0mesh_cells = Function(P0_extruded)
    PETSc.Sys.Print("number of cells:", len(p0mesh_cells.dat.data[:]))

    # Define ocean cavity thickness on extruded mesh
    P1_extruded = FunctionSpace(mesh, 'CG', 1)

    ocean_thickness_extruded = ExtrudedFunction(ocean_thickness, mesh_3d=mesh)
    # move top nodes to correct position:
    cfs = mesh.coordinates.function_space()
    bc = DirichletBC(cfs, as_vector((x, ocean_thickness_extruded.view_3d)), "top")
    bc.apply(mesh.coordinates)

    # Bathymetry 

    x_bar = Constant(300E3) # Characteristic along flow length scale of the bedrock
    x_tilda = x / x_bar  # isomip+ x coordinate used for defining along flow bathymetry/bedrock topography (N.b offset by 320km because of ice domain)
    B0 = Constant(-150.0) # Bedrock topography at x = 0 (in the ice domain!)
    B2 = Constant(-728.8) # Second bedrock topography coefficient 
    B4 = Constant(343.91) # Third bedrock topography coefficient
    B6 = Constant(-50.57) # Forth bedrock topography coefficient

    bathy_x = B0 + B2 * pow(x_tilda, 2) + B4 * pow(x_tilda, 4) + B6 * pow(x_tilda, 6)
    bathymetry = Function(P1_extruded)
    # the adjoint of interpolation to extruded functions seems broken/not implemented (unknnow reference element error)
    with stop_annotating():
        bathymetry.interpolate(conditional(bathy_x < -water_depth,
                                -water_depth,
                                bathy_x))
    #bathymetry.assign(-720)
    print("max bathy : ",bathymetry.dat.data[:].max())

    ice_draft_filename = "Ocean1_input_geom_v1.01.nc"
    ice_draft_file = rasterio.open(f'netcdf:{ice_draft_filename}:lowerSurface', 'r')
    ice_draft = Function(P1_extruded)
    #ice_draft.interpolate(conditional(x - 0.5*dy < shelf_length, (x/shelf_length)*(H2-H1) + H1, H3) - water_depth) 
    ice_draft_base = interpolate_data(ice_draft_file, P1, y_transect=41000)  # Get ice shelf draft along y=41km transect i.e the middle

    #print("max icedraft : ",ice_draft_base.dat.data[:].max())
    #print("min icedraft : ",ice_draft_base.dat.data[:].min())

    ocean_thickness = Function(P1_extruded)

    ice_draft = ExtrudedFunction(ice_draft_base, mesh_3d=mesh)
    #print("max icedraft extruded : ",ice_draft.view_3d.dat.data[:].max())
    #print("min icedraft extruded : ",ice_draft.view_3d.dat.data[:].min())
    # the adjoint of interpolation to extruded functions seems broken/not implemented (unknnow reference element error)
    #with stop_annotating():
    #ocean_thickness.interpolate(ice_draft - bathymetry)

    print("max thickness : ", ocean_thickness.dat.data[:].max())
    print("min thickness : ", ocean_thickness.dat.data[:].min())
    # the adjoint of interpolation to extruded functions seems broken/not implemented (unknnow reference element error)
    with stop_annotating():
        ocean_thickness.interpolate(conditional(ice_draft.view_3d - bathymetry < Constant(10),
                                            Constant(10),
                                            ice_draft.view_3d - bathymetry)) 
    print("max thickness : ", ocean_thickness.dat.data[:].max())
    print("min thickness : ", ocean_thickness.dat.data[:].min())

    # Scale the mesh to make ice shelf slope
    Vc = mesh.coordinates.function_space()
    x, z = SpatialCoordinate(mesh)
    f = Function(Vc).interpolate(as_vector([x, conditional(x + 0.5*dy < shelf_length, ocean_thickness*z/H2, ocean_thickness*z/H3) - -bathymetry]))
    #f = Function(Vc).interpolate(as_vector([x, ocean_thickness*z/H3 - -bathymetry]))
    with stop_annotating():
        mesh.coordinates.assign(f)


    ds = CombinedSurfaceMeasure(mesh, 5)

    PETSc.Sys.Print("Mesh dimension ", mesh.geometric_dimension())

    # Set ocean surface
    #mesh.coordinates.dat.data[:, 1] -= bathmetry

    print("You have Comm WORLD size = ", mesh.comm.size)
    print("You have Comm WORLD rank = ", mesh.comm.rank)

    x, z = SpatialCoordinate(mesh)

    PETSc.Sys.Print("Length of South side (Gl wall) should be 10m: ", assemble((Constant(1.0)*ds(1, domain=mesh))))

    PETSc.Sys.Print("Length of North side (open ocean) should be 720m: ", assemble(Constant(1.0)*ds(2, domain=mesh)))

    PETSc.Sys.Print("Length of bottom: should be 340e3m: ", assemble(Constant(1.0)*ds("bottom", domain=mesh)))

    PETSc.Sys.Print("length of ocean surface should be 160e3m", assemble(conditional(x > shelf_length, Constant(1.0), 0.0)*ds("top", domain=mesh)))

    PETSc.Sys.Print("Length of iceslope: should be ...: ", assemble(conditional(x < shelf_length, Constant(1.0), 0.0)*ds("top", domain=mesh)))

    n = FacetNormal(mesh)
    print("ds_v",assemble(avg(dot(n,n))*dS_v(domain=mesh)))




    if ADJOINT:
        # don't want to integrate over the entire top surface 
        # and conditional doesnt seem to work in adjoint when used in J...
        # wavelength of the step = x distance that fucntion goes from zero to 1. 
        lambda_step = 2 * dy
        k = 2.0 * np.pi / lambda_step 
        x0 = shelf_length - 0.5 * lambda_step  # this is the centre of the step.


        PETSc.Sys.Print("Alternatively using approx step function, length of iceslope: should be 320000.64m: ", assemble( Constant(1.0)*offset_backward_step_approx(x,k,x0)*ds("top", domain=mesh)))

        PETSc.Sys.Print("Alternatively using approx step function, length of bottom up to ice shelf: should equal x0 = {}m and hopefully be close to shelf length: ".format(x0), assemble( Constant(1.0)*offset_backward_step_approx(x,k,x0)*ds("bottom", domain=mesh)))





    ##########
    print(mesh.ufl_cell())
    # Set up function spaces
    v_ele = FiniteElement("RTCE", mesh.ufl_cell(), 2, variant="equispaced")
    #v_ele = FiniteElement("DQ", mesh.ufl_cell(), 1, variant="equispaced")
    V = FunctionSpace(mesh, v_ele) # Velocity space
    W = FunctionSpace(mesh, "Q", 2)  # pressure space
    M = MixedFunctionSpace([V, W])

    # u velocity function space.
    ele = FiniteElement("DQ", mesh.ufl_cell(), 1, variant="equispaced")
    U = FunctionSpace(mesh, ele)
    VDG = VectorFunctionSpace(mesh, "DQ", 1) # velocity for output
    vdg_ele = FiniteElement("DQ", mesh.ufl_cell(), 1, variant="equispaced")
    VDG1 = VectorFunctionSpace(mesh, vdg_ele) # velocity for output

    Q = FunctionSpace(mesh, ele)
    K = FunctionSpace(mesh, ele)
    S = FunctionSpace(mesh, ele)

    P1 = FunctionSpace(mesh, "CG", 1)
    ##########
    print("vel dofs:", V.dim())
    print("pressure dofs:", W.dim())
    print("combined dofs:", M.dim())
    print("scalar dofs:", U.dim())
    print("P1 dofs (no. of nodes):", P1.dim())
    ##########
    # Set up functions
    m = Function(M)
    v_, p_ = m.split()  # function: velocity, pressure
    v, p = split(m)  # expression: velocity, pressure
    v_._name = "velocity"
    p_._name = "perturbation pressure"
    vdg = Function(VDG, name="velocity")
    vdg1 = Function(VDG1, name="velocity")

    rho = Function(K, name="density")
    temp = Function(K, name="temperature")
    sal = Function(S, name="salinity")
    melt = Function(Q, name="melt rate")
    Q_mixed = Function(Q, name="ocean heat flux")
    Q_ice = Function(Q, name="ice heat flux")
    Q_latent = Function(Q, name="latent heat")
    Q_s = Function(Q, name="ocean salt flux")
    Tb = Function(Q, name="boundary freezing temperature")
    Sb = Function(Q, name="boundary salinity")
    full_pressure = Function(M.sub(1), name="full pressure")

    rho_anomaly = Function(P1, name="density anomaly")

    ##########

    # Define a dump file

    dump_file = "./isomip_2d_dx4km_nz15_closed_dump_step_9600" 
    DUMP = False
    if DUMP:
        with DumbCheckpoint(dump_file, mode=FILE_READ) as chk:
            # Checkpoint file open for reading and writing
            chk.load(v_, name="velocity")
            chk.load(p_, name="perturbation_pressure")
            chk.load(sal, name="salinity")
            chk.load(temp, name="temperature")

            # ISOMIP+ warm conditions .
            T_surface = -1.9
            T_bottom = 1.0

            S_surface = 33.8
            S_bottom = 34.7
            
            T_restore = T_surface + (T_bottom - T_surface) * (z / -water_depth)
            S_restore = S_surface + (S_bottom - S_surface) * (z / -water_depth)


    else:
        # Assign Initial conditions
        v_init = zero(mesh.geometric_dimension())
        v_.assign(v_init)


        # ISOMIP+ warm conditions .
        T_surface = -1.9
        T_bottom = 1.0

        S_surface = 33.8
        S_bottom = 34.7

        T_restore = T_surface #+ (T_bottom - T_surface) * (z / -water_depth)
        S_restore = S_surface #+ (S_bottom - S_surface) * (z / -water_depth)

        temp_init = T_restore
        temp.assign(temp_init)

        sal_init = S_restore
        sal.assign(sal_init)

    sal_init_func = Function(sal)
    sal_init_func.assign(sal)
    #h = Function(sal)
        
    #h.dat.data[:] = np.random.random(h.dat.data_ro.shape)
    #sal.dat.data[:] += 1*h.dat.data[:]


    if ADJOINT:
        c = Control(sal) 

    #    v_ele = FiniteElement("DQ", mesh.ufl_cell(), 1, variant="equispaced")
    #    V = VectorFunctionSpace(mesh, v_ele) # Velocity space
    #    M = MixedFunctionSpace([V, W])


    #    m = Function(M)
    #    v_, p_ = m.split()  # function: velocity, pressure
    #    v, p = split(m)  # expression: velocity, pressure
    #    v_._name = "velocity"
    #    p_._name = "perturbation pressure"
    #
     #   v_.project(vadj_)
    #    p_.project(padj_)
    ##########

    # Set up equations
    qdeg = 10

    mom_eq = MomentumEquation(M.sub(0), M.sub(0), quad_degree=qdeg)
    cty_eq = ContinuityEquation(M.sub(1), M.sub(1), quad_degree=qdeg)
    #u_eq = ScalarVelocity2halfDEquation(U, U)
    temp_eq = ScalarAdvectionDiffusionEquation(K, K, quad_degree=qdeg)
    sal_eq = ScalarAdvectionDiffusionEquation(S, S, quad_degree=qdeg)

    ##########

    # Terms for equation fields

    # momentum source: the buoyancy term Boussinesq approx. 


    T_ref = Constant(-1.0)
    S_ref = Constant(34.2)
    beta_temp = Constant(3.733E-5)
    beta_sal = Constant(7.843E-4)
    g = Constant(9.81)
    mom_source = as_vector((0.,-g))*(-beta_temp*(temp - T_ref) + beta_sal * (sal - S_ref)) 

    rho0 = 1027.51
    rho.interpolate(rho0*(1.0-beta_temp * (temp - T_ref) + beta_sal * (sal - S_ref)))
    rho_anomaly.project(-beta_temp * (temp - T_ref) + beta_sal * (sal - S_ref))

    gradrho = Function(P1)  # vertical component of gradient of density anomaly units m^-1
    gradrho.project(Dx(rho_anomaly, mesh.geometric_dimension() - 1))


    # coriolis frequency f-plane assumption at 75deg S. f = 2 omega sin (lat) = 2 * 7.2921E-5 * sin (-75 *2pi/360)
    f = Constant(-1.409E-4)

    class VerticalDensityGradientSolver:
        """Computes vertical density gradient.
                                                                                                                                                                                                                           """
        def __init__(self, rho, solution):
            self.rho = rho
            self.solution = solution
            
            self.fs = self.solution.function_space()
            self.mesh = self.fs.mesh()
            self.n = FacetNormal(self.mesh)
            
            test = TestFunction(self.fs)
            tri = TrialFunction(self.fs)
            vert_dim = self.mesh.geometric_dimension()-1
            
            a = test*tri*dx
            L = -Dx(test, vert_dim)*self.rho*dx + test*self.n[vert_dim]*self.rho*ds_tb #+ avg(rho) * jump(gradrho_test, n[dim]) * dS_h (this is zero because jump(phi,n) = 0 for continuous P1 test function!)
           
            prob = LinearVariationalProblem(a, L, self.solution, constant_jacobian=True)
            self.weak_grad_solver = LinearVariationalSolver(prob) # #, solver_parameters=solver_parameters)
           
        def solve(self):
            self.weak_grad_solver.solve()

    P1fs = FunctionSpace(mesh, "CG", 1) 
    gradrho = Function(P1fs)
    grad_rho_solver = VerticalDensityGradientSolver(rho, gradrho)        

    grad_rho_solver.solve()

    # Scalar source/sink terms at open boundary.
    absorption_factor = Constant(1.0/restoring_time)
    sponge_fraction = 0.02  # fraction of domain where sponge
    # Temperature source term
    source_temp = conditional(x > (1.0-sponge_fraction) * L,
                               absorption_factor * T_restore *((x - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
                              0.0)

    # Salinity source term
    source_sal = conditional(x > (1.0-sponge_fraction) * L,
                             absorption_factor * S_restore  *((x - (1.0-sponge_fraction) * L)/(L * sponge_fraction)), 
                             0.0)

    # Temperature absorption term
    absorp_temp = conditional(x > (1.0-sponge_fraction) * L,
                              absorption_factor * ((x - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
                              0.0)

    # Salinity absorption term
    absorp_sal = conditional(x > (1.0-sponge_fraction) * L,
                             absorption_factor * ((x - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
                             0.0)


    # set Viscosity/diffusivity (m^2/s)
    mu_h = Constant(6.0)
    mu_v = Constant(1e-3)
    kappa_h = Constant(1.0)
    kappa_v = Constant(5e-5)

    # linearly vary viscosity/diffusivity over domain. reduce vertical/diffusion
    open_ocean_kappa_v = kappa_v
    grounding_line_kappa_v = Constant(open_ocean_kappa_v * H1/H2)
    kappa_v_grad = (open_ocean_kappa_v - grounding_line_kappa_v) / shelf_length
    kappa_v_cond = conditional(x < shelf_length, grounding_line_kappa_v + x * kappa_v_grad, open_ocean_kappa_v)


    #kappa_v = Function(P1fs) 
    DeltaS = Constant(1.0)  # rough order of magnitude estimate of change in salinity over restoring region
    gradrho_scale = DeltaS * beta_sal / water_depth  # rough order of magnitude estimate for vertical gradient of density anomaly. units m^-1
    #kappa_v.assign(conditional(gradrho / gradrho_scale < 1e-1, 1e-3, 1e-1))

    mu_tensor = as_tensor([[mu_h, 0], [0, mu_v]])
    kappa_tensor = as_tensor([[kappa_h, 0], [0, kappa_v]])

    #TP1 = TensorFunctionSpace(mesh, "CG", 1)
    mu = mu_tensor #Function(TP1, name='viscosity').assign(mu_tensor)
    kappa_temp = kappa_tensor #Function(TP1, name='temperature diffusion').assign(kappa_tensor)
    kappa_sal = kappa_tensor #Function(TP1, name='salinity diffusion').assign(kappa_tensor)
    ##########

    # Equation fields
    vp_coupling = [{'pressure': 1}, {'velocity': 0}]
    vp_fields = {'viscosity': mu, 'source': mom_source, 'interior_penalty': Constant(3.0)}
    temp_fields = {'diffusivity': kappa_temp, 'velocity': v, 'source': source_temp,
                   'absorption coefficient': absorp_temp}
    sal_fields = {'diffusivity': kappa_sal, 'velocity': v, 'source': source_sal,
                  'absorption coefficient': absorp_sal}

    ##########

    # Get expressions used in melt rate parameterisation
    mp = ThreeEqMeltRateParam(sal, temp, p, z, velocity=pow(dot(vdg, vdg) + 1e-6, 0.5), ice_heat_flux=False)

    ##########

    # assign values of these expressions to functions.
    # so these alter the expression and give new value for functions.
    Q_ice.interpolate(mp.Q_ice)
    Q_mixed.interpolate(mp.Q_mixed)
    Q_latent.interpolate(mp.Q_latent)
    Q_s.interpolate(mp.S_flux_bc)
    melt.interpolate(mp.wb)
    Tb.interpolate(mp.Tb)
    Sb.interpolate(mp.Sb)
    full_pressure.interpolate(mp.P_full)

    ##########


    # Boundary conditions
    # top boundary: no normal flow, drag flowing over ice
    # bottom boundary: no normal flow, drag flowing over bedrock
    # grounding line wall (LHS): no normal flow
    # open ocean (RHS): pressure to account for density differences

    # WEAKLY Enforced BCs
    n = FacetNormal(mesh)
    Temperature_term = -beta_temp * (T_surface * z + 0.5 * (T_bottom - T_surface) * (pow(z,2) / -water_depth) - T_ref * z)
    Salinity_term = beta_sal *  (S_surface * z + 0.5 * (S_bottom - S_surface) * (pow(z,2) / -water_depth) - S_ref * z)
    stress_open_boundary = -n*-g*(Temperature_term + Salinity_term)
    no_normal_flow = 0.

    Vinflow = Constant(0.0) # velocity inflow. 10m^3/s 1km x 1m high. doesnt sound that realistic... or 100m x 10m high
    Tinflow = Constant(rho0 * g * 610 * mp.c) # should be -0.46degC which seems too hot I thought it was more like -2degC. is this because deeper? or saltier...
    PETSc.Sys.Print(Tinflow.values()[0])
    Sinflow = Constant(20.0) 

    # test stress open_boundary
    #sop = Function(W)
    #sop.interpolate(-g*(Temperature_term + Salinity_term))
    #sop_file = File(folder+"boundary_stress.pvd")
    #sop_file.write(sop)


    vp_bcs = {"top": {'un': no_normal_flow, 'drag': conditional(x < shelf_length, 2.5E-3, 0.0)}, 
            1: {'un': no_normal_flow}, 2: {'un': no_normal_flow}, 
            "bottom": {'un': no_normal_flow, 'drag': 2.5E-3}} 

    #temp_bcs = {"top": {'flux': conditional(x + 5*dy < shelf_length, -mp.T_flux_bc, 0.0)}}
    temp_bcs = {"top": {'flux': -mp.T_flux_bc * offset_backward_step_approx(x, k, x0) }}
    #sal_bcs = {"top": {'flux':  conditional(x + 5*dy < shelf_length, -mp.S_flux_bc, 0.0)}}
    sal_bcs = {"top": {'flux': -mp.S_flux_bc * offset_backward_step_approx(x, k, x0) }}


    # STRONGLY Enforced BCs
    strong_bcs = []

    ##########

    # Solver parameters
    mumps_solver_parameters = {
        'snes_monitor': None,
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        "mat_mumps_icntl_14": 200,
        'mat_type': 'aij',
        'snes_max_it': 100,
        'snes_rtol': 1e-8,
        'snes_atol': 1e-6,
    }

    pressure_projection_solver_parameters = {
            'snes_type': 'ksponly',
            'snes_monitor': None,
            'ksp_type': 'preonly',  # we solve the full schur complement exactly, so no need for outer krylov
            'ksp_converged_reason': None,
            'ksp_monitor_true_residual': None,
            'mat_type': 'matfree',
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'pc_fieldsplit_schur_fact_type': 'full',
            # velocity mass block:
            'fieldsplit_0': {
                'ksp_converged_reason': None,
                'ksp_monitor_true_residual': None,
                'ksp_type': 'cg',
                'pc_type': 'python',
                'pc_python_type': 'firedrake.AssembledPC',
                'assembled_pc_type': 'bjacobi',
                'assembled_sub_pc_type': 'sor',
                },
            # schur system: don't explicitly assemble the schur system
            # use cg for outer krylov solve. Use LaplacePC with vertical lumping to assemble pc.
            'fieldsplit_1': {
                'ksp_type': 'preonly',
                'ksp_rtol': 1e-7,
                'ksp_atol': 1e-9,
                'ksp_converged_reason': None,
                'ksp_monitor_true_residual': None,
                'pc_type': 'python',
                'pc_python_type': 'thwaites.LaplacePC',
                #'schur_ksp_converged_reason': None,
                'laplace_pc_type': 'ksp',
                'laplace_ksp_ksp_type': 'cg',
                'laplace_ksp_ksp_rtol': 1e-7,
                'laplace_ksp_ksp_atol': 1e-9,
                'laplace_ksp_ksp_converged_reason': None,
                'laplace_ksp_ksp_monitor_true_residual': None,
                'laplace_ksp_pc_type': 'python',
                'laplace_ksp_pc_python_type': 'thwaites.VerticallyLumpedPC',
            }
        }

    predictor_solver_parameters = {
            'snes_monitor': None,
            'snes_type': 'ksponly',
            'ksp_type': 'gmres',
            'pc_type': 'hypre',
            'pc_hypre_boomeramg_strong_threshold': 0.6,  # really this was added for 3d...
            'ksp_converged_reason': None,
    #        'ksp_monitor_true_residual': None,
            'ksp_rtol': 1e-5,
            'ksp_max_it': 300,
            }

    gmres_solver_parameters = {
            'snes_monitor': None,
            'snes_type': 'ksponly',
            'ksp_type': 'gmres',
            'pc_type': 'sor',
            'ksp_converged_reason': None,
            'ksp_rtol': 1e-5,
            'ksp_max_it': 300,
            }



    vp_solver_parameters = pressure_projection_solver_parameters
    u_solver_parameters = gmres_solver_parameters
    temp_solver_parameters = gmres_solver_parameters
    sal_solver_parameters = gmres_solver_parameters

    ##########

    # define time steps
    dt = 900.
    output_dt = 90000
    output_step = output_dt/dt

    ##########

    # Set up time stepping routines

    vp_timestepper = PressureProjectionTimeIntegrator([mom_eq, cty_eq], m, vp_fields, vp_coupling, dt, vp_bcs,
                                                              solver_parameters=mumps_solver_parameters,
                                                              predictor_solver_parameters=mumps_solver_parameters,
                                                              picard_iterations=1)
    #                                                          pressure_nullspace=VectorSpaceBasis(constant=True))

    # performs pseudo timestep to get good initial pressure
    # this is to avoid inconsistencies in terms (viscosity and advection) that
    # are meant to decouple from pressure projection, but won't if pressure is not initialised
    # do this here, so we can see the initial pressure in pressure_0.pvtu
    if not DUMP:
        # should not be done when picking up
        with timed_stage('initial_pressure'):
            vp_timestepper.initialize_pressure()

    #u_timestepper = DIRK33(u_eq, u, u_fields, dt, u_bcs, solver_parameters=u_solver_parameters)
    temp_timestepper = DIRK33(temp_eq, temp, temp_fields, dt, temp_bcs, solver_parameters=mumps_solver_parameters)
    sal_timestepper = DIRK33(sal_eq, sal, sal_fields, dt, sal_bcs, solver_parameters=mumps_solver_parameters)

    ##########

    # Set up Vectorfolder

    #folder = 'tmp/'


    ###########

    # Output files for velocity, pressure, temperature and salinity
    vdg.project(v_) # DQ2 velocity for output

    vdg1.project(v_) # DQ1 velocity 




    ##########

    #with DumbCheckpoint(folder+"initial_pressure_dump", mode=FILE_UPDATE) as chk:
    #    # Checkpoint file open for reading and writing
    #    chk.store(v_, name="velocity")
    #    chk.store(p_, name="perturbation_pressure")
    #    chk.store(temp, name="temperature")
    #    chk.store(sal, name="salinity")

    ############

    if ADJOINT:
        # adjoint output
        tape = get_working_tape()

        adj_s_file = File(folder+"adj_salinity.pvd")
        tape.add_block(DiagnosticBlock(adj_s_file, sal))

        adj_t_file = File(folder+"adj_temperature.pvd")
        tape.add_block(DiagnosticBlock(adj_t_file, temp))

        #adj_visc_file = File(folder+"adj_viscosity.pvd")
        #tape.add_block(DiagnosticBlock(adj_visc_file, mu))
        #adj_diff_t_file = File(folder+"adj_diffusion_T.pvd")
        #tape.add_block(DiagnosticBlock(adj_diff_t_file, kappa_temp))
        #adj_diff_s_file = File(folder+"adj_diffusion_S.pvd")
        #tape.add_block(DiagnosticBlock(adj_diff_s_file, kappa_sal))

    ####################

    # Add limiter for DG functions
    limiter = VertexBasedP1DGLimiter(U)
    v_comp = Function(U)
    w_comp = Function(U)
    ########

    # Begin time stepping
    t = 0.0
    step = 0

    while t < T - 0.5*dt:
        with timed_stage('velocity-pressure'):
            vp_timestepper.advance(t)
            vdg.project(v_)  # DQ2 velocity for melt and plotting 
    #        vdg1.project(v_) # DQ1 velocity for 
    #        v_comp.interpolate(vdg1[0])
    #        limiter.apply(v_comp)
    #        w_comp.interpolate(vdg1[1])
    #        limiter.apply(w_comp)
    #        vdg1.interpolate(as_vector((v_comp, w_comp)))
        with timed_stage('temperature'):
            temp_timestepper.advance(t)
        with timed_stage('salinity'):
            sal_timestepper.advance(t)


        limiter.apply(sal)
        limiter.apply(temp)

        rho_anomaly.project(-beta_temp * (temp - T_ref) + beta_sal * (sal - S_ref))
        gradrho.project(Dx(rho_anomaly, mesh.geometric_dimension() - 1))
        #kappa_v.assign(conditional((gradrho / gradrho_scale) < 1e-1, 1e-3, 1e-1))
        step += 1
        t += dt

        #if step <= 10: 
        #    Vinflow.assign(0.01 * step * 0.1)
        
        with timed_stage('output'):
           if step % output_step == 0:
               # dumb checkpoint for starting from last timestep reached
               with DumbCheckpoint(folder+"dump.h5", mode=FILE_UPDATE) as chk:
                   # Checkpoint file open for reading and writing
                   chk.store(v_, name="velocity")
                   chk.store(p_, name="perturbation_pressure")
                   chk.store(temp, name="temperature")
                   chk.store(sal, name="salinity")

               # Update melt rate functions
               rho.interpolate(rho0*(1.0-beta_temp * (temp - T_ref) + beta_sal * (sal - S_ref)))
               Q_ice.interpolate(mp.Q_ice)
               Q_mixed.interpolate(mp.Q_mixed)
               Q_latent.interpolate(mp.Q_latent)
               Q_s.interpolate(mp.S_flux_bc)
               melt.interpolate(mp.wb)
               Tb.interpolate(mp.Tb)
               Sb.interpolate(mp.Sb)
               full_pressure.interpolate(mp.P_full)
        

               
               # Write out files
               if ADJOINT:
                   # Output adjoint T / S
                   tape.add_block(DiagnosticBlock(adj_t_file, temp))
                   tape.add_block(DiagnosticBlock(adj_s_file, sal))
               
               time_str = str(step)
        
               PETSc.Sys.Print("t=", t)
        
               PETSc.Sys.Print("integrated melt =", assemble(conditional(x < shelf_length, melt, 0.0) * ds("top")))
               if ADJOINT:
                   PETSc.Sys.Print("alternatively integrated melt =", assemble(melt * offset_backward_step_approx(x,k,x0) * ds("top")))

        if t % (3600 * 24) == 0:
            with DumbCheckpoint(folder+"dump_step_{}.h5".format(step), mode=FILE_CREATE) as chk:
                # Checkpoint file open for reading and writing at regular interval
                chk.store(v_, name="velocity")
                chk.store(p_, name="perturbation_pressure")
                chk.store(temp, name="temperature")
                chk.store(sal, name="salinity")

    if ADJOINT:
        melt.project(mp.wb)
        #J = assemble(conditional(x < shelf_length, mp.wb, 0.0) * ds("top"))
        J = assemble(sal**2 * dx)
        print(J)
        rf = ReducedFunctional(J, c)

        #tape.reset_variables()
        J.block_variable.adj_value = 1.0
        #tape.visualise()
        # evaluate all adjoint blocks to ensure we get complete adjoint solution
        # currently requires fix in dolfin_adjoint_common/blocks/solving.py:
        #    meshtype derivative (line 204) is broken, so just return None instead
        #with timed_stage('adjoint'):
        #    tape.evaluate_adj()
    #    grad = rf.derivative()
        #File(folder+'grad.pvd').write(grad)
        
        h = Function(sal)
        
        h.dat.data[:] = np.random.random(h.dat.data_ro.shape)
        print("hmax", h.dat.data_ro.max())
        print("hmin", h.dat.data_ro.min())
        print("sal max", sal.dat.data_ro.max())
        print("sal min", sal.dat.data_ro.min())
        
        print("J", J)
    #    print("rf(sal)", rf(sal))
    #    print("rf(salinit)", rf(sal_init_func))
    #    print("peturb rf", rf(sal+h))
        tt = taylor_test(rf, sal, h)
        assert np.allclose(tt, [2.0, 2.0, 2.0], rtol=5e-2)
