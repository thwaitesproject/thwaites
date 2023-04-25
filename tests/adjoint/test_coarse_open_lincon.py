# Buoyancy driven overturning circulation
# beneath ice shelf. Wedge geometry. 10km
from thwaites import *
from thwaites.utility import get_top_boundary, cavity_thickness
from firedrake.petsc import PETSc
from firedrake import FacetNormal
import pandas as pd
import argparse
import numpy as np
from pyop2.profiling import timed_stage

from firedrake_adjoint import *
from thwaites.diagnostic_block import DiagnosticBlock
from thwaites.diagnostic_block import DiagnosticConstantBlock
##########


def test_ice_shelf_coarse_open_lincon_Tslope():
    #nz = args.nz #10

    ip_factor = Constant(50.)
    #dt = 1.0
    restoring_time = 86400.

    ##########

    #  Generate mesh
    L = 10E3
    H1 = 2.
    H2 = 102.
    dy = 50.0
    ny = round(L/dy)
    #nz = 50
    dz = 1.0

    # create mesh
    mesh = Mesh("coarse.msh")

    PETSc.Sys.Print("Mesh dimension ", mesh.geometric_dimension())

    # shift z = 0 to surface of ocean. N.b z = 0 is outside domain.
    PETSc.Sys.Print("Length of lhs", assemble(Constant(1.0)*ds(1, domain=mesh)))

    PETSc.Sys.Print("Length of rhs", assemble(Constant(1.0)*ds(2, domain=mesh)))

    PETSc.Sys.Print("Length of bottom", assemble(Constant(1.0)*ds(3, domain=mesh)))

    PETSc.Sys.Print("Length of top", assemble(Constant(1.0)*ds(4, domain=mesh)))


    water_depth = 600.0
    mesh.coordinates.dat.data[:, 1] -= water_depth


    print("You have Comm WORLD size = ", mesh.comm.size)
    print("You have Comm WORLD rank = ", mesh.comm.rank)

    y, z = SpatialCoordinate(mesh)

    ##########

    # Set up function spaces
    V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
    W = FunctionSpace(mesh, "CG", 2)  # pressure space
    M = MixedFunctionSpace([V, W])

    # u velocity function space.
    U = FunctionSpace(mesh, "DG", 1)

    Q = FunctionSpace(mesh, "DG", 1)  # melt function space
    K = FunctionSpace(mesh, "DG", 1)    # temperature space
    S = FunctionSpace(mesh, "DG", 1)    # salinity space

    P1 = FunctionSpace(mesh, "CG", 1)
    print("vel dofs:", V.dim())
    print("pressure dofs:", W.dim())
    print("combined dofs:", M.dim())
    print("scalar dofs:", U.dim())
    print("P1 dofs (no. of nodes):", P1.dim())
    ##########

    # Set up functions
    m = Function(M)
    v_, p_ = m.split()  # function: y component of velocity, pressure
    v, p = split(m)  # expression: y component of velocity, pressure
    v_._name = "v_velocity"
    p_._name = "perturbation pressure"
    #u = Function(U, name="x velocity")  # x component of velocity

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

    ##########

    # Define a dump file
    dump_file = "./17.02.23_dump_50days_open_qadv_TSconst"

    DUMP = False #True
    if DUMP:
        with DumbCheckpoint(dump_file, mode=FILE_UPDATE) as chk:
            # Checkpoint file open for reading and writing
            chk.load(v_, name="v_velocity")
            chk.load(p_, name="perturbation_pressure")
            #chk.load(u, name="u_velocity")
    #        chk.load(sal, name="salinity")
    #        chk.load(temp, name="temperature")

            # from holland et al 2008b. constant T below 200m depth. varying sal.
            T_200m_depth = 1.0

            S_200m_depth = 34.4
            #S_bottom = 34.8
            #salinity_gradient = (S_bottom - S_200m_depth) / -H2
            #S_surface = S_200m_depth - (salinity_gradient * (H2 - water_depth))  # projected linear slope to surface.

            T_restore = Constant(T_200m_depth)
            S_restore = Constant(S_200m_depth) #S_surface + (S_bottom - S_surface) * (z / -water_depth)

            #T slope: 0.00042646953633639534
            #T intercept: 0.9797220017439936
            #S slope: -0.0006793874278126845
            #S intercept: 34.37197394968902
            T_slope = Constant(0.000426)
            T_intercept = Constant(0.9797)
            S_slope = Constant(-0.000679)
            S_intercept = Constant(34.37)
            
            temp.interpolate(T_intercept + T_slope * z)

            sal.interpolate(S_intercept + S_slope * z)


            # make T/S restore fields to calculate adjoint sensitity
            T_restorefield = Function(P1, name="Trestore field")
            S_restorefield = Function(P1, name="Srestore field")

            T_restorefield.assign(T_restore)
            S_restorefield.assign(S_restore)


    else:
        # Assign Initial conditions
        v_init = zero(mesh.geometric_dimension())
        v_.assign(v_init)

        #u_init = Constant(0.0)
        #u.interpolate(u_init)

        # ignore below was used to get 50 day run...
        T_200m_depth = -0.5
        T_bottom = 1.0
        temp_gradient = (T_bottom - T_200m_depth) / -H2
        T_surface = T_200m_depth - (temp_gradient * (H2 - water_depth))  # projected linear slope to surface.

        T_restore = T_surface + (T_bottom - T_surface) * (z / -water_depth)


        S_200m_depth = 34.4
        S_bottom = 34.8
        salinity_gradient = (S_bottom - S_200m_depth) / -H2
        S_surface = S_200m_depth - (salinity_gradient * (H2 - water_depth))  # projected linear slope to surface.
        S_restore = S_surface + (S_bottom - S_surface) * (z / -water_depth)
        
        T_restorefield = Function(P1, name="Trestore field")
        S_restorefield = Function(P1, name="Srestore field")

        T_restorefield.interpolate(T_restore)
        S_restorefield.interpolate(S_restore)

        # below is code for initialising and rhs bc...

        T_slope = Constant(0)
        T_intercept = Constant(1.0)
        S_slope = Constant(0)
        S_intercept = Constant(34.4)
            
        c = Control(T_slope) 
        c1 = Control(T_intercept) 
        c2 = Control(S_slope) 
        c3 = Control(S_intercept) 
        
        temp.interpolate(T_intercept + T_slope * z)
        sal.interpolate(S_intercept + S_slope * z)
    #    temp_init = T_restore
    #    temp.interpolate(temp_init)

    #    sal_init = Constant(34.4)
    #    sal_init = S_restore
    #    sal.interpolate(sal_init)
            #T_slope = Constant(0.0)
            #T_intercept = Constant(1.0)
            #S_slope = Constant(0.0)
            #S_intercept = Constant(34.0)
            
            #temp.interpolate(T_intercept + T_slope * z)

            #sal.interpolate(S_intercept + S_slope * z)
    #c = Control(temp)


    ##########

    # Set up equations
    mom_eq = MomentumEquation(M.sub(0), M.sub(0))
    cty_eq = ContinuityEquation(M.sub(1), M.sub(1))
    #u_eq = ScalarVelocity2halfDEquation(U, U)
    temp_eq = ScalarAdvectionDiffusionEquation(K, K)
    sal_eq = ScalarAdvectionDiffusionEquation(S, S)

    ##########

    # Terms for equation fields

    # momentum source: the buoyancy term Boussinesq approx. From mitgcm default
    T_ref = Constant(0.0)
    S_ref = Constant(35)
    beta_temp = Constant(2.0E-4)
    beta_sal = Constant(7.4E-4)
    g = Constant(9.81)
    mom_source = as_vector((0., -g))*(-beta_temp*(temp - T_ref) + beta_sal * (sal - S_ref))

    rho0 = 1030.
    rho.interpolate(rho0*(1.0-beta_temp * (temp - T_ref) + beta_sal * (sal - S_ref)))
    # coriolis frequency f-plane assumption at 75deg S. f = 2 omega sin (lat) = 2 * 7.2921E-5 * sin (-75 *2pi/360)
    #f = Constant(-1.409E-4)

    # Scalar source/sink terms at open boundary.
    absorption_factor = Constant(1.0/restoring_time)
    sponge_fraction = 0.2  # fraction of domain where sponge
    # Temperature source term
    source_temp = conditional(y > (1.0-sponge_fraction) * L,
                              absorption_factor * T_restorefield,
                              0.0)

    # Salinity source term
    source_sal = conditional(y > (1.0-sponge_fraction) * L,
                             absorption_factor * S_restorefield,
                             0.0)

    # Temperature absorption term
    absorp_temp = conditional(y > (1.0-sponge_fraction) * L,
                              absorption_factor,
                              0.0)

    # Salinity absorption term
    absorp_sal = conditional(y > (1.0-sponge_fraction) * L,
                             absorption_factor,
                             0.0)


    # linearly vary viscosity/diffusivity over domain. reduce vertical/diffusion
    kappa_h = Constant(0.25)
    kappa_v = Constant(0.001)
    #kappa_v = Constant(args.Kh*dz/dy)
    #grounding_line_kappa_v = Constant(open_ocean_kappa_v*H1/H2)
    #kappa_v_grad = (open_ocean_kappa_v-grounding_line_kappa_v)/L
    #kappa_v = grounding_line_kappa_v + y*kappa_v_grad

    #sponge_kappa_h = conditional(y > (1.0-sponge_fraction) * L,
    #                             1000. * kappa_h * ((y - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
    #                             kappa_h)

    #sponge_kappa_v = conditional(y > (1.0-sponge_fraction) * L,
    #                             1000. * kappa_v * ((y - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
    #                             kappa_v)

    kappa = as_tensor([[kappa_h, 0], [0, kappa_v]])
    #kappa = Constant([[kappa_h, 0], [0, kappa_v]]) # for taylor test need to use Constant for some reason...

    TP1 = TensorFunctionSpace(mesh, "CG", 1)
    kappa_temp = Function(TP1, name='temperature diffusion').project(kappa)
    kappa_sal = Function(TP1, name='salinity diffusion').project(kappa)
    mu = Function(TP1, name='viscosity').project(kappa)


    # Interior penalty term
    # 3*cot(min_angle)*(p+1)*p*nu_max/nu_min

    ip_alpha = Constant(3*dy/dz*2*ip_factor)
    # Equation fields
    vp_coupling = [{'pressure': 1}, {'velocity': 0}]
    vp_fields = {'viscosity': mu, 'source': mom_source, 'interior_penalty': ip_alpha}
    #u_fields = {'diffusivity': mu, 'velocity': v, 'interior_penalty': ip_alpha, 'coriolis_frequency': f}
    temp_fields = {'diffusivity': kappa_temp, 'velocity': v, 'interior_penalty': ip_alpha }#, 'source': source_temp,
    #               'absorption coefficient': absorp_temp}
    sal_fields = {'diffusivity': kappa_sal, 'velocity': v, 'interior_penalty': ip_alpha }#, 'source': source_sal,
     #             'absorption coefficient': absorp_sal}

    ##########

    # Get expressions used in melt rate parameterisation
    mp = ThreeEqMeltRateParam(sal, temp, p, z, velocity=pow(dot(v, v), 0.5), HJ99Gamma=True)

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

    # Plotting top boundary.
    shelf_boundary_points = get_top_boundary(cavity_length=L, cavity_height=H2, water_depth=water_depth)
    top_boundary_mp = pd.DataFrame()


    def top_boundary_to_csv(boundary_points, df, t_str):
        df['Qice_t_' + t_str] = Q_ice.at(boundary_points)
        df['Qmixed_t_' + t_str] = Q_mixed.at(boundary_points)
        df['Qlat_t_' + t_str] = Q_latent.at(boundary_points)
        df['Qsalt_t_' + t_str] = Q_s.at(boundary_points)
        df['Melt_t' + t_str] = melt.at(boundary_points)
        df['Tb_t_' + t_str] = Tb.at(boundary_points)
        df['P_t_' + t_str] = full_pressure.at(boundary_points)
        df['Sal_t_' + t_str] = sal.at(boundary_points)
        df['Temp_t_' + t_str] = temp.at(boundary_points)
        df["integrated_melt_t_ " + t_str] = assemble(melt * ds(4))

        if mesh.comm.rank == 0:
            top_boundary_mp.to_csv(folder+"top_boundary_data.csv")


    ##########

    # Boundary conditions
    # top boundary: no normal flow, drag flowing over ice
    # bottom boundary: no normal flow, drag flowing over bedrock
    # grounding line wall (LHS): no normal flow
    # open ocean (RHS): pressure to account for density differences

    # WEAKLY Enforced BCs
    n = FacetNormal(mesh)
    Temperature_term = -beta_temp * (0.5 * T_slope * pow(z, 2)  + (T_intercept-T_ref) * z)
    Salinity_term = beta_sal * (0.5 * S_slope * pow(z, 2)  + (S_intercept-S_ref) * z)
    stress_open_boundary = -n*-g*(Temperature_term + Salinity_term)
    no_normal_flow = 0.
    ice_drag = 0.0097

    import scipy



    z_rhs = np.linspace(-498.01, -599.99, 400)
    rhs_nodes = []
    for zi in z_rhs:
        rhs_nodes.append([9999.995, zi])

    print(rhs_nodes)
    rhs_nodes = VertexOnlyMesh(mesh,rhs_nodes,missing_points_behaviour='warn')

    DG0_vom = FunctionSpace(rhs_nodes, "DG", 0)
    temp_vom = Function(DG0_vom)
    sal_vom = Function(DG0_vom)

    temp_vom.interpolate(temp)
    sal_vom.interpolate(sal)

    print(sal_vom.dat.data)

    p_rhs = []
    def dynamic_pressure(T, S):

    # because flow is out the domain with positive slope dont do any modification of sal/temp.

     #   for i in range(10):
    #        S[i] -= 0.1*(10-i)*2e-4
    #    S[:10] -= 0.00016  # try making the top of the domain a bit fresher as a hack to get faster flow at the rhs...
    #    for i in range(10, len(T)):  
    #        S[i] = 34
    #        T[i] = -0.5
       
        density_rhs =  -g.values()[0] *(-beta_temp.values()[0] * (T-T_ref.values()[0]) +  beta_sal.values()[0] * (S - S_ref.values()[0]))
      #  print(density_rhs)



    #    print("len z", len(z_rhs))
    #    print("len density", len(density_rhs))
        pcumulative = scipy.integrate.cumulative_trapezoid(density_rhs, z_rhs, initial=0)
     #   print("pcumulative", pcumulative)
     #   print("len pcumulative", len(pcumulative))

        return pcumulative

    p_dyn = dynamic_pressure(temp_vom.dat.data, sal_vom.dat.data)
    dynamic_pressure_interp = scipy.interpolate.interp1d(z_rhs, p_dyn, bounds_error=False,fill_value='extrapolate') 


    # Now make the VectorFunctionSpace corresponding to V.
    W_vector = VectorFunctionSpace(mesh, W.ufl_element())

    # Next, interpolate the coordinates onto the nodes of W.
    X = interpolate(mesh.coordinates, W_vector)
    print("X[:,1]", X.dat.data[:,1])
    # Make an output function.
    stress_open_boundary_dynamic = Function(W)

    # Use the external data function to interpolate the values of f.
    stress_open_boundary_dynamic.dat.data[:] = dynamic_pressure_interp(X.dat.data_ro[:,1])

    # test stress open_boundary
    #sop = Function(W)
    #sop.interpolate(-g*(Temperature_term + Salinity_term))
    #sop_file = File(folder+"boundary_stress.pvd")
    #sop_file.write(sop)


    vp_bcs = {4: {'un': no_normal_flow, 'drag': ice_drag}, 2: {'stress': stress_open_boundary},
              3: {'un': no_normal_flow, 'drag': 0.0025}, 1: {'un': no_normal_flow}}
    #u_bcs = {2: {'q': Constant(0.0)}}

    temp_bcs = {4: {'flux': -mp.T_flux_bc}, 2: {'qadv': T_intercept + T_slope * z}}

    sal_bcs = {4: {'flux': -mp.S_flux_bc}, 2: {'qadv': S_intercept + S_slope * z}}



    # STRONGLY Enforced BCs
    # open ocean (RHS): no tangential flow because viscosity of outside ocean resists vertical flow.
    strong_bcs = []#DirichletBC(M.sub(0).sub(1), 0, 2)]

    ##########

    # Solver parameters
    mumps_solver_parameters = {
        'snes_monitor': None,
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij',
        'snes_max_it': 100,
        'snes_rtol': 1e-8,
        'snes_atol': 1e-6,
    }

    vp_solver_parameters = mumps_solver_parameters
    u_solver_parameters = mumps_solver_parameters
    temp_solver_parameters = mumps_solver_parameters
    sal_solver_parameters = mumps_solver_parameters

    ##########

    # Plotting depth profiles.
    z500m = cavity_thickness(5E2, 0., H1, L, H2)
    z1km = cavity_thickness(1E3, 0., H1, L, H2)
    z2km = cavity_thickness(2E3, 0., H1, L, H2)
    z4km = cavity_thickness(4E3, 0., H1, L, H2)
    z6km = cavity_thickness(6E3, 0., H1, L, H2)


    z_profile500m = np.linspace(z500m-water_depth-1., 1.-water_depth, 50)
    z_profile1km = np.linspace(z1km-water_depth-1., 1.-water_depth, 50)
    z_profile2km = np.linspace(z2km-water_depth-1., 1.-water_depth, 50)
    z_profile4km = np.linspace(z4km-water_depth-1., 1.-water_depth, 50)
    z_profile6km = np.linspace(z6km-water_depth-1., 1.-water_depth, 50)


    depth_profile500m = []
    depth_profile1km = []
    depth_profile2km = []
    depth_profile4km = []
    depth_profile6km = []

    for d5e2, d1km, d2km, d4km, d6km in zip(z_profile500m, z_profile1km, z_profile2km, z_profile4km, z_profile6km):
        depth_profile500m.append([5E2, d5e2])
        depth_profile1km.append([1E3, d1km])
        depth_profile2km.append([2E3, d2km])
        depth_profile4km.append([4E3, d4km])
        depth_profile6km.append([6E3, d6km])

    velocity_depth_profile500m = pd.DataFrame()
    velocity_depth_profile1km = pd.DataFrame()
    velocity_depth_profile2km = pd.DataFrame()
    velocity_depth_profile4km = pd.DataFrame()
    velocity_depth_profile6km = pd.DataFrame()

    velocity_depth_profile500m['Z_profile'] = z_profile500m
    velocity_depth_profile1km['Z_profile'] = z_profile1km
    velocity_depth_profile2km['Z_profile'] = z_profile2km
    velocity_depth_profile4km['Z_profile'] = z_profile4km
    velocity_depth_profile6km['Z_profile'] = z_profile6km


    def depth_profile_to_csv(profile, df, depth, t_str):
        #df['U_t_' + t_str] = u.at(profile)
        vw = np.array(v_.at(profile))
        vv = vw[:, 0]
        ww = vw[:, 1]
        df['V_t_' + t_str] = vv
        df['W_t_' + t_str] = ww
        if mesh.comm.rank == 0:
            df.to_csv(folder+depth+"_profile.csv")

    #### VOM for doing objective function...

    z_profile_adjoint = np.linspace(-525, -595, 15)
    z_profile_nodes = []
    for zi in z_profile_adjoint:
        z_profile_nodes.append([7550, zi])

    print(z_profile_nodes)
    z_profile_nodes = VertexOnlyMesh(mesh,z_profile_nodes,missing_points_behaviour='warn')

    DG0_vom_profile = FunctionSpace(z_profile_nodes, "DG", 0)
    temp_vom_profile = Function(DG0_vom_profile)
    sal_vom_profile = Function(DG0_vom_profile)

    temp_vom_profile.interpolate(temp)
    sal_vom_profile.interpolate(sal)


    adjoint_profile7550m = pd.DataFrame()
    adjoint_profile7550m['Z_profile'] = z_profile_adjoint

    melt_node = VertexOnlyMesh(mesh, [[7550, -522.7]], missing_points_behaviour='warn')
    DG0_vom_meltnode = FunctionSpace(melt_node, "DG", 0)
    melt_vom_node = Function(DG0_vom_meltnode)

    melt_vom_node.interpolate(mp.wb)

    def adjoint_profile_to_csv(df, t_str):
        df['T_t_' + t_str] = temp_vom_profile.dat.data_ro
        df['S_t_' + t_str] = sal_vom_profile.dat.data_ro
        if mesh.comm.rank == 0:
            df.to_csv(folder+"TS_foradjoint_7550m_profile.csv")

    adjoint_melt7550m = pd.DataFrame()
    def adjoint_melt_to_csv(df, t_str):
        df['Melt_t_' + t_str] = melt_vom_node.dat.data_ro
        if mesh.comm.rank == 0:
            df.to_csv(folder+"melt_foradjoint_7550m_profile.csv")

    ##### 
    # read linear simulations
    #target_folder = "/data/2d_adjoint/17.02.23_3_eq_param_ufric_dt300.0_dtOutput86400.0_T4320000.0_ip50.0_tres86400.0constant_Kh0.25_Kv0.001_structured_dy500_dz2_no_limiter_nosponge_open_qadvbc_pdyn_linTS/"
    adjoint_profile7550m_target = pd.read_csv("TS_foradjoint_7550m_profile_250423.csv")

    temp_profile_target = adjoint_profile7550m_target['T_t_14400']
    sal_profile_target = adjoint_profile7550m_target['S_t_14400']


    temp_vom_profile_target = Function(DG0_vom_profile)
    sal_vom_profile_target = Function(DG0_vom_profile)

    temp_vom_profile_target.dat.data[:] = temp_profile_target[:]
    sal_vom_profile_target.dat.data[:] = sal_profile_target[:]


    print("temp vom", temp_vom_profile.dat.data[:])

    print("temp vom target", temp_vom_profile_target.dat.data[:])

    ##########

    # define time steps
    dt = 300.0 
    T = 3000.0

    ##########

    # Set up time stepping routines
    vp_timestepper = CrankNicolsonSaddlePointTimeIntegrator([mom_eq, cty_eq], m, vp_fields, vp_coupling, dt, vp_bcs,
                                                            solver_parameters=vp_solver_parameters, strong_bcs=strong_bcs)

    #u_timestepper = DIRK33(u_eq, u, u_fields, dt, u_bcs, solver_parameters=u_solver_parameters)
    temp_timestepper = DIRK33(temp_eq, temp, temp_fields, dt, temp_bcs, solver_parameters=temp_solver_parameters)
    sal_timestepper = DIRK33(sal_eq, sal, sal_fields, dt, sal_bcs, solver_parameters=sal_solver_parameters)

    ##########

    # Set up folder



    # Depth profiles
    #depth_profile_to_csv(depth_profile500m, velocity_depth_profile500m, "500m", '0.0')
    #depth_profile_to_csv(depth_profile1km, velocity_depth_profile1km, "1km", '0.0')
    #depth_profile_to_csv(depth_profile2km, velocity_depth_profile2km, "2km", '0.0')
    #depth_profile_to_csv(depth_profile4km, velocity_depth_profile4km, "4km", '0.0')
    #depth_profile_to_csv(depth_profile6km, velocity_depth_profile6km, "6km", '0.0')

    ########


    # Begin time stepping
    t = 0.0
    step = 0


    while t < T - 0.5*dt:
        with timed_stage('velocity-pressure'):
            vp_timestepper.advance(t)
            #u_timestepper.advance(t)
        with timed_stage('temperature'):
            temp_timestepper.advance(t)
        with timed_stage('salinity'):
            sal_timestepper.advance(t)
        
        # dynamic pressure rhs
        temp_vom.interpolate(temp)
        sal_vom.interpolate(sal)
        p_dyn = dynamic_pressure(temp_vom.dat.data, sal_vom.dat.data)
        dynamic_pressure_interp = scipy.interpolate.interp1d(z_rhs, p_dyn, bounds_error=False,fill_value='extrapolate') 

        stress_open_boundary_dynamic.dat.data[:] = dynamic_pressure_interp(X.dat.data_ro[:,1])
        
        step += 1
        t += dt

    melt.project(mp.wb)
    #J = assemble(mp.wb*ds(4))
    N = len(temp_vom_profile.dat.data_ro)
    temp_vom_profile.interpolate(temp)
    sal_vom_profile.interpolate(sal)

    scale = Constant(1.0 / (0.5 * beta_sal)**2)
    alpha = 0.1
    J = assemble(scale*0.5/N * ((-beta_temp*(temp_vom_profile - temp_vom_profile_target))**2 + (beta_sal*(sal_vom_profile - sal_vom_profile_target))**2) * dx)  #+ assemble(0.5*alpha*scale*((beta_sal**2)*inner(grad(S_restorefield), grad(S_restorefield))+ (beta_temp**2)*inner(grad(T_restorefield), grad(T_restorefield)) )*dx)


    def eval_cb_pre(m):
        print("eval_cb_pre")
        print(len(m))
        print("T slope:", m[0].values()[0])
        print("T intercept:", m[1].values()[0])
        print("S slope:", m[2].values()[0])
        print("S intercept:", m[3].values()[0])
        tape.add_block(DiagnosticConstantBlock(T_slope, "T slope :"))
        tape.add_block(DiagnosticConstantBlock(T_intercept, "T intercept :"))
        tape.add_block(DiagnosticConstantBlock(S_slope, "S slope :"))
        tape.add_block(DiagnosticConstantBlock(S_intercept, "S intercept :"))


    def eval_cb(j, m):
        print("eval_cb")

        print(len(m))
        print("T slope:", m[0].values()[0])
        print("T intercept:", m[1].values()[0])
        print("S slope:", m[2].values()[0])
        print("S intercept:", m[3].values()[0])

        print("J = ", j)
        with DumbCheckpoint(folder+"dump.h5", mode=FILE_UPDATE) as chk:
            # Checkpoint file open for reading and writing
            chk.store(v_, name="v_velocity")
            chk.store(p_, name="perturbation_pressure")
            #chk.store(u, name="u_velocity")
            chk.store(temp, name="temperature")
            chk.store(sal, name="salinity")
            chk.store(T_restorefield_vis, name="temp restore")
            chk.store(S_restorefield_vis, name="sal restore")
    print(J)

    def derivative_cb_post(j, djdm, m):
        print("------------")
        print(" derivative cb post")
        print("------------")

    rf = ReducedFunctional(J, [c]) #, c1,c2,c3], eval_cb_post=eval_cb, eval_cb_pre=eval_cb_pre, derivative_cb_post=derivative_cb_post)

    bounds = [[-0.02, -10, -0.01, 30],[0.02, 10., 0., 40. ] ]

    #g_opt = minimize(rf, bounds=bounds, options={"disp": True})

    #tape.reset_variables()
    #J.adj_value = 1.0

    J.block_variable.adj_value = 1.0
    #tape.visualise()
    # evaluate all adjoint blocks to ensure we get complete adjoint solution
    # currently requires fix in dolfin_adjoint_common/blocks/solving.py:
    #    meshtype derivative (line 204) is broken, so just return None instead
    #with timed_stage('adjoint'):
     #  tape.evaluate_adj()

    print(len(T_restorefield.dat.data))

    #grad = rf.derivative()
    #File(folder+'grad.pvd').write(grad)

    h = Constant(T_slope)#S_restorefield)
    h.dat.data[:] = np.random.random(h.dat.data_ro.shape) # 2* for temp... 
    tt = taylor_test(rf, T_slope, h)
    print("Tslope TT ten steps")

    print(tt)
    assert np.allclose(tt, [2.0, 2.0, 2.0], rtol=5e-2)


def test_ice_shelf_coarse_open_lincon_Tintercept():
    #nz = args.nz #10

    ip_factor = Constant(50.)
    #dt = 1.0
    restoring_time = 86400.

    ##########

    #  Generate mesh
    L = 10E3
    H1 = 2.
    H2 = 102.
    dy = 50.0
    ny = round(L/dy)
    #nz = 50
    dz = 1.0

    # create mesh
    mesh = Mesh("coarse.msh")

    PETSc.Sys.Print("Mesh dimension ", mesh.geometric_dimension())

    # shift z = 0 to surface of ocean. N.b z = 0 is outside domain.
    PETSc.Sys.Print("Length of lhs", assemble(Constant(1.0)*ds(1, domain=mesh)))

    PETSc.Sys.Print("Length of rhs", assemble(Constant(1.0)*ds(2, domain=mesh)))

    PETSc.Sys.Print("Length of bottom", assemble(Constant(1.0)*ds(3, domain=mesh)))

    PETSc.Sys.Print("Length of top", assemble(Constant(1.0)*ds(4, domain=mesh)))


    water_depth = 600.0
    mesh.coordinates.dat.data[:, 1] -= water_depth


    print("You have Comm WORLD size = ", mesh.comm.size)
    print("You have Comm WORLD rank = ", mesh.comm.rank)

    y, z = SpatialCoordinate(mesh)

    ##########

    # Set up function spaces
    V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
    W = FunctionSpace(mesh, "CG", 2)  # pressure space
    M = MixedFunctionSpace([V, W])

    # u velocity function space.
    U = FunctionSpace(mesh, "DG", 1)

    Q = FunctionSpace(mesh, "DG", 1)  # melt function space
    K = FunctionSpace(mesh, "DG", 1)    # temperature space
    S = FunctionSpace(mesh, "DG", 1)    # salinity space

    P1 = FunctionSpace(mesh, "CG", 1)
    print("vel dofs:", V.dim())
    print("pressure dofs:", W.dim())
    print("combined dofs:", M.dim())
    print("scalar dofs:", U.dim())
    print("P1 dofs (no. of nodes):", P1.dim())
    ##########

    # Set up functions
    m = Function(M)
    v_, p_ = m.split()  # function: y component of velocity, pressure
    v, p = split(m)  # expression: y component of velocity, pressure
    v_._name = "v_velocity"
    p_._name = "perturbation pressure"
    #u = Function(U, name="x velocity")  # x component of velocity

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

    ##########

    # Define a dump file
    dump_file = "./17.02.23_dump_50days_open_qadv_TSconst"

    DUMP = False #True
    if DUMP:
        with DumbCheckpoint(dump_file, mode=FILE_UPDATE) as chk:
            # Checkpoint file open for reading and writing
            chk.load(v_, name="v_velocity")
            chk.load(p_, name="perturbation_pressure")
            #chk.load(u, name="u_velocity")
    #        chk.load(sal, name="salinity")
    #        chk.load(temp, name="temperature")

            # from holland et al 2008b. constant T below 200m depth. varying sal.
            T_200m_depth = 1.0

            S_200m_depth = 34.4
            #S_bottom = 34.8
            #salinity_gradient = (S_bottom - S_200m_depth) / -H2
            #S_surface = S_200m_depth - (salinity_gradient * (H2 - water_depth))  # projected linear slope to surface.

            T_restore = Constant(T_200m_depth)
            S_restore = Constant(S_200m_depth) #S_surface + (S_bottom - S_surface) * (z / -water_depth)

            #T slope: 0.00042646953633639534
            #T intercept: 0.9797220017439936
            #S slope: -0.0006793874278126845
            #S intercept: 34.37197394968902
            T_slope = Constant(0.000426)
            T_intercept = Constant(0.9797)
            S_slope = Constant(-0.000679)
            S_intercept = Constant(34.37)
            
            temp.interpolate(T_intercept + T_slope * z)

            sal.interpolate(S_intercept + S_slope * z)


            # make T/S restore fields to calculate adjoint sensitity
            T_restorefield = Function(P1, name="Trestore field")
            S_restorefield = Function(P1, name="Srestore field")

            T_restorefield.assign(T_restore)
            S_restorefield.assign(S_restore)


    else:
        # Assign Initial conditions
        v_init = zero(mesh.geometric_dimension())
        v_.assign(v_init)

        #u_init = Constant(0.0)
        #u.interpolate(u_init)

        # ignore below was used to get 50 day run...
        T_200m_depth = -0.5
        T_bottom = 1.0
        temp_gradient = (T_bottom - T_200m_depth) / -H2
        T_surface = T_200m_depth - (temp_gradient * (H2 - water_depth))  # projected linear slope to surface.

        T_restore = T_surface + (T_bottom - T_surface) * (z / -water_depth)


        S_200m_depth = 34.4
        S_bottom = 34.8
        salinity_gradient = (S_bottom - S_200m_depth) / -H2
        S_surface = S_200m_depth - (salinity_gradient * (H2 - water_depth))  # projected linear slope to surface.
        S_restore = S_surface + (S_bottom - S_surface) * (z / -water_depth)
        
        T_restorefield = Function(P1, name="Trestore field")
        S_restorefield = Function(P1, name="Srestore field")

        T_restorefield.interpolate(T_restore)
        S_restorefield.interpolate(S_restore)

        # below is code for initialising and rhs bc...

        T_slope = Constant(0)
        T_intercept = Constant(1.0)
        S_slope = Constant(0)
        S_intercept = Constant(34.4)
            
        c = Control(T_slope) 
        c1 = Control(T_intercept) 
        c2 = Control(S_slope) 
        c3 = Control(S_intercept) 
        
        temp.interpolate(T_intercept + T_slope * z)
        sal.interpolate(S_intercept + S_slope * z)
    #    temp_init = T_restore
    #    temp.interpolate(temp_init)

    #    sal_init = Constant(34.4)
    #    sal_init = S_restore
    #    sal.interpolate(sal_init)
            #T_slope = Constant(0.0)
            #T_intercept = Constant(1.0)
            #S_slope = Constant(0.0)
            #S_intercept = Constant(34.0)
            
            #temp.interpolate(T_intercept + T_slope * z)

            #sal.interpolate(S_intercept + S_slope * z)
    #c = Control(temp)


    ##########

    # Set up equations
    mom_eq = MomentumEquation(M.sub(0), M.sub(0))
    cty_eq = ContinuityEquation(M.sub(1), M.sub(1))
    #u_eq = ScalarVelocity2halfDEquation(U, U)
    temp_eq = ScalarAdvectionDiffusionEquation(K, K)
    sal_eq = ScalarAdvectionDiffusionEquation(S, S)

    ##########

    # Terms for equation fields

    # momentum source: the buoyancy term Boussinesq approx. From mitgcm default
    T_ref = Constant(0.0)
    S_ref = Constant(35)
    beta_temp = Constant(2.0E-4)
    beta_sal = Constant(7.4E-4)
    g = Constant(9.81)
    mom_source = as_vector((0., -g))*(-beta_temp*(temp - T_ref) + beta_sal * (sal - S_ref))

    rho0 = 1030.
    rho.interpolate(rho0*(1.0-beta_temp * (temp - T_ref) + beta_sal * (sal - S_ref)))
    # coriolis frequency f-plane assumption at 75deg S. f = 2 omega sin (lat) = 2 * 7.2921E-5 * sin (-75 *2pi/360)
    #f = Constant(-1.409E-4)

    # Scalar source/sink terms at open boundary.
    absorption_factor = Constant(1.0/restoring_time)
    sponge_fraction = 0.2  # fraction of domain where sponge
    # Temperature source term
    source_temp = conditional(y > (1.0-sponge_fraction) * L,
                              absorption_factor * T_restorefield,
                              0.0)

    # Salinity source term
    source_sal = conditional(y > (1.0-sponge_fraction) * L,
                             absorption_factor * S_restorefield,
                             0.0)

    # Temperature absorption term
    absorp_temp = conditional(y > (1.0-sponge_fraction) * L,
                              absorption_factor,
                              0.0)

    # Salinity absorption term
    absorp_sal = conditional(y > (1.0-sponge_fraction) * L,
                             absorption_factor,
                             0.0)


    # linearly vary viscosity/diffusivity over domain. reduce vertical/diffusion
    kappa_h = Constant(0.25)
    kappa_v = Constant(0.001)
    #kappa_v = Constant(args.Kh*dz/dy)
    #grounding_line_kappa_v = Constant(open_ocean_kappa_v*H1/H2)
    #kappa_v_grad = (open_ocean_kappa_v-grounding_line_kappa_v)/L
    #kappa_v = grounding_line_kappa_v + y*kappa_v_grad

    #sponge_kappa_h = conditional(y > (1.0-sponge_fraction) * L,
    #                             1000. * kappa_h * ((y - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
    #                             kappa_h)

    #sponge_kappa_v = conditional(y > (1.0-sponge_fraction) * L,
    #                             1000. * kappa_v * ((y - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
    #                             kappa_v)

    kappa = as_tensor([[kappa_h, 0], [0, kappa_v]])
    #kappa = Constant([[kappa_h, 0], [0, kappa_v]]) # for taylor test need to use Constant for some reason...

    TP1 = TensorFunctionSpace(mesh, "CG", 1)
    kappa_temp = Function(TP1, name='temperature diffusion').project(kappa)
    kappa_sal = Function(TP1, name='salinity diffusion').project(kappa)
    mu = Function(TP1, name='viscosity').project(kappa)


    # Interior penalty term
    # 3*cot(min_angle)*(p+1)*p*nu_max/nu_min

    ip_alpha = Constant(3*dy/dz*2*ip_factor)
    # Equation fields
    vp_coupling = [{'pressure': 1}, {'velocity': 0}]
    vp_fields = {'viscosity': mu, 'source': mom_source, 'interior_penalty': ip_alpha}
    #u_fields = {'diffusivity': mu, 'velocity': v, 'interior_penalty': ip_alpha, 'coriolis_frequency': f}
    temp_fields = {'diffusivity': kappa_temp, 'velocity': v, 'interior_penalty': ip_alpha }#, 'source': source_temp,
    #               'absorption coefficient': absorp_temp}
    sal_fields = {'diffusivity': kappa_sal, 'velocity': v, 'interior_penalty': ip_alpha }#, 'source': source_sal,
     #             'absorption coefficient': absorp_sal}

    ##########

    # Get expressions used in melt rate parameterisation
    mp = ThreeEqMeltRateParam(sal, temp, p, z, velocity=pow(dot(v, v), 0.5), HJ99Gamma=True)

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

    # Plotting top boundary.
    shelf_boundary_points = get_top_boundary(cavity_length=L, cavity_height=H2, water_depth=water_depth)
    top_boundary_mp = pd.DataFrame()


    def top_boundary_to_csv(boundary_points, df, t_str):
        df['Qice_t_' + t_str] = Q_ice.at(boundary_points)
        df['Qmixed_t_' + t_str] = Q_mixed.at(boundary_points)
        df['Qlat_t_' + t_str] = Q_latent.at(boundary_points)
        df['Qsalt_t_' + t_str] = Q_s.at(boundary_points)
        df['Melt_t' + t_str] = melt.at(boundary_points)
        df['Tb_t_' + t_str] = Tb.at(boundary_points)
        df['P_t_' + t_str] = full_pressure.at(boundary_points)
        df['Sal_t_' + t_str] = sal.at(boundary_points)
        df['Temp_t_' + t_str] = temp.at(boundary_points)
        df["integrated_melt_t_ " + t_str] = assemble(melt * ds(4))

        if mesh.comm.rank == 0:
            top_boundary_mp.to_csv(folder+"top_boundary_data.csv")


    ##########

    # Boundary conditions
    # top boundary: no normal flow, drag flowing over ice
    # bottom boundary: no normal flow, drag flowing over bedrock
    # grounding line wall (LHS): no normal flow
    # open ocean (RHS): pressure to account for density differences

    # WEAKLY Enforced BCs
    n = FacetNormal(mesh)
    Temperature_term = -beta_temp * (0.5 * T_slope * pow(z, 2)  + (T_intercept-T_ref) * z)
    Salinity_term = beta_sal * (0.5 * S_slope * pow(z, 2)  + (S_intercept-S_ref) * z)
    stress_open_boundary = -n*-g*(Temperature_term + Salinity_term)
    no_normal_flow = 0.
    ice_drag = 0.0097

    import scipy



    z_rhs = np.linspace(-498.01, -599.99, 400)
    rhs_nodes = []
    for zi in z_rhs:
        rhs_nodes.append([9999.995, zi])

    print(rhs_nodes)
    rhs_nodes = VertexOnlyMesh(mesh,rhs_nodes,missing_points_behaviour='warn')

    DG0_vom = FunctionSpace(rhs_nodes, "DG", 0)
    temp_vom = Function(DG0_vom)
    sal_vom = Function(DG0_vom)

    temp_vom.interpolate(temp)
    sal_vom.interpolate(sal)

    print(sal_vom.dat.data)

    p_rhs = []
    def dynamic_pressure(T, S):

    # because flow is out the domain with positive slope dont do any modification of sal/temp.

     #   for i in range(10):
    #        S[i] -= 0.1*(10-i)*2e-4
    #    S[:10] -= 0.00016  # try making the top of the domain a bit fresher as a hack to get faster flow at the rhs...
    #    for i in range(10, len(T)):  
    #        S[i] = 34
    #        T[i] = -0.5
       
        density_rhs =  -g.values()[0] *(-beta_temp.values()[0] * (T-T_ref.values()[0]) +  beta_sal.values()[0] * (S - S_ref.values()[0]))
      #  print(density_rhs)



    #    print("len z", len(z_rhs))
    #    print("len density", len(density_rhs))
        pcumulative = scipy.integrate.cumulative_trapezoid(density_rhs, z_rhs, initial=0)
     #   print("pcumulative", pcumulative)
     #   print("len pcumulative", len(pcumulative))

        return pcumulative

    p_dyn = dynamic_pressure(temp_vom.dat.data, sal_vom.dat.data)
    dynamic_pressure_interp = scipy.interpolate.interp1d(z_rhs, p_dyn, bounds_error=False,fill_value='extrapolate') 


    # Now make the VectorFunctionSpace corresponding to V.
    W_vector = VectorFunctionSpace(mesh, W.ufl_element())

    # Next, interpolate the coordinates onto the nodes of W.
    X = interpolate(mesh.coordinates, W_vector)
    print("X[:,1]", X.dat.data[:,1])
    # Make an output function.
    stress_open_boundary_dynamic = Function(W)

    # Use the external data function to interpolate the values of f.
    stress_open_boundary_dynamic.dat.data[:] = dynamic_pressure_interp(X.dat.data_ro[:,1])

    # test stress open_boundary
    #sop = Function(W)
    #sop.interpolate(-g*(Temperature_term + Salinity_term))
    #sop_file = File(folder+"boundary_stress.pvd")
    #sop_file.write(sop)


    vp_bcs = {4: {'un': no_normal_flow, 'drag': ice_drag}, 2: {'stress': stress_open_boundary},
              3: {'un': no_normal_flow, 'drag': 0.0025}, 1: {'un': no_normal_flow}}
    #u_bcs = {2: {'q': Constant(0.0)}}

    temp_bcs = {4: {'flux': -mp.T_flux_bc}, 2: {'qadv': T_intercept + T_slope * z}}

    sal_bcs = {4: {'flux': -mp.S_flux_bc}, 2: {'qadv': S_intercept + S_slope * z}}



    # STRONGLY Enforced BCs
    # open ocean (RHS): no tangential flow because viscosity of outside ocean resists vertical flow.
    strong_bcs = []#DirichletBC(M.sub(0).sub(1), 0, 2)]

    ##########

    # Solver parameters
    mumps_solver_parameters = {
        'snes_monitor': None,
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij',
        'snes_max_it': 100,
        'snes_rtol': 1e-8,
        'snes_atol': 1e-6,
    }

    vp_solver_parameters = mumps_solver_parameters
    u_solver_parameters = mumps_solver_parameters
    temp_solver_parameters = mumps_solver_parameters
    sal_solver_parameters = mumps_solver_parameters

    ##########

    # Plotting depth profiles.
    z500m = cavity_thickness(5E2, 0., H1, L, H2)
    z1km = cavity_thickness(1E3, 0., H1, L, H2)
    z2km = cavity_thickness(2E3, 0., H1, L, H2)
    z4km = cavity_thickness(4E3, 0., H1, L, H2)
    z6km = cavity_thickness(6E3, 0., H1, L, H2)


    z_profile500m = np.linspace(z500m-water_depth-1., 1.-water_depth, 50)
    z_profile1km = np.linspace(z1km-water_depth-1., 1.-water_depth, 50)
    z_profile2km = np.linspace(z2km-water_depth-1., 1.-water_depth, 50)
    z_profile4km = np.linspace(z4km-water_depth-1., 1.-water_depth, 50)
    z_profile6km = np.linspace(z6km-water_depth-1., 1.-water_depth, 50)


    depth_profile500m = []
    depth_profile1km = []
    depth_profile2km = []
    depth_profile4km = []
    depth_profile6km = []

    for d5e2, d1km, d2km, d4km, d6km in zip(z_profile500m, z_profile1km, z_profile2km, z_profile4km, z_profile6km):
        depth_profile500m.append([5E2, d5e2])
        depth_profile1km.append([1E3, d1km])
        depth_profile2km.append([2E3, d2km])
        depth_profile4km.append([4E3, d4km])
        depth_profile6km.append([6E3, d6km])

    velocity_depth_profile500m = pd.DataFrame()
    velocity_depth_profile1km = pd.DataFrame()
    velocity_depth_profile2km = pd.DataFrame()
    velocity_depth_profile4km = pd.DataFrame()
    velocity_depth_profile6km = pd.DataFrame()

    velocity_depth_profile500m['Z_profile'] = z_profile500m
    velocity_depth_profile1km['Z_profile'] = z_profile1km
    velocity_depth_profile2km['Z_profile'] = z_profile2km
    velocity_depth_profile4km['Z_profile'] = z_profile4km
    velocity_depth_profile6km['Z_profile'] = z_profile6km


    def depth_profile_to_csv(profile, df, depth, t_str):
        #df['U_t_' + t_str] = u.at(profile)
        vw = np.array(v_.at(profile))
        vv = vw[:, 0]
        ww = vw[:, 1]
        df['V_t_' + t_str] = vv
        df['W_t_' + t_str] = ww
        if mesh.comm.rank == 0:
            df.to_csv(folder+depth+"_profile.csv")

    #### VOM for doing objective function...

    z_profile_adjoint = np.linspace(-525, -595, 15)
    z_profile_nodes = []
    for zi in z_profile_adjoint:
        z_profile_nodes.append([7550, zi])

    print(z_profile_nodes)
    z_profile_nodes = VertexOnlyMesh(mesh,z_profile_nodes,missing_points_behaviour='warn')

    DG0_vom_profile = FunctionSpace(z_profile_nodes, "DG", 0)
    temp_vom_profile = Function(DG0_vom_profile)
    sal_vom_profile = Function(DG0_vom_profile)

    temp_vom_profile.interpolate(temp)
    sal_vom_profile.interpolate(sal)


    adjoint_profile7550m = pd.DataFrame()
    adjoint_profile7550m['Z_profile'] = z_profile_adjoint

    melt_node = VertexOnlyMesh(mesh, [[7550, -522.7]], missing_points_behaviour='warn')
    DG0_vom_meltnode = FunctionSpace(melt_node, "DG", 0)
    melt_vom_node = Function(DG0_vom_meltnode)

    melt_vom_node.interpolate(mp.wb)

    def adjoint_profile_to_csv(df, t_str):
        df['T_t_' + t_str] = temp_vom_profile.dat.data_ro
        df['S_t_' + t_str] = sal_vom_profile.dat.data_ro
        if mesh.comm.rank == 0:
            df.to_csv(folder+"TS_foradjoint_7550m_profile.csv")

    adjoint_melt7550m = pd.DataFrame()
    def adjoint_melt_to_csv(df, t_str):
        df['Melt_t_' + t_str] = melt_vom_node.dat.data_ro
        if mesh.comm.rank == 0:
            df.to_csv(folder+"melt_foradjoint_7550m_profile.csv")

    ##### 
    # read linear simulations
    #target_folder = "/data/2d_adjoint/17.02.23_3_eq_param_ufric_dt300.0_dtOutput86400.0_T4320000.0_ip50.0_tres86400.0constant_Kh0.25_Kv0.001_structured_dy500_dz2_no_limiter_nosponge_open_qadvbc_pdyn_linTS/"
    adjoint_profile7550m_target = pd.read_csv("TS_foradjoint_7550m_profile_250423.csv")

    temp_profile_target = adjoint_profile7550m_target['T_t_14400']
    sal_profile_target = adjoint_profile7550m_target['S_t_14400']


    temp_vom_profile_target = Function(DG0_vom_profile)
    sal_vom_profile_target = Function(DG0_vom_profile)

    temp_vom_profile_target.dat.data[:] = temp_profile_target[:]
    sal_vom_profile_target.dat.data[:] = sal_profile_target[:]


    print("temp vom", temp_vom_profile.dat.data[:])

    print("temp vom target", temp_vom_profile_target.dat.data[:])

    ##########

    # define time steps
    dt = 300.0 
    T = 3000.0

    ##########

    # Set up time stepping routines
    vp_timestepper = CrankNicolsonSaddlePointTimeIntegrator([mom_eq, cty_eq], m, vp_fields, vp_coupling, dt, vp_bcs,
                                                            solver_parameters=vp_solver_parameters, strong_bcs=strong_bcs)

    #u_timestepper = DIRK33(u_eq, u, u_fields, dt, u_bcs, solver_parameters=u_solver_parameters)
    temp_timestepper = DIRK33(temp_eq, temp, temp_fields, dt, temp_bcs, solver_parameters=temp_solver_parameters)
    sal_timestepper = DIRK33(sal_eq, sal, sal_fields, dt, sal_bcs, solver_parameters=sal_solver_parameters)

    ##########

    # Set up folder



    # Depth profiles
    #depth_profile_to_csv(depth_profile500m, velocity_depth_profile500m, "500m", '0.0')
    #depth_profile_to_csv(depth_profile1km, velocity_depth_profile1km, "1km", '0.0')
    #depth_profile_to_csv(depth_profile2km, velocity_depth_profile2km, "2km", '0.0')
    #depth_profile_to_csv(depth_profile4km, velocity_depth_profile4km, "4km", '0.0')
    #depth_profile_to_csv(depth_profile6km, velocity_depth_profile6km, "6km", '0.0')

    ########


    # Begin time stepping
    t = 0.0
    step = 0


    while t < T - 0.5*dt:
        with timed_stage('velocity-pressure'):
            vp_timestepper.advance(t)
            #u_timestepper.advance(t)
        with timed_stage('temperature'):
            temp_timestepper.advance(t)
        with timed_stage('salinity'):
            sal_timestepper.advance(t)
        
        # dynamic pressure rhs
        temp_vom.interpolate(temp)
        sal_vom.interpolate(sal)
        p_dyn = dynamic_pressure(temp_vom.dat.data, sal_vom.dat.data)
        dynamic_pressure_interp = scipy.interpolate.interp1d(z_rhs, p_dyn, bounds_error=False,fill_value='extrapolate') 

        stress_open_boundary_dynamic.dat.data[:] = dynamic_pressure_interp(X.dat.data_ro[:,1])
        
        step += 1
        t += dt

    melt.project(mp.wb)
    #J = assemble(mp.wb*ds(4))
    N = len(temp_vom_profile.dat.data_ro)
    temp_vom_profile.interpolate(temp)
    sal_vom_profile.interpolate(sal)

    scale = Constant(1.0 / (0.5 * beta_sal)**2)
    alpha = 0.1
    J = assemble(scale*0.5/N * ((-beta_temp*(temp_vom_profile - temp_vom_profile_target))**2 + (beta_sal*(sal_vom_profile - sal_vom_profile_target))**2) * dx)  #+ assemble(0.5*alpha*scale*((beta_sal**2)*inner(grad(S_restorefield), grad(S_restorefield))+ (beta_temp**2)*inner(grad(T_restorefield), grad(T_restorefield)) )*dx)


    def eval_cb_pre(m):
        print("eval_cb_pre")
        print(len(m))
        print("T slope:", m[0].values()[0])
        print("T intercept:", m[1].values()[0])
        print("S slope:", m[2].values()[0])
        print("S intercept:", m[3].values()[0])
        tape.add_block(DiagnosticConstantBlock(T_slope, "T slope :"))
        tape.add_block(DiagnosticConstantBlock(T_intercept, "T intercept :"))
        tape.add_block(DiagnosticConstantBlock(S_slope, "S slope :"))
        tape.add_block(DiagnosticConstantBlock(S_intercept, "S intercept :"))


    def eval_cb(j, m):
        print("eval_cb")

        print(len(m))
        print("T slope:", m[0].values()[0])
        print("T intercept:", m[1].values()[0])
        print("S slope:", m[2].values()[0])
        print("S intercept:", m[3].values()[0])

        print("J = ", j)
        with DumbCheckpoint(folder+"dump.h5", mode=FILE_UPDATE) as chk:
            # Checkpoint file open for reading and writing
            chk.store(v_, name="v_velocity")
            chk.store(p_, name="perturbation_pressure")
            #chk.store(u, name="u_velocity")
            chk.store(temp, name="temperature")
            chk.store(sal, name="salinity")
            chk.store(T_restorefield_vis, name="temp restore")
            chk.store(S_restorefield_vis, name="sal restore")
    print(J)

    def derivative_cb_post(j, djdm, m):
        print("------------")
        print(" derivative cb post")
        print("------------")

    rf = ReducedFunctional(J, [c1]) #, c1,c2,c3], eval_cb_post=eval_cb, eval_cb_pre=eval_cb_pre, derivative_cb_post=derivative_cb_post)

    bounds = [[-0.02, -10, -0.01, 30],[0.02, 10., 0., 40. ] ]

    #g_opt = minimize(rf, bounds=bounds, options={"disp": True})

    #tape.reset_variables()
    #J.adj_value = 1.0

    J.block_variable.adj_value = 1.0
    #tape.visualise()
    # evaluate all adjoint blocks to ensure we get complete adjoint solution
    # currently requires fix in dolfin_adjoint_common/blocks/solving.py:
    #    meshtype derivative (line 204) is broken, so just return None instead
    #with timed_stage('adjoint'):
     #  tape.evaluate_adj()

    print(len(T_restorefield.dat.data))

    #grad = rf.derivative()
    #File(folder+'grad.pvd').write(grad)

    h = Constant(T_intercept)#S_restorefield)
    h.dat.data[:] = np.random.random(h.dat.data_ro.shape) # 2* for temp... 
    tt = taylor_test(rf, T_intercept, h)
    print("Tintercept TT ten steps")

    print(tt)
    assert np.allclose(tt, [2.0, 2.0, 2.0], rtol=5e-2)


def test_ice_shelf_coarse_open_lincon_Sslope():
    #nz = args.nz #10

    ip_factor = Constant(50.)
    #dt = 1.0
    restoring_time = 86400.

    ##########

    #  Generate mesh
    L = 10E3
    H1 = 2.
    H2 = 102.
    dy = 50.0
    ny = round(L/dy)
    #nz = 50
    dz = 1.0

    # create mesh
    mesh = Mesh("coarse.msh")

    PETSc.Sys.Print("Mesh dimension ", mesh.geometric_dimension())

    # shift z = 0 to surface of ocean. N.b z = 0 is outside domain.
    PETSc.Sys.Print("Length of lhs", assemble(Constant(1.0)*ds(1, domain=mesh)))

    PETSc.Sys.Print("Length of rhs", assemble(Constant(1.0)*ds(2, domain=mesh)))

    PETSc.Sys.Print("Length of bottom", assemble(Constant(1.0)*ds(3, domain=mesh)))

    PETSc.Sys.Print("Length of top", assemble(Constant(1.0)*ds(4, domain=mesh)))


    water_depth = 600.0
    mesh.coordinates.dat.data[:, 1] -= water_depth


    print("You have Comm WORLD size = ", mesh.comm.size)
    print("You have Comm WORLD rank = ", mesh.comm.rank)

    y, z = SpatialCoordinate(mesh)

    ##########

    # Set up function spaces
    V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
    W = FunctionSpace(mesh, "CG", 2)  # pressure space
    M = MixedFunctionSpace([V, W])

    # u velocity function space.
    U = FunctionSpace(mesh, "DG", 1)

    Q = FunctionSpace(mesh, "DG", 1)  # melt function space
    K = FunctionSpace(mesh, "DG", 1)    # temperature space
    S = FunctionSpace(mesh, "DG", 1)    # salinity space

    P1 = FunctionSpace(mesh, "CG", 1)
    print("vel dofs:", V.dim())
    print("pressure dofs:", W.dim())
    print("combined dofs:", M.dim())
    print("scalar dofs:", U.dim())
    print("P1 dofs (no. of nodes):", P1.dim())
    ##########

    # Set up functions
    m = Function(M)
    v_, p_ = m.split()  # function: y component of velocity, pressure
    v, p = split(m)  # expression: y component of velocity, pressure
    v_._name = "v_velocity"
    p_._name = "perturbation pressure"
    #u = Function(U, name="x velocity")  # x component of velocity

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

    ##########

    # Define a dump file
    dump_file = "./17.02.23_dump_50days_open_qadv_TSconst"

    DUMP = False #True
    if DUMP:
        with DumbCheckpoint(dump_file, mode=FILE_UPDATE) as chk:
            # Checkpoint file open for reading and writing
            chk.load(v_, name="v_velocity")
            chk.load(p_, name="perturbation_pressure")
            #chk.load(u, name="u_velocity")
    #        chk.load(sal, name="salinity")
    #        chk.load(temp, name="temperature")

            # from holland et al 2008b. constant T below 200m depth. varying sal.
            T_200m_depth = 1.0

            S_200m_depth = 34.4
            #S_bottom = 34.8
            #salinity_gradient = (S_bottom - S_200m_depth) / -H2
            #S_surface = S_200m_depth - (salinity_gradient * (H2 - water_depth))  # projected linear slope to surface.

            T_restore = Constant(T_200m_depth)
            S_restore = Constant(S_200m_depth) #S_surface + (S_bottom - S_surface) * (z / -water_depth)

            #T slope: 0.00042646953633639534
            #T intercept: 0.9797220017439936
            #S slope: -0.0006793874278126845
            #S intercept: 34.37197394968902
            T_slope = Constant(0.000426)
            T_intercept = Constant(0.9797)
            S_slope = Constant(-0.000679)
            S_intercept = Constant(34.37)
            
            temp.interpolate(T_intercept + T_slope * z)

            sal.interpolate(S_intercept + S_slope * z)


            # make T/S restore fields to calculate adjoint sensitity
            T_restorefield = Function(P1, name="Trestore field")
            S_restorefield = Function(P1, name="Srestore field")

            T_restorefield.assign(T_restore)
            S_restorefield.assign(S_restore)


    else:
        # Assign Initial conditions
        v_init = zero(mesh.geometric_dimension())
        v_.assign(v_init)

        #u_init = Constant(0.0)
        #u.interpolate(u_init)

        # ignore below was used to get 50 day run...
        T_200m_depth = -0.5
        T_bottom = 1.0
        temp_gradient = (T_bottom - T_200m_depth) / -H2
        T_surface = T_200m_depth - (temp_gradient * (H2 - water_depth))  # projected linear slope to surface.

        T_restore = T_surface + (T_bottom - T_surface) * (z / -water_depth)


        S_200m_depth = 34.4
        S_bottom = 34.8
        salinity_gradient = (S_bottom - S_200m_depth) / -H2
        S_surface = S_200m_depth - (salinity_gradient * (H2 - water_depth))  # projected linear slope to surface.
        S_restore = S_surface + (S_bottom - S_surface) * (z / -water_depth)
        
        T_restorefield = Function(P1, name="Trestore field")
        S_restorefield = Function(P1, name="Srestore field")

        T_restorefield.interpolate(T_restore)
        S_restorefield.interpolate(S_restore)

        # below is code for initialising and rhs bc...

        T_slope = Constant(0)
        T_intercept = Constant(1.0)
        S_slope = Constant(0)
        S_intercept = Constant(34.4)
            
        c = Control(T_slope) 
        c1 = Control(T_intercept) 
        c2 = Control(S_slope) 
        c3 = Control(S_intercept) 
        
        temp.interpolate(T_intercept + T_slope * z)
        sal.interpolate(S_intercept + S_slope * z)
    #    temp_init = T_restore
    #    temp.interpolate(temp_init)

    #    sal_init = Constant(34.4)
    #    sal_init = S_restore
    #    sal.interpolate(sal_init)
            #T_slope = Constant(0.0)
            #T_intercept = Constant(1.0)
            #S_slope = Constant(0.0)
            #S_intercept = Constant(34.0)
            
            #temp.interpolate(T_intercept + T_slope * z)

            #sal.interpolate(S_intercept + S_slope * z)
    #c = Control(temp)


    ##########

    # Set up equations
    mom_eq = MomentumEquation(M.sub(0), M.sub(0))
    cty_eq = ContinuityEquation(M.sub(1), M.sub(1))
    #u_eq = ScalarVelocity2halfDEquation(U, U)
    temp_eq = ScalarAdvectionDiffusionEquation(K, K)
    sal_eq = ScalarAdvectionDiffusionEquation(S, S)

    ##########

    # Terms for equation fields

    # momentum source: the buoyancy term Boussinesq approx. From mitgcm default
    T_ref = Constant(0.0)
    S_ref = Constant(35)
    beta_temp = Constant(2.0E-4)
    beta_sal = Constant(7.4E-4)
    g = Constant(9.81)
    mom_source = as_vector((0., -g))*(-beta_temp*(temp - T_ref) + beta_sal * (sal - S_ref))

    rho0 = 1030.
    rho.interpolate(rho0*(1.0-beta_temp * (temp - T_ref) + beta_sal * (sal - S_ref)))
    # coriolis frequency f-plane assumption at 75deg S. f = 2 omega sin (lat) = 2 * 7.2921E-5 * sin (-75 *2pi/360)
    #f = Constant(-1.409E-4)

    # Scalar source/sink terms at open boundary.
    absorption_factor = Constant(1.0/restoring_time)
    sponge_fraction = 0.2  # fraction of domain where sponge
    # Temperature source term
    source_temp = conditional(y > (1.0-sponge_fraction) * L,
                              absorption_factor * T_restorefield,
                              0.0)

    # Salinity source term
    source_sal = conditional(y > (1.0-sponge_fraction) * L,
                             absorption_factor * S_restorefield,
                             0.0)

    # Temperature absorption term
    absorp_temp = conditional(y > (1.0-sponge_fraction) * L,
                              absorption_factor,
                              0.0)

    # Salinity absorption term
    absorp_sal = conditional(y > (1.0-sponge_fraction) * L,
                             absorption_factor,
                             0.0)


    # linearly vary viscosity/diffusivity over domain. reduce vertical/diffusion
    kappa_h = Constant(0.25)
    kappa_v = Constant(0.001)
    #kappa_v = Constant(args.Kh*dz/dy)
    #grounding_line_kappa_v = Constant(open_ocean_kappa_v*H1/H2)
    #kappa_v_grad = (open_ocean_kappa_v-grounding_line_kappa_v)/L
    #kappa_v = grounding_line_kappa_v + y*kappa_v_grad

    #sponge_kappa_h = conditional(y > (1.0-sponge_fraction) * L,
    #                             1000. * kappa_h * ((y - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
    #                             kappa_h)

    #sponge_kappa_v = conditional(y > (1.0-sponge_fraction) * L,
    #                             1000. * kappa_v * ((y - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
    #                             kappa_v)

    kappa = as_tensor([[kappa_h, 0], [0, kappa_v]])
    #kappa = Constant([[kappa_h, 0], [0, kappa_v]]) # for taylor test need to use Constant for some reason...

    TP1 = TensorFunctionSpace(mesh, "CG", 1)
    kappa_temp = Function(TP1, name='temperature diffusion').project(kappa)
    kappa_sal = Function(TP1, name='salinity diffusion').project(kappa)
    mu = Function(TP1, name='viscosity').project(kappa)


    # Interior penalty term
    # 3*cot(min_angle)*(p+1)*p*nu_max/nu_min

    ip_alpha = Constant(3*dy/dz*2*ip_factor)
    # Equation fields
    vp_coupling = [{'pressure': 1}, {'velocity': 0}]
    vp_fields = {'viscosity': mu, 'source': mom_source, 'interior_penalty': ip_alpha}
    #u_fields = {'diffusivity': mu, 'velocity': v, 'interior_penalty': ip_alpha, 'coriolis_frequency': f}
    temp_fields = {'diffusivity': kappa_temp, 'velocity': v, 'interior_penalty': ip_alpha }#, 'source': source_temp,
    #               'absorption coefficient': absorp_temp}
    sal_fields = {'diffusivity': kappa_sal, 'velocity': v, 'interior_penalty': ip_alpha }#, 'source': source_sal,
     #             'absorption coefficient': absorp_sal}

    ##########

    # Get expressions used in melt rate parameterisation
    mp = ThreeEqMeltRateParam(sal, temp, p, z, velocity=pow(dot(v, v), 0.5), HJ99Gamma=True)

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

    # Plotting top boundary.
    shelf_boundary_points = get_top_boundary(cavity_length=L, cavity_height=H2, water_depth=water_depth)
    top_boundary_mp = pd.DataFrame()


    def top_boundary_to_csv(boundary_points, df, t_str):
        df['Qice_t_' + t_str] = Q_ice.at(boundary_points)
        df['Qmixed_t_' + t_str] = Q_mixed.at(boundary_points)
        df['Qlat_t_' + t_str] = Q_latent.at(boundary_points)
        df['Qsalt_t_' + t_str] = Q_s.at(boundary_points)
        df['Melt_t' + t_str] = melt.at(boundary_points)
        df['Tb_t_' + t_str] = Tb.at(boundary_points)
        df['P_t_' + t_str] = full_pressure.at(boundary_points)
        df['Sal_t_' + t_str] = sal.at(boundary_points)
        df['Temp_t_' + t_str] = temp.at(boundary_points)
        df["integrated_melt_t_ " + t_str] = assemble(melt * ds(4))

        if mesh.comm.rank == 0:
            top_boundary_mp.to_csv(folder+"top_boundary_data.csv")


    ##########

    # Boundary conditions
    # top boundary: no normal flow, drag flowing over ice
    # bottom boundary: no normal flow, drag flowing over bedrock
    # grounding line wall (LHS): no normal flow
    # open ocean (RHS): pressure to account for density differences

    # WEAKLY Enforced BCs
    n = FacetNormal(mesh)
    Temperature_term = -beta_temp * (0.5 * T_slope * pow(z, 2)  + (T_intercept-T_ref) * z)
    Salinity_term = beta_sal * (0.5 * S_slope * pow(z, 2)  + (S_intercept-S_ref) * z)
    stress_open_boundary = -n*-g*(Temperature_term + Salinity_term)
    no_normal_flow = 0.
    ice_drag = 0.0097

    import scipy



    z_rhs = np.linspace(-498.01, -599.99, 400)
    rhs_nodes = []
    for zi in z_rhs:
        rhs_nodes.append([9999.995, zi])

    print(rhs_nodes)
    rhs_nodes = VertexOnlyMesh(mesh,rhs_nodes,missing_points_behaviour='warn')

    DG0_vom = FunctionSpace(rhs_nodes, "DG", 0)
    temp_vom = Function(DG0_vom)
    sal_vom = Function(DG0_vom)

    temp_vom.interpolate(temp)
    sal_vom.interpolate(sal)

    print(sal_vom.dat.data)

    p_rhs = []
    def dynamic_pressure(T, S):

    # because flow is out the domain with positive slope dont do any modification of sal/temp.

     #   for i in range(10):
    #        S[i] -= 0.1*(10-i)*2e-4
    #    S[:10] -= 0.00016  # try making the top of the domain a bit fresher as a hack to get faster flow at the rhs...
    #    for i in range(10, len(T)):  
    #        S[i] = 34
    #        T[i] = -0.5
       
        density_rhs =  -g.values()[0] *(-beta_temp.values()[0] * (T-T_ref.values()[0]) +  beta_sal.values()[0] * (S - S_ref.values()[0]))
      #  print(density_rhs)



    #    print("len z", len(z_rhs))
    #    print("len density", len(density_rhs))
        pcumulative = scipy.integrate.cumulative_trapezoid(density_rhs, z_rhs, initial=0)
     #   print("pcumulative", pcumulative)
     #   print("len pcumulative", len(pcumulative))

        return pcumulative

    p_dyn = dynamic_pressure(temp_vom.dat.data, sal_vom.dat.data)
    dynamic_pressure_interp = scipy.interpolate.interp1d(z_rhs, p_dyn, bounds_error=False,fill_value='extrapolate') 


    # Now make the VectorFunctionSpace corresponding to V.
    W_vector = VectorFunctionSpace(mesh, W.ufl_element())

    # Next, interpolate the coordinates onto the nodes of W.
    X = interpolate(mesh.coordinates, W_vector)
    print("X[:,1]", X.dat.data[:,1])
    # Make an output function.
    stress_open_boundary_dynamic = Function(W)

    # Use the external data function to interpolate the values of f.
    stress_open_boundary_dynamic.dat.data[:] = dynamic_pressure_interp(X.dat.data_ro[:,1])

    # test stress open_boundary
    #sop = Function(W)
    #sop.interpolate(-g*(Temperature_term + Salinity_term))
    #sop_file = File(folder+"boundary_stress.pvd")
    #sop_file.write(sop)


    vp_bcs = {4: {'un': no_normal_flow, 'drag': ice_drag}, 2: {'stress': stress_open_boundary},
              3: {'un': no_normal_flow, 'drag': 0.0025}, 1: {'un': no_normal_flow}}
    #u_bcs = {2: {'q': Constant(0.0)}}

    temp_bcs = {4: {'flux': -mp.T_flux_bc}, 2: {'qadv': T_intercept + T_slope * z}}

    sal_bcs = {4: {'flux': -mp.S_flux_bc}, 2: {'qadv': S_intercept + S_slope * z}}



    # STRONGLY Enforced BCs
    # open ocean (RHS): no tangential flow because viscosity of outside ocean resists vertical flow.
    strong_bcs = []#DirichletBC(M.sub(0).sub(1), 0, 2)]

    ##########

    # Solver parameters
    mumps_solver_parameters = {
        'snes_monitor': None,
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij',
        'snes_max_it': 100,
        'snes_rtol': 1e-8,
        'snes_atol': 1e-6,
    }

    vp_solver_parameters = mumps_solver_parameters
    u_solver_parameters = mumps_solver_parameters
    temp_solver_parameters = mumps_solver_parameters
    sal_solver_parameters = mumps_solver_parameters

    ##########

    # Plotting depth profiles.
    z500m = cavity_thickness(5E2, 0., H1, L, H2)
    z1km = cavity_thickness(1E3, 0., H1, L, H2)
    z2km = cavity_thickness(2E3, 0., H1, L, H2)
    z4km = cavity_thickness(4E3, 0., H1, L, H2)
    z6km = cavity_thickness(6E3, 0., H1, L, H2)


    z_profile500m = np.linspace(z500m-water_depth-1., 1.-water_depth, 50)
    z_profile1km = np.linspace(z1km-water_depth-1., 1.-water_depth, 50)
    z_profile2km = np.linspace(z2km-water_depth-1., 1.-water_depth, 50)
    z_profile4km = np.linspace(z4km-water_depth-1., 1.-water_depth, 50)
    z_profile6km = np.linspace(z6km-water_depth-1., 1.-water_depth, 50)


    depth_profile500m = []
    depth_profile1km = []
    depth_profile2km = []
    depth_profile4km = []
    depth_profile6km = []

    for d5e2, d1km, d2km, d4km, d6km in zip(z_profile500m, z_profile1km, z_profile2km, z_profile4km, z_profile6km):
        depth_profile500m.append([5E2, d5e2])
        depth_profile1km.append([1E3, d1km])
        depth_profile2km.append([2E3, d2km])
        depth_profile4km.append([4E3, d4km])
        depth_profile6km.append([6E3, d6km])

    velocity_depth_profile500m = pd.DataFrame()
    velocity_depth_profile1km = pd.DataFrame()
    velocity_depth_profile2km = pd.DataFrame()
    velocity_depth_profile4km = pd.DataFrame()
    velocity_depth_profile6km = pd.DataFrame()

    velocity_depth_profile500m['Z_profile'] = z_profile500m
    velocity_depth_profile1km['Z_profile'] = z_profile1km
    velocity_depth_profile2km['Z_profile'] = z_profile2km
    velocity_depth_profile4km['Z_profile'] = z_profile4km
    velocity_depth_profile6km['Z_profile'] = z_profile6km


    def depth_profile_to_csv(profile, df, depth, t_str):
        #df['U_t_' + t_str] = u.at(profile)
        vw = np.array(v_.at(profile))
        vv = vw[:, 0]
        ww = vw[:, 1]
        df['V_t_' + t_str] = vv
        df['W_t_' + t_str] = ww
        if mesh.comm.rank == 0:
            df.to_csv(folder+depth+"_profile.csv")

    #### VOM for doing objective function...

    z_profile_adjoint = np.linspace(-525, -595, 15)
    z_profile_nodes = []
    for zi in z_profile_adjoint:
        z_profile_nodes.append([7550, zi])

    print(z_profile_nodes)
    z_profile_nodes = VertexOnlyMesh(mesh,z_profile_nodes,missing_points_behaviour='warn')

    DG0_vom_profile = FunctionSpace(z_profile_nodes, "DG", 0)
    temp_vom_profile = Function(DG0_vom_profile)
    sal_vom_profile = Function(DG0_vom_profile)

    temp_vom_profile.interpolate(temp)
    sal_vom_profile.interpolate(sal)


    adjoint_profile7550m = pd.DataFrame()
    adjoint_profile7550m['Z_profile'] = z_profile_adjoint

    melt_node = VertexOnlyMesh(mesh, [[7550, -522.7]], missing_points_behaviour='warn')
    DG0_vom_meltnode = FunctionSpace(melt_node, "DG", 0)
    melt_vom_node = Function(DG0_vom_meltnode)

    melt_vom_node.interpolate(mp.wb)

    def adjoint_profile_to_csv(df, t_str):
        df['T_t_' + t_str] = temp_vom_profile.dat.data_ro
        df['S_t_' + t_str] = sal_vom_profile.dat.data_ro
        if mesh.comm.rank == 0:
            df.to_csv(folder+"TS_foradjoint_7550m_profile.csv")

    adjoint_melt7550m = pd.DataFrame()
    def adjoint_melt_to_csv(df, t_str):
        df['Melt_t_' + t_str] = melt_vom_node.dat.data_ro
        if mesh.comm.rank == 0:
            df.to_csv(folder+"melt_foradjoint_7550m_profile.csv")

    ##### 
    # read linear simulations
    #target_folder = "/data/2d_adjoint/17.02.23_3_eq_param_ufric_dt300.0_dtOutput86400.0_T4320000.0_ip50.0_tres86400.0constant_Kh0.25_Kv0.001_structured_dy500_dz2_no_limiter_nosponge_open_qadvbc_pdyn_linTS/"
    adjoint_profile7550m_target = pd.read_csv("TS_foradjoint_7550m_profile_250423.csv")

    temp_profile_target = adjoint_profile7550m_target['T_t_14400']
    sal_profile_target = adjoint_profile7550m_target['S_t_14400']


    temp_vom_profile_target = Function(DG0_vom_profile)
    sal_vom_profile_target = Function(DG0_vom_profile)

    temp_vom_profile_target.dat.data[:] = temp_profile_target[:]
    sal_vom_profile_target.dat.data[:] = sal_profile_target[:]


    print("temp vom", temp_vom_profile.dat.data[:])

    print("temp vom target", temp_vom_profile_target.dat.data[:])

    ##########

    # define time steps
    dt = 300.0 
    T = 3000.0

    ##########

    # Set up time stepping routines
    vp_timestepper = CrankNicolsonSaddlePointTimeIntegrator([mom_eq, cty_eq], m, vp_fields, vp_coupling, dt, vp_bcs,
                                                            solver_parameters=vp_solver_parameters, strong_bcs=strong_bcs)

    #u_timestepper = DIRK33(u_eq, u, u_fields, dt, u_bcs, solver_parameters=u_solver_parameters)
    temp_timestepper = DIRK33(temp_eq, temp, temp_fields, dt, temp_bcs, solver_parameters=temp_solver_parameters)
    sal_timestepper = DIRK33(sal_eq, sal, sal_fields, dt, sal_bcs, solver_parameters=sal_solver_parameters)

    ##########

    # Set up folder



    # Depth profiles
    #depth_profile_to_csv(depth_profile500m, velocity_depth_profile500m, "500m", '0.0')
    #depth_profile_to_csv(depth_profile1km, velocity_depth_profile1km, "1km", '0.0')
    #depth_profile_to_csv(depth_profile2km, velocity_depth_profile2km, "2km", '0.0')
    #depth_profile_to_csv(depth_profile4km, velocity_depth_profile4km, "4km", '0.0')
    #depth_profile_to_csv(depth_profile6km, velocity_depth_profile6km, "6km", '0.0')

    ########


    # Begin time stepping
    t = 0.0
    step = 0


    while t < T - 0.5*dt:
        with timed_stage('velocity-pressure'):
            vp_timestepper.advance(t)
            #u_timestepper.advance(t)
        with timed_stage('temperature'):
            temp_timestepper.advance(t)
        with timed_stage('salinity'):
            sal_timestepper.advance(t)
        
        # dynamic pressure rhs
        temp_vom.interpolate(temp)
        sal_vom.interpolate(sal)
        p_dyn = dynamic_pressure(temp_vom.dat.data, sal_vom.dat.data)
        dynamic_pressure_interp = scipy.interpolate.interp1d(z_rhs, p_dyn, bounds_error=False,fill_value='extrapolate') 

        stress_open_boundary_dynamic.dat.data[:] = dynamic_pressure_interp(X.dat.data_ro[:,1])
        
        step += 1
        t += dt

    melt.project(mp.wb)
    #J = assemble(mp.wb*ds(4))
    N = len(temp_vom_profile.dat.data_ro)
    temp_vom_profile.interpolate(temp)
    sal_vom_profile.interpolate(sal)

    scale = Constant(1.0 / (0.5 * beta_sal)**2)
    alpha = 0.1
    J = assemble(scale*0.5/N * ((-beta_temp*(temp_vom_profile - temp_vom_profile_target))**2 + (beta_sal*(sal_vom_profile - sal_vom_profile_target))**2) * dx)  #+ assemble(0.5*alpha*scale*((beta_sal**2)*inner(grad(S_restorefield), grad(S_restorefield))+ (beta_temp**2)*inner(grad(T_restorefield), grad(T_restorefield)) )*dx)


    def eval_cb_pre(m):
        print("eval_cb_pre")
        print(len(m))
        print("T slope:", m[0].values()[0])
        print("T intercept:", m[1].values()[0])
        print("S slope:", m[2].values()[0])
        print("S intercept:", m[3].values()[0])
        tape.add_block(DiagnosticConstantBlock(T_slope, "T slope :"))
        tape.add_block(DiagnosticConstantBlock(T_intercept, "T intercept :"))
        tape.add_block(DiagnosticConstantBlock(S_slope, "S slope :"))
        tape.add_block(DiagnosticConstantBlock(S_intercept, "S intercept :"))


    def eval_cb(j, m):
        print("eval_cb")

        print(len(m))
        print("T slope:", m[0].values()[0])
        print("T intercept:", m[1].values()[0])
        print("S slope:", m[2].values()[0])
        print("S intercept:", m[3].values()[0])

        print("J = ", j)
        with DumbCheckpoint(folder+"dump.h5", mode=FILE_UPDATE) as chk:
            # Checkpoint file open for reading and writing
            chk.store(v_, name="v_velocity")
            chk.store(p_, name="perturbation_pressure")
            #chk.store(u, name="u_velocity")
            chk.store(temp, name="temperature")
            chk.store(sal, name="salinity")
            chk.store(T_restorefield_vis, name="temp restore")
            chk.store(S_restorefield_vis, name="sal restore")
    print(J)

    def derivative_cb_post(j, djdm, m):
        print("------------")
        print(" derivative cb post")
        print("------------")

    rf = ReducedFunctional(J, [c2]) #, c1,c2,c3], eval_cb_post=eval_cb, eval_cb_pre=eval_cb_pre, derivative_cb_post=derivative_cb_post)

    bounds = [[-0.02, -10, -0.01, 30],[0.02, 10., 0., 40. ] ]

    #g_opt = minimize(rf, bounds=bounds, options={"disp": True})

    #tape.reset_variables()
    #J.adj_value = 1.0

    J.block_variable.adj_value = 1.0
    #tape.visualise()
    # evaluate all adjoint blocks to ensure we get complete adjoint solution
    # currently requires fix in dolfin_adjoint_common/blocks/solving.py:
    #    meshtype derivative (line 204) is broken, so just return None instead
    #with timed_stage('adjoint'):
     #  tape.evaluate_adj()

    print(len(T_restorefield.dat.data))

    #grad = rf.derivative()
    #File(folder+'grad.pvd').write(grad)

    h = Constant(S_slope)#S_restorefield)
    h.dat.data[:] = np.random.random(h.dat.data_ro.shape) # 2* for temp... 
    tt = taylor_test(rf, S_slope, h)
    print("Sslope TT ten steps")

    print(tt)
    assert np.allclose(tt, [2.0, 2.0, 2.0], rtol=5e-2)


def test_ice_shelf_coarse_open_lincon_Sintercept():
    #nz = args.nz #10

    ip_factor = Constant(50.)
    #dt = 1.0
    restoring_time = 86400.

    ##########

    #  Generate mesh
    L = 10E3
    H1 = 2.
    H2 = 102.
    dy = 50.0
    ny = round(L/dy)
    #nz = 50
    dz = 1.0

    # create mesh
    mesh = Mesh("coarse.msh")

    PETSc.Sys.Print("Mesh dimension ", mesh.geometric_dimension())

    # shift z = 0 to surface of ocean. N.b z = 0 is outside domain.
    PETSc.Sys.Print("Length of lhs", assemble(Constant(1.0)*ds(1, domain=mesh)))

    PETSc.Sys.Print("Length of rhs", assemble(Constant(1.0)*ds(2, domain=mesh)))

    PETSc.Sys.Print("Length of bottom", assemble(Constant(1.0)*ds(3, domain=mesh)))

    PETSc.Sys.Print("Length of top", assemble(Constant(1.0)*ds(4, domain=mesh)))


    water_depth = 600.0
    mesh.coordinates.dat.data[:, 1] -= water_depth


    print("You have Comm WORLD size = ", mesh.comm.size)
    print("You have Comm WORLD rank = ", mesh.comm.rank)

    y, z = SpatialCoordinate(mesh)

    ##########

    # Set up function spaces
    V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
    W = FunctionSpace(mesh, "CG", 2)  # pressure space
    M = MixedFunctionSpace([V, W])

    # u velocity function space.
    U = FunctionSpace(mesh, "DG", 1)

    Q = FunctionSpace(mesh, "DG", 1)  # melt function space
    K = FunctionSpace(mesh, "DG", 1)    # temperature space
    S = FunctionSpace(mesh, "DG", 1)    # salinity space

    P1 = FunctionSpace(mesh, "CG", 1)
    print("vel dofs:", V.dim())
    print("pressure dofs:", W.dim())
    print("combined dofs:", M.dim())
    print("scalar dofs:", U.dim())
    print("P1 dofs (no. of nodes):", P1.dim())
    ##########

    # Set up functions
    m = Function(M)
    v_, p_ = m.split()  # function: y component of velocity, pressure
    v, p = split(m)  # expression: y component of velocity, pressure
    v_._name = "v_velocity"
    p_._name = "perturbation pressure"
    #u = Function(U, name="x velocity")  # x component of velocity

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

    ##########

    # Define a dump file
    dump_file = "./17.02.23_dump_50days_open_qadv_TSconst"

    DUMP = False #True
    if DUMP:
        with DumbCheckpoint(dump_file, mode=FILE_UPDATE) as chk:
            # Checkpoint file open for reading and writing
            chk.load(v_, name="v_velocity")
            chk.load(p_, name="perturbation_pressure")
            #chk.load(u, name="u_velocity")
    #        chk.load(sal, name="salinity")
    #        chk.load(temp, name="temperature")

            # from holland et al 2008b. constant T below 200m depth. varying sal.
            T_200m_depth = 1.0

            S_200m_depth = 34.4
            #S_bottom = 34.8
            #salinity_gradient = (S_bottom - S_200m_depth) / -H2
            #S_surface = S_200m_depth - (salinity_gradient * (H2 - water_depth))  # projected linear slope to surface.

            T_restore = Constant(T_200m_depth)
            S_restore = Constant(S_200m_depth) #S_surface + (S_bottom - S_surface) * (z / -water_depth)

            #T slope: 0.00042646953633639534
            #T intercept: 0.9797220017439936
            #S slope: -0.0006793874278126845
            #S intercept: 34.37197394968902
            T_slope = Constant(0.000426)
            T_intercept = Constant(0.9797)
            S_slope = Constant(-0.000679)
            S_intercept = Constant(34.37)
            
            temp.interpolate(T_intercept + T_slope * z)

            sal.interpolate(S_intercept + S_slope * z)


            # make T/S restore fields to calculate adjoint sensitity
            T_restorefield = Function(P1, name="Trestore field")
            S_restorefield = Function(P1, name="Srestore field")

            T_restorefield.assign(T_restore)
            S_restorefield.assign(S_restore)


    else:
        # Assign Initial conditions
        v_init = zero(mesh.geometric_dimension())
        v_.assign(v_init)

        #u_init = Constant(0.0)
        #u.interpolate(u_init)

        # ignore below was used to get 50 day run...
        T_200m_depth = -0.5
        T_bottom = 1.0
        temp_gradient = (T_bottom - T_200m_depth) / -H2
        T_surface = T_200m_depth - (temp_gradient * (H2 - water_depth))  # projected linear slope to surface.

        T_restore = T_surface + (T_bottom - T_surface) * (z / -water_depth)


        S_200m_depth = 34.4
        S_bottom = 34.8
        salinity_gradient = (S_bottom - S_200m_depth) / -H2
        S_surface = S_200m_depth - (salinity_gradient * (H2 - water_depth))  # projected linear slope to surface.
        S_restore = S_surface + (S_bottom - S_surface) * (z / -water_depth)
        
        T_restorefield = Function(P1, name="Trestore field")
        S_restorefield = Function(P1, name="Srestore field")

        T_restorefield.interpolate(T_restore)
        S_restorefield.interpolate(S_restore)

        # below is code for initialising and rhs bc...

        T_slope = Constant(0)
        T_intercept = Constant(1.0)
        S_slope = Constant(0)
        S_intercept = Constant(34.4)
            
        c = Control(T_slope) 
        c1 = Control(T_intercept) 
        c2 = Control(S_slope) 
        c3 = Control(S_intercept) 
        
        temp.interpolate(T_intercept + T_slope * z)
        sal.interpolate(S_intercept + S_slope * z)
    #    temp_init = T_restore
    #    temp.interpolate(temp_init)

    #    sal_init = Constant(34.4)
    #    sal_init = S_restore
    #    sal.interpolate(sal_init)
            #T_slope = Constant(0.0)
            #T_intercept = Constant(1.0)
            #S_slope = Constant(0.0)
            #S_intercept = Constant(34.0)
            
            #temp.interpolate(T_intercept + T_slope * z)

            #sal.interpolate(S_intercept + S_slope * z)
    #c = Control(temp)


    ##########

    # Set up equations
    mom_eq = MomentumEquation(M.sub(0), M.sub(0))
    cty_eq = ContinuityEquation(M.sub(1), M.sub(1))
    #u_eq = ScalarVelocity2halfDEquation(U, U)
    temp_eq = ScalarAdvectionDiffusionEquation(K, K)
    sal_eq = ScalarAdvectionDiffusionEquation(S, S)

    ##########

    # Terms for equation fields

    # momentum source: the buoyancy term Boussinesq approx. From mitgcm default
    T_ref = Constant(0.0)
    S_ref = Constant(35)
    beta_temp = Constant(2.0E-4)
    beta_sal = Constant(7.4E-4)
    g = Constant(9.81)
    mom_source = as_vector((0., -g))*(-beta_temp*(temp - T_ref) + beta_sal * (sal - S_ref))

    rho0 = 1030.
    rho.interpolate(rho0*(1.0-beta_temp * (temp - T_ref) + beta_sal * (sal - S_ref)))
    # coriolis frequency f-plane assumption at 75deg S. f = 2 omega sin (lat) = 2 * 7.2921E-5 * sin (-75 *2pi/360)
    #f = Constant(-1.409E-4)

    # Scalar source/sink terms at open boundary.
    absorption_factor = Constant(1.0/restoring_time)
    sponge_fraction = 0.2  # fraction of domain where sponge
    # Temperature source term
    source_temp = conditional(y > (1.0-sponge_fraction) * L,
                              absorption_factor * T_restorefield,
                              0.0)

    # Salinity source term
    source_sal = conditional(y > (1.0-sponge_fraction) * L,
                             absorption_factor * S_restorefield,
                             0.0)

    # Temperature absorption term
    absorp_temp = conditional(y > (1.0-sponge_fraction) * L,
                              absorption_factor,
                              0.0)

    # Salinity absorption term
    absorp_sal = conditional(y > (1.0-sponge_fraction) * L,
                             absorption_factor,
                             0.0)


    # linearly vary viscosity/diffusivity over domain. reduce vertical/diffusion
    kappa_h = Constant(0.25)
    kappa_v = Constant(0.001)
    #kappa_v = Constant(args.Kh*dz/dy)
    #grounding_line_kappa_v = Constant(open_ocean_kappa_v*H1/H2)
    #kappa_v_grad = (open_ocean_kappa_v-grounding_line_kappa_v)/L
    #kappa_v = grounding_line_kappa_v + y*kappa_v_grad

    #sponge_kappa_h = conditional(y > (1.0-sponge_fraction) * L,
    #                             1000. * kappa_h * ((y - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
    #                             kappa_h)

    #sponge_kappa_v = conditional(y > (1.0-sponge_fraction) * L,
    #                             1000. * kappa_v * ((y - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
    #                             kappa_v)

    kappa = as_tensor([[kappa_h, 0], [0, kappa_v]])
    #kappa = Constant([[kappa_h, 0], [0, kappa_v]]) # for taylor test need to use Constant for some reason...

    TP1 = TensorFunctionSpace(mesh, "CG", 1)
    kappa_temp = Function(TP1, name='temperature diffusion').project(kappa)
    kappa_sal = Function(TP1, name='salinity diffusion').project(kappa)
    mu = Function(TP1, name='viscosity').project(kappa)


    # Interior penalty term
    # 3*cot(min_angle)*(p+1)*p*nu_max/nu_min

    ip_alpha = Constant(3*dy/dz*2*ip_factor)
    # Equation fields
    vp_coupling = [{'pressure': 1}, {'velocity': 0}]
    vp_fields = {'viscosity': mu, 'source': mom_source, 'interior_penalty': ip_alpha}
    #u_fields = {'diffusivity': mu, 'velocity': v, 'interior_penalty': ip_alpha, 'coriolis_frequency': f}
    temp_fields = {'diffusivity': kappa_temp, 'velocity': v, 'interior_penalty': ip_alpha }#, 'source': source_temp,
    #               'absorption coefficient': absorp_temp}
    sal_fields = {'diffusivity': kappa_sal, 'velocity': v, 'interior_penalty': ip_alpha }#, 'source': source_sal,
     #             'absorption coefficient': absorp_sal}

    ##########

    # Get expressions used in melt rate parameterisation
    mp = ThreeEqMeltRateParam(sal, temp, p, z, velocity=pow(dot(v, v), 0.5), HJ99Gamma=True)

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

    # Plotting top boundary.
    shelf_boundary_points = get_top_boundary(cavity_length=L, cavity_height=H2, water_depth=water_depth)
    top_boundary_mp = pd.DataFrame()


    def top_boundary_to_csv(boundary_points, df, t_str):
        df['Qice_t_' + t_str] = Q_ice.at(boundary_points)
        df['Qmixed_t_' + t_str] = Q_mixed.at(boundary_points)
        df['Qlat_t_' + t_str] = Q_latent.at(boundary_points)
        df['Qsalt_t_' + t_str] = Q_s.at(boundary_points)
        df['Melt_t' + t_str] = melt.at(boundary_points)
        df['Tb_t_' + t_str] = Tb.at(boundary_points)
        df['P_t_' + t_str] = full_pressure.at(boundary_points)
        df['Sal_t_' + t_str] = sal.at(boundary_points)
        df['Temp_t_' + t_str] = temp.at(boundary_points)
        df["integrated_melt_t_ " + t_str] = assemble(melt * ds(4))

        if mesh.comm.rank == 0:
            top_boundary_mp.to_csv(folder+"top_boundary_data.csv")


    ##########

    # Boundary conditions
    # top boundary: no normal flow, drag flowing over ice
    # bottom boundary: no normal flow, drag flowing over bedrock
    # grounding line wall (LHS): no normal flow
    # open ocean (RHS): pressure to account for density differences

    # WEAKLY Enforced BCs
    n = FacetNormal(mesh)
    Temperature_term = -beta_temp * (0.5 * T_slope * pow(z, 2)  + (T_intercept-T_ref) * z)
    Salinity_term = beta_sal * (0.5 * S_slope * pow(z, 2)  + (S_intercept-S_ref) * z)
    stress_open_boundary = -n*-g*(Temperature_term + Salinity_term)
    no_normal_flow = 0.
    ice_drag = 0.0097

    import scipy



    z_rhs = np.linspace(-498.01, -599.99, 400)
    rhs_nodes = []
    for zi in z_rhs:
        rhs_nodes.append([9999.995, zi])

    print(rhs_nodes)
    rhs_nodes = VertexOnlyMesh(mesh,rhs_nodes,missing_points_behaviour='warn')

    DG0_vom = FunctionSpace(rhs_nodes, "DG", 0)
    temp_vom = Function(DG0_vom)
    sal_vom = Function(DG0_vom)

    temp_vom.interpolate(temp)
    sal_vom.interpolate(sal)

    print(sal_vom.dat.data)

    p_rhs = []
    def dynamic_pressure(T, S):

    # because flow is out the domain with positive slope dont do any modification of sal/temp.

     #   for i in range(10):
    #        S[i] -= 0.1*(10-i)*2e-4
    #    S[:10] -= 0.00016  # try making the top of the domain a bit fresher as a hack to get faster flow at the rhs...
    #    for i in range(10, len(T)):  
    #        S[i] = 34
    #        T[i] = -0.5
       
        density_rhs =  -g.values()[0] *(-beta_temp.values()[0] * (T-T_ref.values()[0]) +  beta_sal.values()[0] * (S - S_ref.values()[0]))
      #  print(density_rhs)



    #    print("len z", len(z_rhs))
    #    print("len density", len(density_rhs))
        pcumulative = scipy.integrate.cumulative_trapezoid(density_rhs, z_rhs, initial=0)
     #   print("pcumulative", pcumulative)
     #   print("len pcumulative", len(pcumulative))

        return pcumulative

    p_dyn = dynamic_pressure(temp_vom.dat.data, sal_vom.dat.data)
    dynamic_pressure_interp = scipy.interpolate.interp1d(z_rhs, p_dyn, bounds_error=False,fill_value='extrapolate') 


    # Now make the VectorFunctionSpace corresponding to V.
    W_vector = VectorFunctionSpace(mesh, W.ufl_element())

    # Next, interpolate the coordinates onto the nodes of W.
    X = interpolate(mesh.coordinates, W_vector)
    print("X[:,1]", X.dat.data[:,1])
    # Make an output function.
    stress_open_boundary_dynamic = Function(W)

    # Use the external data function to interpolate the values of f.
    stress_open_boundary_dynamic.dat.data[:] = dynamic_pressure_interp(X.dat.data_ro[:,1])

    # test stress open_boundary
    #sop = Function(W)
    #sop.interpolate(-g*(Temperature_term + Salinity_term))
    #sop_file = File(folder+"boundary_stress.pvd")
    #sop_file.write(sop)


    vp_bcs = {4: {'un': no_normal_flow, 'drag': ice_drag}, 2: {'stress': stress_open_boundary},
              3: {'un': no_normal_flow, 'drag': 0.0025}, 1: {'un': no_normal_flow}}
    #u_bcs = {2: {'q': Constant(0.0)}}

    temp_bcs = {4: {'flux': -mp.T_flux_bc}, 2: {'qadv': T_intercept + T_slope * z}}

    sal_bcs = {4: {'flux': -mp.S_flux_bc}, 2: {'qadv': S_intercept + S_slope * z}}



    # STRONGLY Enforced BCs
    # open ocean (RHS): no tangential flow because viscosity of outside ocean resists vertical flow.
    strong_bcs = []#DirichletBC(M.sub(0).sub(1), 0, 2)]

    ##########

    # Solver parameters
    mumps_solver_parameters = {
        'snes_monitor': None,
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij',
        'snes_max_it': 100,
        'snes_rtol': 1e-8,
        'snes_atol': 1e-6,
    }

    vp_solver_parameters = mumps_solver_parameters
    u_solver_parameters = mumps_solver_parameters
    temp_solver_parameters = mumps_solver_parameters
    sal_solver_parameters = mumps_solver_parameters

    ##########

    # Plotting depth profiles.
    z500m = cavity_thickness(5E2, 0., H1, L, H2)
    z1km = cavity_thickness(1E3, 0., H1, L, H2)
    z2km = cavity_thickness(2E3, 0., H1, L, H2)
    z4km = cavity_thickness(4E3, 0., H1, L, H2)
    z6km = cavity_thickness(6E3, 0., H1, L, H2)


    z_profile500m = np.linspace(z500m-water_depth-1., 1.-water_depth, 50)
    z_profile1km = np.linspace(z1km-water_depth-1., 1.-water_depth, 50)
    z_profile2km = np.linspace(z2km-water_depth-1., 1.-water_depth, 50)
    z_profile4km = np.linspace(z4km-water_depth-1., 1.-water_depth, 50)
    z_profile6km = np.linspace(z6km-water_depth-1., 1.-water_depth, 50)


    depth_profile500m = []
    depth_profile1km = []
    depth_profile2km = []
    depth_profile4km = []
    depth_profile6km = []

    for d5e2, d1km, d2km, d4km, d6km in zip(z_profile500m, z_profile1km, z_profile2km, z_profile4km, z_profile6km):
        depth_profile500m.append([5E2, d5e2])
        depth_profile1km.append([1E3, d1km])
        depth_profile2km.append([2E3, d2km])
        depth_profile4km.append([4E3, d4km])
        depth_profile6km.append([6E3, d6km])

    velocity_depth_profile500m = pd.DataFrame()
    velocity_depth_profile1km = pd.DataFrame()
    velocity_depth_profile2km = pd.DataFrame()
    velocity_depth_profile4km = pd.DataFrame()
    velocity_depth_profile6km = pd.DataFrame()

    velocity_depth_profile500m['Z_profile'] = z_profile500m
    velocity_depth_profile1km['Z_profile'] = z_profile1km
    velocity_depth_profile2km['Z_profile'] = z_profile2km
    velocity_depth_profile4km['Z_profile'] = z_profile4km
    velocity_depth_profile6km['Z_profile'] = z_profile6km


    def depth_profile_to_csv(profile, df, depth, t_str):
        #df['U_t_' + t_str] = u.at(profile)
        vw = np.array(v_.at(profile))
        vv = vw[:, 0]
        ww = vw[:, 1]
        df['V_t_' + t_str] = vv
        df['W_t_' + t_str] = ww
        if mesh.comm.rank == 0:
            df.to_csv(folder+depth+"_profile.csv")

    #### VOM for doing objective function...

    z_profile_adjoint = np.linspace(-525, -595, 15)
    z_profile_nodes = []
    for zi in z_profile_adjoint:
        z_profile_nodes.append([7550, zi])

    print(z_profile_nodes)
    z_profile_nodes = VertexOnlyMesh(mesh,z_profile_nodes,missing_points_behaviour='warn')

    DG0_vom_profile = FunctionSpace(z_profile_nodes, "DG", 0)
    temp_vom_profile = Function(DG0_vom_profile)
    sal_vom_profile = Function(DG0_vom_profile)

    temp_vom_profile.interpolate(temp)
    sal_vom_profile.interpolate(sal)


    adjoint_profile7550m = pd.DataFrame()
    adjoint_profile7550m['Z_profile'] = z_profile_adjoint

    melt_node = VertexOnlyMesh(mesh, [[7550, -522.7]], missing_points_behaviour='warn')
    DG0_vom_meltnode = FunctionSpace(melt_node, "DG", 0)
    melt_vom_node = Function(DG0_vom_meltnode)

    melt_vom_node.interpolate(mp.wb)

    def adjoint_profile_to_csv(df, t_str):
        df['T_t_' + t_str] = temp_vom_profile.dat.data_ro
        df['S_t_' + t_str] = sal_vom_profile.dat.data_ro
        if mesh.comm.rank == 0:
            df.to_csv(folder+"TS_foradjoint_7550m_profile.csv")

    adjoint_melt7550m = pd.DataFrame()
    def adjoint_melt_to_csv(df, t_str):
        df['Melt_t_' + t_str] = melt_vom_node.dat.data_ro
        if mesh.comm.rank == 0:
            df.to_csv(folder+"melt_foradjoint_7550m_profile.csv")

    ##### 
    # read linear simulations
    #target_folder = "/data/2d_adjoint/17.02.23_3_eq_param_ufric_dt300.0_dtOutput86400.0_T4320000.0_ip50.0_tres86400.0constant_Kh0.25_Kv0.001_structured_dy500_dz2_no_limiter_nosponge_open_qadvbc_pdyn_linTS/"
    adjoint_profile7550m_target = pd.read_csv("TS_foradjoint_7550m_profile_250423.csv")

    temp_profile_target = adjoint_profile7550m_target['T_t_14400']
    sal_profile_target = adjoint_profile7550m_target['S_t_14400']


    temp_vom_profile_target = Function(DG0_vom_profile)
    sal_vom_profile_target = Function(DG0_vom_profile)

    temp_vom_profile_target.dat.data[:] = temp_profile_target[:]
    sal_vom_profile_target.dat.data[:] = sal_profile_target[:]


    print("temp vom", temp_vom_profile.dat.data[:])

    print("temp vom target", temp_vom_profile_target.dat.data[:])

    ##########

    # define time steps
    dt = 300.0 
    T = 3000.0

    ##########

    # Set up time stepping routines
    vp_timestepper = CrankNicolsonSaddlePointTimeIntegrator([mom_eq, cty_eq], m, vp_fields, vp_coupling, dt, vp_bcs,
                                                            solver_parameters=vp_solver_parameters, strong_bcs=strong_bcs)

    #u_timestepper = DIRK33(u_eq, u, u_fields, dt, u_bcs, solver_parameters=u_solver_parameters)
    temp_timestepper = DIRK33(temp_eq, temp, temp_fields, dt, temp_bcs, solver_parameters=temp_solver_parameters)
    sal_timestepper = DIRK33(sal_eq, sal, sal_fields, dt, sal_bcs, solver_parameters=sal_solver_parameters)

    ##########

    # Set up folder



    # Depth profiles
    #depth_profile_to_csv(depth_profile500m, velocity_depth_profile500m, "500m", '0.0')
    #depth_profile_to_csv(depth_profile1km, velocity_depth_profile1km, "1km", '0.0')
    #depth_profile_to_csv(depth_profile2km, velocity_depth_profile2km, "2km", '0.0')
    #depth_profile_to_csv(depth_profile4km, velocity_depth_profile4km, "4km", '0.0')
    #depth_profile_to_csv(depth_profile6km, velocity_depth_profile6km, "6km", '0.0')

    ########


    # Begin time stepping
    t = 0.0
    step = 0


    while t < T - 0.5*dt:
        with timed_stage('velocity-pressure'):
            vp_timestepper.advance(t)
            #u_timestepper.advance(t)
        with timed_stage('temperature'):
            temp_timestepper.advance(t)
        with timed_stage('salinity'):
            sal_timestepper.advance(t)
        
        # dynamic pressure rhs
        temp_vom.interpolate(temp)
        sal_vom.interpolate(sal)
        p_dyn = dynamic_pressure(temp_vom.dat.data, sal_vom.dat.data)
        dynamic_pressure_interp = scipy.interpolate.interp1d(z_rhs, p_dyn, bounds_error=False,fill_value='extrapolate') 

        stress_open_boundary_dynamic.dat.data[:] = dynamic_pressure_interp(X.dat.data_ro[:,1])
        
        step += 1
        t += dt

    melt.project(mp.wb)
    #J = assemble(mp.wb*ds(4))
    N = len(temp_vom_profile.dat.data_ro)
    temp_vom_profile.interpolate(temp)
    sal_vom_profile.interpolate(sal)

    scale = Constant(1.0 / (0.5 * beta_sal)**2)
    alpha = 0.1
    J = assemble(scale*0.5/N * ((-beta_temp*(temp_vom_profile - temp_vom_profile_target))**2 + (beta_sal*(sal_vom_profile - sal_vom_profile_target))**2) * dx)  #+ assemble(0.5*alpha*scale*((beta_sal**2)*inner(grad(S_restorefield), grad(S_restorefield))+ (beta_temp**2)*inner(grad(T_restorefield), grad(T_restorefield)) )*dx)


    def eval_cb_pre(m):
        print("eval_cb_pre")
        print(len(m))
        print("T slope:", m[0].values()[0])
        print("T intercept:", m[1].values()[0])
        print("S slope:", m[2].values()[0])
        print("S intercept:", m[3].values()[0])
        tape.add_block(DiagnosticConstantBlock(T_slope, "T slope :"))
        tape.add_block(DiagnosticConstantBlock(T_intercept, "T intercept :"))
        tape.add_block(DiagnosticConstantBlock(S_slope, "S slope :"))
        tape.add_block(DiagnosticConstantBlock(S_intercept, "S intercept :"))


    def eval_cb(j, m):
        print("eval_cb")

        print(len(m))
        print("T slope:", m[0].values()[0])
        print("T intercept:", m[1].values()[0])
        print("S slope:", m[2].values()[0])
        print("S intercept:", m[3].values()[0])

        print("J = ", j)
        with DumbCheckpoint(folder+"dump.h5", mode=FILE_UPDATE) as chk:
            # Checkpoint file open for reading and writing
            chk.store(v_, name="v_velocity")
            chk.store(p_, name="perturbation_pressure")
            #chk.store(u, name="u_velocity")
            chk.store(temp, name="temperature")
            chk.store(sal, name="salinity")
            chk.store(T_restorefield_vis, name="temp restore")
            chk.store(S_restorefield_vis, name="sal restore")
    print(J)

    def derivative_cb_post(j, djdm, m):
        print("------------")
        print(" derivative cb post")
        print("------------")

    rf = ReducedFunctional(J, [c3]) #, c1,c2,c3], eval_cb_post=eval_cb, eval_cb_pre=eval_cb_pre, derivative_cb_post=derivative_cb_post)

    bounds = [[-0.02, -10, -0.01, 30],[0.02, 10., 0., 40. ] ]

    #g_opt = minimize(rf, bounds=bounds, options={"disp": True})

    #tape.reset_variables()
    #J.adj_value = 1.0

    J.block_variable.adj_value = 1.0
    #tape.visualise()
    # evaluate all adjoint blocks to ensure we get complete adjoint solution
    # currently requires fix in dolfin_adjoint_common/blocks/solving.py:
    #    meshtype derivative (line 204) is broken, so just return None instead
    #with timed_stage('adjoint'):
     #  tape.evaluate_adj()

    print(len(T_restorefield.dat.data))

    #grad = rf.derivative()
    #File(folder+'grad.pvd').write(grad)

    h = Constant(S_intercept)#S_restorefield)
    h.dat.data[:] = np.random.random(h.dat.data_ro.shape) # 2* for temp... 
    tt = taylor_test(rf, S_intercept, h)
    print("Sintercept TT ten steps")

    print(tt)
    assert np.allclose(tt, [2.0, 2.0, 2.0], rtol=5e-2)


