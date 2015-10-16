#ifndef CENTRAL2D_H
#define CENTRAL2D_H

#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <iostream>
#include <omp.h>
//ldoc on
/**
 * # Jiang-Tadmor central difference scheme
 * 
 * [Jiang and Tadmor][jt] proposed a high-resolution finite difference
 * scheme for solving hyperbolic PDE systems in two space dimensions.
 * The method is particularly attractive because, unlike many other
 * methods in this space, it does not require that we write any
 * solvers for problems with special initial data (so-called Riemann
 * problems), nor even that we compute Jacobians of the flux
 * functions.
 * 
 * While this code is based loosely on the Fortran code at the end of
 * Jiang and Tadmor's paper, we've written the current code to be
 * physics-agnostic (rather than hardwiring it to the shallow water
 * equations -- or the Euler equations in the Jiang-Tadmor paper).
 * If you're interested in the Euler equations, feel free to add your
 * own physics class to support them!
 * 
 * [jt]: http://www.cscamm.umd.edu/tadmor/pub/central-schemes/Jiang-Tadmor.SISSC-98.pdf
 * 
 * ## Staggered grids
 * 
 * The Jiang-Tadmor scheme works by alternating between a main grid
 * and a staggered grid offset by half a step in each direction.
 * Understanding this is important, particularly if you want to apply
 * a domain decomposition method and batch time steps between
 * synchronization barriers in your parallel code!
 * 
 * In even-numbered steps, the entry `u(i,j)` in the array of solution
 * values represents the average value of a cell centered at a point
 * $(x_i,y_j)$.  At the following odd-numbered step, the same entry
 * represents values for a cell centered at $(x_i + \Delta x/2, y_j +
 * \Delta y/2)$.  However, whenever we run a simulation, we always take
 * an even number of steps, so that outside the solver we can just think
 * about values on the main grid.  If `uold` and `unew` represent the
 * information at two successive *even* time steps (i.e. they represent
 * data on the same grid), then `unew(i,j)` depends indirectly on
 * `u(p,q)` for $i-3 \leq p \leq i+3$ and $j-3 \leq q \leq j+3$.
 * 
 * We currently manage this implicitly: the arrays at even time steps
 * represent cell values on the main grid, and arrays at odd steps
 * represent cell values on the staggered grid.  Our main `run` 
 * function always takes an even number of time steps to ensure we end
 * up on the primary grid.
 * 
 * ## Interface
 * 
 * We want a clean separation between the physics, the solver,
 * and the auxiliary limiter methods used by the solver.  At the same
 * time, we don't want to pay the overhead (mostly in terms of lost
 * optimization opportunities) for calling across an abstraction
 * barrier in the inner loops of our solver.  We can get around this
 * in C++ by providing the solver with *template arguments*, resolved
 * at compile time, that describe separate classes to implement the
 * physics and the limiter.
 *
 * The `Central2D` solver class takes two template arguments:
 * `Physics` and `Limiter`.  For `Physics`, we expect the name of a class
 * that defines:
 * 
 *  - A type for numerical data (`real`)
 *  - A type for solution and flux vectors in each cell (`vec`)
 *  - A flux computation function (`flux(vec& F, vec& G, const vec& U)`)
 *  - A wave speed computation function 
 *    (`wave_speed(real& cx, real& cy, const vec& U)`).
 * 
 * The `Limiter` argument is a type with a static function `limdiff`
 * with the signature
 * 
 *         limdiff(fm, f0, fp)
 * 
 * The semantics are that `fm`, `f0`, and `fp` are three successive
 * grid points in some direction, and the function returns an approximate
 * (scaled) derivative value from these points.
 * 
 * The solver keeps arrays for the solution, flux values, derivatives
 * of the solution and the fluxes, and the solution at the next time
 * point.  We use the C++ `vector` class to manage storage for these
 * arrays; but since we want to think of them as 2D arrays, we also
 * provide convenience functions to access them with multiple indices
 * (though we maintain C-style 0-based indexing).  The internal arrays
 * are padded with ghost cells; the ghost cell in the lower left corner
 * of the domain has index (0,0).
 */

template <class Physics, class Limiter>
class Central2D {
public:
    typedef typename Physics::real real;
    typedef typename Physics::vec  vec;

    Central2D(real w, real h,     // Domain width / height
              int nx, int ny, // Number of cells in x/y (without ghosts)
              int time_steps, // number of time steps: added to be able to declare sub_sim as an object of sim
			  real cfl = 0.45) :  // Max allowed CFL number
        nx(nx), ny(ny), time_steps(time_steps),
	w(w), h(h),
        nx_all(nx + 2*time_steps*nghost),
        ny_all(ny + 2*time_steps*nghost),
        dx(w/nx), dy(h/ny),
        cfl(cfl), 
        u_h_   ( nx_all * ny_all),
        u_hu_  ( nx_all * ny_all),
        u_hv_  ( nx_all * ny_all),
        f0_    ( nx_all * ny_all),
        f1_    ( nx_all * ny_all),
        f2_    ( nx_all * ny_all),
        g0_    ( nx_all * ny_all),
        g1_    ( nx_all * ny_all),
        g2_    ( nx_all * ny_all),
        ux_h_  ( nx_all * ny_all),
        ux_hu_ ( nx_all * ny_all),
        ux_hv_ ( nx_all * ny_all),
        uy_h_  ( nx_all * ny_all),
        uy_hu_ ( nx_all * ny_all),
        uy_hv_ ( nx_all * ny_all),
        fx0_   ( nx_all * ny_all),
        fx1_   ( nx_all * ny_all),
        fx2_   ( nx_all * ny_all),
        gy0_   ( nx_all * ny_all),
        gy1_   ( nx_all * ny_all),
        gy2_   ( nx_all * ny_all),
        v_h_   ( nx_all * ny_all),
        v_hu_  ( nx_all * ny_all),
        v_hv_  ( nx_all * ny_all),
        uh_h_   ( nx_all * ny_all),
        uh_hu_  ( nx_all * ny_all),
        uh_hv_  ( nx_all * ny_all)
		{}

    // Advance from time 0 to time tfinal
    void run(real tfinal);

    // Call f(Uxy, x, y) at each cell center to set initial conditions
    // f1 sets the initial conditions for h, f2 does hu, and f3 does hv
    template <typename F>
    void init(F f1, F f2, F f3);

    // Diagnostics
    void solution_check();

    // Array size accessors
    int xsize() const { return nx; }
    int ysize() const { return ny; }
    
    // Read / write elements of simulation state
    real& operator()(int i, int j) {
        return u_h_[offset(i+time_steps*nghost,j+time_steps*nghost)];
    }
    
    const real& operator()(int i, int j) const {
        return u_h_[offset(i+time_steps*nghost,j+time_steps*nghost)];
    }
    
private:
    static constexpr int nghost = 3;   // Number of ghost cells
    const real w; //added those as variables of object as we need to pass them to children
    const real h;
    const int time_steps;
    const int nx, ny;         // Number of (non-ghost) cells in x/y
    const int nx_all, ny_all; // Total cells in x/y (including ghost)
    const real dx, dy;        // Cell size in x/y
    const real cfl;           // Allowed CFL number

    std::vector<real> u_h_;   // h component of solution
    std::vector<real> u_hu_;  // hu component of solution
    std::vector<real> u_hv_;  // hv component of solution

    std::vector<real> f0_;    // First component of flux in x
    std::vector<real> f1_;    // Second component of flux in x
    std::vector<real> f2_;    // Third component of flux in x

    std::vector<real> g0_;    // First component of flux in y
    std::vector<real> g1_;    // Second component of flux in y
    std::vector<real> g2_;    // Third component of flux in y

    std::vector<real> ux_h_;  // x differences of u
    std::vector<real> ux_hu_; // x differences of u
    std::vector<real> ux_hv_; // x differences of u

    std::vector<real> uy_h_;  // y differences of u
    std::vector<real> uy_hu_; // y differences of u
    std::vector<real> uy_hv_; // y differences of u

    std::vector<real> fx0_;   // x differences of f
    std::vector<real> fx1_;   // x differences of f
    std::vector<real> fx2_;   // x differences of f

    std::vector<real> gy0_;   // y differences of g
    std::vector<real> gy1_;   // y differences of g
    std::vector<real> gy2_;   // y differences of g

    std::vector<real> v_h_;   // h component of solution values at next step
    std::vector<real> v_hu_;  // hu component of solution values at next step
    std::vector<real> v_hv_;  // hv component of solution values at next step
	
    std::vector<real> uh_h_;   // h component of solution values at half step
    std::vector<real> uh_hu_;  // hu component of solution values at half step
    std::vector<real> uh_hv_;  // hv component of solution values at half step
	
    // Array accessor functions

    int offset(int ix, int iy) const { return iy*nx_all+ix; }

    real& u_h(int ix, int iy)   { return u_h_[offset(ix,iy)];   }
    real& u_hu(int ix, int iy)  { return u_hu_[offset(ix,iy)];  }
    real& u_hv(int ix, int iy)  { return u_hv_[offset(ix,iy)];  }

    real& v_h(int ix, int iy)   { return v_h_[offset(ix,iy)];   }
    real& v_hu(int ix, int iy)  { return v_hu_[offset(ix,iy)];  }
    real& v_hv(int ix, int iy)  { return v_hv_[offset(ix,iy)];  }

	real& uh_h(int ix, int iy)   { return uh_h_[offset(ix,iy)];   }
    real& uh_hu(int ix, int iy)  { return uh_hu_[offset(ix,iy)];  }
    real& uh_hv(int ix, int iy)  { return uh_hv_[offset(ix,iy)];  }
		
    real& f0(int ix, int iy)    { return f0_[offset(ix,iy)];    }
    real& f1(int ix, int iy)    { return f1_[offset(ix,iy)];    }
    real& f2(int ix, int iy)    { return f2_[offset(ix,iy)];    }

    real& g0(int ix, int iy)    { return g0_[offset(ix,iy)];    }
    real& g1(int ix, int iy)    { return g1_[offset(ix,iy)];    }
    real& g2(int ix, int iy)    { return g2_[offset(ix,iy)];    }

    real& ux_h(int ix, int iy)  { return ux_h_[offset(ix,iy)];  }
    real& ux_hu(int ix, int iy) { return ux_hu_[offset(ix,iy)]; }
    real& ux_hv(int ix, int iy) { return ux_hv_[offset(ix,iy)]; }

    real& uy_h(int ix, int iy)  { return uy_h_[offset(ix,iy)];  }
    real& uy_hu(int ix, int iy) { return uy_hu_[offset(ix,iy)]; }
    real& uy_hv(int ix, int iy) { return uy_hv_[offset(ix,iy)]; }

    real& fx0(int ix, int iy)   { return fx0_[offset(ix,iy)];   }
    real& fx1(int ix, int iy)   { return fx1_[offset(ix,iy)];   }
    real& fx2(int ix, int iy)   { return fx2_[offset(ix,iy)];   }

    real& gy0(int ix, int iy)   { return gy0_[offset(ix,iy)];   }
    real& gy1(int ix, int iy)   { return gy1_[offset(ix,iy)];   }
    real& gy2(int ix, int iy)   { return gy2_[offset(ix,iy)];   }

    // Wrapped accessor (periodic BC)
    int ioffset(int ix, int iy) {
        return offset( (ix+nx-nghost) % nx + nghost,
                       (iy+ny-nghost) % ny + nghost );
    }

    // vec& uwrap(int ix, int iy)  { return u_[ioffset(ix,iy)]; }
    real& u_h_wrap(int ix, int iy)  { return u_h_[ioffset(ix,iy)]; }
    real& u_hu_wrap(int ix, int iy)  { return u_hu_[ioffset(ix,iy)]; }
    real& u_hv_wrap(int ix, int iy)  { return u_hv_[ioffset(ix,iy)]; }

    // Apply limiter to all components in a vector
    static void limdiff(real& du, const real& um, const real& u0, const real& up) {
        du = Limiter::limdiff(um, u0, up);
    }

    // Stages of the main algorithm
    void apply_periodic();
    void compute_fg_speeds(real& cx, real& cy);
    void limited_derivs();
    void compute_step(int io, real dt);
    void init_smallgrid( Central2D<Physics, Limiter>& sub_sim, int s, int size_ratio );
    void map_to_maingrid( Central2D<Physics, Limiter>& sub_sim, int s, int size_ratio );

};


/**
 * ## Initialization
 * 
 * Before starting the simulation, we need to be able to set the
 * initial conditions.  The `init` function does exactly this by
 * running a callback function at the center of each cell in order
 * to initialize the cell $U$ value.  For the purposes of this function,
 * cell $(i,j)$ is the subdomain 
 * $[i \Delta x, (i+1) \Delta x] \times [j \Delta y, (j+1) \Delta y]$.
 */

template <class Physics, class Limiter>
template <typename F>
void Central2D<Physics, Limiter>::init(F f0, F f1, F f2)
{
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix)
            f0(u_h(ix,iy), (ix+0.5)*dx, (iy+0.5)*dy);
    }

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix)
            f1(u_hu(ix,iy), (ix+0.5)*dx, (iy+0.5)*dy);
    }

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix)
            f2(u_hv(ix,iy), (ix+0.5)*dx, (iy+0.5)*dy);
    }
    
}

/**
 * ## Time stepper implementation
 * 
 * ### Boundary conditions
 * 
 * In finite volume methods, boundary conditions are typically applied by
 * setting appropriate values in ghost cells.  For our framework, we will
 * apply periodic boundary conditions; that is, waves that exit one side
 * of the domain will enter from the other side.
 * 
 * We apply the conditions by assuming that the cells with coordinates
 * `nghost <= ix <= nx+nghost` and `nghost <= iy <= ny+nghost` are
 * "canonical", and setting the values for all other cells `(ix,iy)`
 * to the corresponding canonical values `(ix+p*nx,iy+q*ny)` for some
 * integers `p` and `q`.
 */

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::apply_periodic()
{
    // Copy data between right and left boundaries
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nghost; ++ix) {
            u_h(ix,          iy) = u_h_wrap(ix,          iy);
            u_h(nx+nghost+ix,iy) = u_h_wrap(nx+nghost+ix,iy);
        }

    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nghost; ++ix) {
            u_hu(ix,          iy) = u_hu_wrap(ix,          iy);
            u_hu(nx+nghost+ix,iy) = u_hu_wrap(nx+nghost+ix,iy);
        }

    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nghost; ++ix) {
            u_hv(ix,          iy) = u_hv_wrap(ix,          iy);
            u_hv(nx+nghost+ix,iy) = u_hv_wrap(nx+nghost+ix,iy);
        }

    // Copy data between top and bottom boundaries
    for (int ix = 0; ix < nx_all; ++ix)
        for (int iy = 0; iy < nghost; ++iy) {
            u_h(ix,          iy) = u_h_wrap(ix,          iy);
            u_h(ix,ny+nghost+iy) = u_h_wrap(ix,ny+nghost+iy);
        }

    for (int ix = 0; ix < nx_all; ++ix)
        for (int iy = 0; iy < nghost; ++iy) {
            u_hu(ix,          iy) = u_hu_wrap(ix,          iy);
            u_hu(ix,ny+nghost+iy) = u_hu_wrap(ix,ny+nghost+iy);
        }

    for (int ix = 0; ix < nx_all; ++ix)
        for (int iy = 0; iy < nghost; ++iy) {
            u_hv(ix,          iy) = u_hv_wrap(ix,          iy);
            u_hv(ix,ny+nghost+iy) = u_hv_wrap(ix,ny+nghost+iy);
        }
}


/**
 * ### Initial flux and speed computations
 * 
 * At the start of each time step, we need the flux values at
 * cell centers (to advance the numerical method) and a bound
 * on the wave speeds in the $x$ and $y$ directions (so that
 * we can choose a time step that respects the specified upper
 * bound on the CFL number).
 */

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::compute_fg_speeds(real& cx_, real& cy_)
{
    using namespace std;
    real cx = 1.0e-15;
    real cy = 1.0e-15;

    // Update the FU[0] component
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            Physics::flux_f00(f0(ix,iy), u_hu(ix,iy));
        }

    // Update the FU[1] component with hu
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            Physics::flux_f10(f1(ix,iy), u_hu(ix,iy));
        }

    // Update the FU[1] component with h
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            Physics::flux_f12(f1(ix,iy), u_h(ix,iy));
        }
    
    // Update the FU[2] component with hv
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            Physics::flux_f20(f2(ix,iy), u_hu(ix,iy));
        }
    
    // Update the FU[2] component with hv
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            Physics::flux_f21(f2(ix,iy), u_hv(ix,iy));
        }

    // Update the FU[2] component with h
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            Physics::flux_f22(f2(ix,iy), u_h(ix,iy));
        }
            
    // Update the GU[0] component with hv
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            Physics::flux_g00(g0(ix,iy), u_hv(ix,iy));
        }

    // Update the GU[1] component with hv
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            Physics::flux_g10(g1(ix,iy), u_hv(ix,iy));
        }

    // Update the GU[1] component with hu
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            Physics::flux_g10(g1(ix,iy), u_hu(ix,iy));
        }

    // Update the GU[1] component with h
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            Physics::flux_g12(g1(ix,iy), u_h(ix,iy));
        }
    
    // Update the GU[2] component with hv
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            Physics::flux_g20(g2(ix,iy), u_hv(ix,iy));
        }

    // Update the GU[2] component with h
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            Physics::flux_g22(g2(ix,iy), u_h(ix,iy));
        }
    
    //TODO: Maybe there is a better way to vectorize this?
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            real cell_cx, cell_cy;
            Physics::wave_speed(cell_cx, cell_cy, u_h(ix,iy), u_hu(ix,iy), u_hv(ix,iy));
            cx = max(cx, cell_cx);
            cy = max(cy, cell_cy);
        }
    cx_ = cx;
    cy_ = cy;
}

/**
 * ### Derivatives with limiters
 * 
 * In order to advance the time step, we also need to estimate
 * derivatives of the fluxes and the solution values at each cell.
 * In order to maintain stability, we apply a limiter here.
 */

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::limited_derivs()
{
    // x derivatives
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix)
            limdiff( ux_h(ix,iy), u_h(ix-1,iy), u_h(ix,iy), u_h(ix+1,iy) );

    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix)
            limdiff( ux_hu(ix,iy), u_hu(ix-1,iy), u_hu(ix,iy), u_hu(ix+1,iy) );

    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix)
            limdiff( ux_hv(ix,iy), u_hv(ix-1,iy), u_hv(ix,iy), u_hv(ix+1,iy) );

    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix)
            limdiff( fx0(ix,iy), f0(ix-1,iy), f0(ix,iy), f0(ix+1,iy) );

    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix)
            limdiff( fx1(ix,iy), f1(ix-1,iy), f1(ix,iy), f1(ix+1,iy) );

    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix)
            limdiff( fx2(ix,iy), f2(ix-1,iy), f2(ix,iy), f2(ix+1,iy) );


    // y derivatives
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix)
            limdiff( uy_h(ix,iy), u_h(ix,iy-1), u_h(ix,iy), u_h(ix,iy+1) );

    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix)
            limdiff( uy_hu(ix,iy), u_hu(ix,iy-1), u_hu(ix,iy), u_hu(ix,iy+1) );

    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix)
            limdiff( uy_hv(ix,iy), u_hv(ix,iy-1), u_hv(ix,iy), u_hv(ix,iy+1) );

    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix)
            limdiff( gy0(ix,iy), g0(ix,iy-1), g0(ix,iy), g0(ix,iy+1) );

    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix)
            limdiff( gy1(ix,iy), g1(ix,iy-1), g1(ix,iy), g1(ix,iy+1) );

    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix)
            limdiff( gy2(ix,iy), g2(ix,iy-1), g2(ix,iy), g2(ix,iy+1) );
}


/**
 * ### Advancing a time step
 * 
 * Take one step of the numerical scheme.  This consists of two pieces:
 * a first-order corrector computed at a half time step, which is used
 * to obtain new $F$ and $G$ values; and a corrector step that computes
 * the solution at the full step.  For full details, we refer to the
 * [Jiang and Tadmor paper][jt].
 * 
 * The `compute_step` function takes two arguments: the `io` flag
 * which is the time step modulo 2 (0 if even, 1 if odd); and the `dt`
 * flag, which actually determines the time step length.  We need
 * to know the even-vs-odd distinction because the Jiang-Tadmor
 * scheme alternates between a primary grid (on even steps) and a
 * staggered grid (on odd steps).  This means that the data at $(i,j)$
 * in an even step and the data at $(i,j)$ in an odd step represent
 * values at different locations in space, offset by half a space step
 * in each direction.  Every other step, we shift things back by one
 * mesh cell in each direction, essentially resetting to the primary
 * indexing scheme.
 */

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::compute_step(int io, real dt)
{
    real dtcdx2 = 0.5 * dt / dx;
    real dtcdy2 = 0.5 * dt / dy;
    // Predictor (flux values of f and g at half step)
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {
	    uh_h(ix,iy)=u_h(ix,iy);
            uh_h(ix, iy) -= dtcdx2 * fx0(ix, iy);
            uh_h(ix, iy) -= dtcdy2 * gy0(ix, iy);
			
	    uh_hu(ix,iy)=u_hu(ix,iy);
            uh_hu(ix, iy) -= dtcdx2 * fx1(ix, iy);
            uh_hu(ix, iy) -= dtcdy2 * gy1(ix, iy);
			
	    uh_hv(ix,iy)=u_hv(ix,iy);			
            uh_hv(ix, iy) -= dtcdx2 * fx2(ix, iy);
            uh_hv(ix, iy) -= dtcdy2 * gy2(ix, iy);
 	       
        	if (!(uh_h(ix,iy)>0)){ printf("at uh comp, i:%d, j%d, h:%g, h before:%g fx0 %g, gyo %g, \n", ix,iy,uh_h(ix,iy), u_h(ix,iy), fx0(ix, iy), gy0(ix, iy)); assert(0);}
		}
    // Update the FU[0] component
    for (int iy = 1; iy < ny_all -1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {
            Physics::flux_f00(f0(ix,iy), uh_hu(ix,iy));
        }

    // Update the FU[1] component with hu
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {
            Physics::flux_f10(f1(ix,iy), uh_hu(ix,iy));
        }

    // Update the FU[1] component with h
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {
            Physics::flux_f12(f1(ix,iy), uh_h(ix,iy));
        }
    
    // Update the FU[2] component with hu
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {
            Physics::flux_f20(f2(ix,iy), uh_hu(ix,iy));
        }
    
    // Update the FU[2] component with hv
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {
            Physics::flux_f21(f2(ix,iy), uh_hv(ix,iy));
        }

    // Update the FU[2] component with h
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {
            Physics::flux_f22(f2(ix,iy), uh_h(ix,iy));
        }
            
    // Update the GU[0] component with hv
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {
            Physics::flux_g00(g0(ix,iy), uh_hv(ix,iy));
        }

    // Update the GU[1] component with hv
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {
            Physics::flux_g10(g1(ix,iy), uh_hv(ix,iy));
        }

    // Update the GU[1] component with hu
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {
            Physics::flux_g11(g1(ix,iy), uh_hu(ix,iy));
        }

    // Update the GU[1] component with h
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {
            Physics::flux_g12(g1(ix,iy), uh_h(ix,iy));
        }
    
    // Update the GU[2] component with hv
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {
            Physics::flux_g20(g2(ix,iy), uh_hv(ix,iy));
        }

    // Update the GU[2] component with h
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {
            Physics::flux_g22(g2(ix,iy), uh_h(ix,iy));
        }

    // Corrector for h component (finish the step)
    for (int iy = nghost-io; iy < ny_all-nghost-io; ++iy)
        for (int ix = nghost-io; ix < nx_all-nghost-io; ++ix) {
                v_h(ix,iy) =
                    0.2500 * ( u_h(ix,  iy) + u_h(ix+1,iy)      +
                               u_h(ix,iy+1) + u_h(ix+1,iy+1))   -
                    0.0625 * ( ux_h(ix+1,iy  ) - ux_h(ix,iy)    +
                               ux_h(ix+1,iy+1) - ux_h(ix,iy+1)  +
                               uy_h(ix,  iy+1) - uy_h(ix,  iy)  +
                               uy_h(ix+1,iy+1) - uy_h(ix+1,iy)) -
                    dtcdx2 * ( f0(ix+1,iy  )   - f0(ix,iy)      +
                               f0(ix+1,iy+1)   - f0(ix,iy+1))   -
                    dtcdy2 * ( g0(ix,  iy+1)   - g0(ix,  iy)    +
                               g0(ix+1,iy+1)   - g0(ix+1,iy)); 
	
        	if (!(v_h(ix,iy)>0)){ printf("at v comp, i:%d, j%d, h:%g \n", ix,iy,v_h(ix,iy)); assert(0);}
	}
    // Corrector for hu component (finish the step)
    for (int iy = nghost-io; iy < ny_all-nghost-io; ++iy)
        for (int ix = nghost-io; ix < nx_all-nghost-io; ++ix) {
                v_hu(ix,iy) =
                    0.2500 * ( u_hu(ix,  iy) + u_hu(ix+1,iy)      +
                               u_hu(ix,iy+1) + u_hu(ix+1,iy+1))   -
                    0.0625 * ( ux_hu(ix+1,iy  ) - ux_hu(ix,iy)    +
                               ux_hu(ix+1,iy+1) - ux_hu(ix,iy+1)  +
                               uy_hu(ix,  iy+1) - uy_hu(ix,  iy)  +
                               uy_hu(ix+1,iy+1) - uy_hu(ix+1,iy)) -
                    dtcdx2 * ( f1(ix+1,iy  )   - f1(ix,iy)      +
                               f1(ix+1,iy+1)   - f1(ix,iy+1))   -
                    dtcdy2 * ( g1(ix,  iy+1)   - g1(ix,  iy)    +
                               g1(ix+1,iy+1)   - g1(ix+1,iy));
        }

    // Corrector for hv component (finish the step)
    for (int iy = nghost-io; iy < ny_all-nghost-io; ++iy)
        for (int ix = nghost-io; ix < nx_all-nghost-io; ++ix) {
                v_hv(ix,iy) =
                    0.2500 * ( u_hv(ix,  iy) + u_hv(ix+1,iy)      +
                               u_hv(ix,iy+1) + u_hv(ix+1,iy+1))   -
                    0.0625 * ( ux_hv(ix+1,iy  ) - ux_hv(ix,iy)    +
                               ux_hv(ix+1,iy+1) - ux_hv(ix,iy+1)  +
                               uy_hv(ix,  iy+1) - uy_hv(ix,  iy)  +
                               uy_hv(ix+1,iy+1) - uy_hv(ix+1,iy)) -
                    dtcdx2 * ( f2(ix+1,iy  )   - f2(ix,iy)      +
                               f2(ix+1,iy+1)   - f2(ix,iy+1))   -
                    dtcdy2 * ( g2(ix,  iy+1)   - g2(ix,  iy)    +
                               g2(ix+1,iy+1)   - g2(ix+1,iy));
        }

    // Copy from v storage back to main grid
    for (int j = nghost; j < ny_all-nghost; ++j){
        for (int i = nghost; i < nx_all-nghost; ++i){
            u_h(i,j) = v_h(i-io,j-io);
        }
    }
    for (int j = nghost; j < ny_all-nghost; ++j){
        for (int i = nghost; i < nx_all-nghost; ++i){
            u_hu(i,j) = v_hu(i-io,j-io);
        }
    }
    for (int j = nghost; j < ny_all-nghost; ++j){
        for (int i = nghost; i < nx_all-nghost; ++i){
            u_hv(i,j) = v_hv(i-io,j-io);
        }
    }
}


/**
 * ### Advance time
 * 
 * The `run` method advances from time 0 (initial conditions) to time
 * `tfinal`.  Note that `run` can be called repeatedly; for example,
 * we might want to advance for a period of time, write out a picture,
 * advance more, and write another picture.  In this sense, `tfinal`
 * should be interpreted as an offset from the time represented by
 * the simulator at the start of the call, rather than as an absolute time.
 * 
 * We always take an even number of steps so that the solution
 * at the end lives on the main grid instead of the staggered grid. 
 */

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::run(real tfinal)
{

    
    bool done = false;
    real t = 0;
    int size_ratio=10; // big/small
    int sub_size = nx/size_ratio; // size of subdomain
    int sub_number = nx*nx/sub_size/sub_size;
    int time_steps= 10; // number of time steps done before synchronisation -- MUST BE EVEN
    printf("sub_number is: %d \n w: %g, w/ratio: %g \n", sub_number, w, w/size_ratio);
    while (!done) { 
		printf("not done");
        	real dt;
		real cx, cy;
		compute_fg_speeds(cx, cy);
		cx = 1.0*cx; // overestimating cx and cy as we wont be recomputing it for the next #time_steps steps
		cy=1.0*cy;
		dt = cfl / std::max(cx/dx, cy/dy);
		if (t+time_steps*dt >= tfinal){ // if the next #time_steps steps bring us to the end, set dt to be 1/time_steps of that
			dt = (tfinal-t)/time_steps; // could probably make this better by having two different dt's -- could have at most (time_steps -1) unnecessarily calls
		}
		#pragma omp parallel for 
		//shared(u_h, u_hv, u_hu, maxspeed) private(s, sub_sim)\ i think this is not needed? not sure
		for(int s=0; s < sub_number; ++s){
			Central2D<Physics, Limiter> sub_sim(w/size_ratio, h/size_ratio, sub_size, sub_size, time_steps);// builds sub-simulation on smaller grid
			init_smallgrid(sub_sim, s, size_ratio);
			real local_cx, local_cy;
			for (int io = 0; io < time_steps; ++io) {
				sub_sim.compute_fg_speeds(local_cx, local_cy);
				sub_sim.limited_derivs(); 
				sub_sim.compute_step(io%2, dt);
			}
			map_to_maingrid( sub_sim, s, size_ratio);
		}
		if(t+time_steps*dt==tfinal){
			done=true;}
		else{t+=time_steps*dt;}
	}	
}
/**
 * ### Diagnostics
 * 
 * The numerical method is supposed to preserve (up to rounding
 * errors) the total volume of water in the domain and the total
 * momentum.  Ideally, we should also not see negative water heights,
 * since that will cause the system of equations to blow up.  For
 * debugging convenience, we'll plan to periodically print diagnostic
 * information about these conserved quantities (and about the range
 * of water heights).
 */

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::solution_check()
{
    using namespace std;
    real h_sum = 0, hu_sum = 0, hv_sum = 0;
    real hmin = u_h(nghost,nghost);
    real hmax = hmin;
    for (int j = 0; j < ny_all; ++j) // changed this to whole grid as there is no ghost cell in main grid
        for (int i = 0; i < nx_all; ++i) {
            real h = u_h(i,j);
            h_sum += h;
            hu_sum += u_hu(i,j);
            hv_sum += u_hv(i,j);
            hmax = max(h, hmax);
            hmin = min(h, hmin);
            if (!(h>0)){printf("at i:%d, j:%d , h= %g \n",i,j, h); } 
	    assert( h > 0) ;
        }
    real cell_area = dx*dy;
    h_sum *= cell_area;
    hu_sum *= cell_area;
    hv_sum *= cell_area;
    printf("-\n  Volume: %g\n  Momentum: (%g, %g)\n  Range: [%g, %g]\n",
           h_sum, hu_sum, hv_sum, hmin, hmax);
}

/** This function intializes a sub_simulation grid to part of a main simulation 
* grid.
* s is the index of the subgrid inside of the main grid
*/
template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::init_smallgrid( Central2D<Physics, Limiter>& sub_sim, int s, int size_ratio ){
	int ycoor= (s/size_ratio)*sub_sim.nx;
	int xcoor = (s % size_ratio)*sub_sim.nx;
	int t = sub_sim.time_steps;
	int x,y;

	for( int i=0; i < sub_sim.nx_all; ++i){
		for( int j=0; j < sub_sim.ny_all; ++j){
			if( xcoor -t*nghost+i < 0){
				x=nx+(xcoor-t*nghost+i);
			}else if( xcoor -t*nghost+i >= nx){
				x=xcoor-t*nghost+i - nx;
			}else{ x = xcoor-t*nghost+i; }	
			
			if( ycoor -t*nghost+j < 0){
				y=ny+(ycoor-t*nghost+j);
			}else if( ycoor -t*nghost+j >= ny){
				y=ycoor-t*nghost+j - ny;
			}else{ y = ycoor-t*nghost+j; }	
			if (u_h(x,y)==0){printf("init small grid fails x=%d, y=%d, for %d, %d \n",x,y,xcoor-t*nghost+i, ycoor-t*nghost+j); assert (0);}		
			sub_sim.u_h(i,j)=u_h(x,y);
			sub_sim.u_hu(i,j)=u_hu(x,y);
			sub_sim.u_hv(i,j)=u_hv(x,y);
	
		}
	}
	
}

/** This function maps the smaller grid back to their position on the big grid
* does not include the ghost cells of the small grid
*
*/
template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::map_to_maingrid( Central2D<Physics, Limiter>& sub_sim, int s, int size_ratio ){
	int ycoor= (s/size_ratio)*sub_sim.nx;
	int xcoor = (s % size_ratio)*sub_sim.nx;
	int t = sub_sim.time_steps;


	for( int i=0; i < sub_sim.nx; ++i){
		for( int j=0; j < sub_sim.nx; ++j){
			u_h(xcoor+i,ycoor+j)= sub_sim.u_h(i+t*nghost,j+t*nghost);
			u_hu(xcoor+i,ycoor+j)= sub_sim.u_hu(i+t*nghost,j+t*nghost);
			u_hv(xcoor+i,ycoor+j)= sub_sim.u_hv(i+t*nghost,j+t*nghost);
			
		}
	}
	
}



//ldoc off
#endif /* CENTRAL2D_H*/
