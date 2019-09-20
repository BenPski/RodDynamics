---
title: "Cosserat Theory: Introduction"
---

$$
\newcommand{\der}[2]{\frac{d#1}{d#2}}
\newcommand{\pder}[2]{\frac{\partial#1}{\partial#2}}
\newcommand{\inner}[2]{\langle #1, #2 \rangle}
$$

# Overview

A terse introduction:

Cosserat theory is an approach to modeling deformable bodies that handles large deformations (finite strains) and treats the base element of the body as having more structure than a point particle (microstructure). For Cosserat theories the body is assumed to be made of infinitesimal rigid bodies which are described by a position and orientation and are also known as micropolar theories. Each element has an orthonormal frame attached to it referred to as the directors which allows the orientation to be determined relative to a fixed frame and the position is the translation between the fixed frame and the director frame, the director frame will also be referred to as the body frame. The location in the body will be parameterized by spatial parameters, $s_i$, that are usually defined over some bounded interval. For a 3D body there will be 3 spatial parameters, though in some restrictions (e.g., surfaces and rods) these can be renamed and simplified for convenience. 

Geometrically exact theories are special cases of Cosserat theories, I'm not entirely sure of what the distinction is and can't find the initial place where the distinction was made. It seems like it depends on what stress-strain pair is used though I'm not too confident, if they are energy conjugates then the theory is geometrically exact. In our case I use the Green-Lagrange strain and the 2nd Piola-Kirchhoff stress which are energy conjugates and therefore should fall under the geometrically exact theories.
 
There are generalizations that exist, such as allowing the director frame to not be orthonormal, but the Cosserat approach seems to be a solid foundation for working with in soft robotics. Green-Naghdi theory and other microstructural theories are generalizations to include more deformation modes (e.g., inflation) and theories with less deformations are Kirchhoff theory and the elastica both of which primarily focus on rods. 
 
# Kinematics

For each element in the body we have an orientation and a position to describe it's configuration relative to a fixed frame. The orientation is described by a rotation matrix, $R\in SO(3)$, and the position is a vector, $p\in\mathbb{R}^3$. We can merge these into a homogeneous transformation matrix $g\in SE(3)$:

\begin{equation}
    g = \begin{bmatrix} R & p \\ 0 & 1 \end{bmatrix}
\end{equation}

Since $g$ is an element of the Special Euclidean group, $SE(3)$, we can define the derivatives as:
\begin{equation}
    \pder{}{x} g = g\hat{\xi}
\end{equation}
where $x$ is some smooth parameter for the Lie group (e.g., time or space) and $\hat{\xi}$ is the Lie algebra element of $se(3)$. The $\hat{}$ denotes the isomorphism between the vector and matrix representation of the algebra element, $\hat{}: \mathbb{R}^6 \rightarrow \mathbb{R}^{4\times4}$. Here we used the left derivative rather than the right derivative so that the algebra elements can be interpreted to be in the body frame. 

For the time derivative we define:
\begin{equation}
    \pder{}{t} g = \dot{g} = g\hat{\eta}
\end{equation}
where $\eta$ is the temporal twist or generalized velocity. Here I'm using terminology borrowed from Screw theory, a twist in Screw theory is the generalized velocity for a rigid body where it includes both the angular and linear velocities. Since there are derivatives both in time and space for a soft body I differentiate them by calling the temporal or spatial twists.  

For the space derivatives:
\begin{equation}
    \pder{}{s_i} g = g'_i = g\hat{\xi}_i
\end{equation}
where $i$ has been used to distinguish which spatial parameter is being referred to and $\xi$ is the spatial twist. 

We can also use the fact that partial derivatives commute to derive relationships between algebra elements.

\begin{align}
    \pder{}{x_i}\pder{}{x_j} g &= \pder{}{x_j}\pder{}{x_i} g \\
    \pder{}{x_i}(g\hat{\xi}_j) &= \pder{}{x_j}(g\hat{\xi}_i) \\
    g\hat{\xi}_i\hat{\xi}_j + g\pder{}{x_i}\hat{\xi}_j &= g\hat{\xi}_j\hat{\xi}_i + g\pder{}{x_j}\hat{\xi}_i \\
    \pder{}{x_i}\hat{\xi}_j &= \pder{}{x_j}\hat{\xi}_i + \left[\hat{\xi}_j, \hat{\xi}_i \right] \\
    \pder{}{x_i}\xi_j &= \pder{}{x_j}\xi_i + ad_{\xi_j}\xi_i
\end{align}
this allows us to relate the derivatives of various twists together, for example a temporal and spatial twist:

\begin{equation}
    \dot{\xi} = \eta' + ad_\xi\eta
\end{equation}
where $'$ is the spatial derivative.

This summarizes the kinematics for a body with Cosserat theory. 

# Deformations

To describe the deformations of a body we can use the deformation gradient (not actually a gradient) and the Green-Lagrange strain tensor to measure the strain. 

The deformation gradient is defined by the change in the position relative to its original position, for Cosserat theory we have:

\begin{equation}
    F = \begin{bmatrix} \pder{p_x}{s_1} & \pder{p_y}{s_2} & \pder{p_z}{s_3} \end{bmatrix}
\end{equation}
where $F$ is the deformation gradient, the subscripted $p$'s are the components of the position, and the $s$'s are the spatial parameters. Generally the spatial parameters should be selected so that $F = I_{3\times3}$ in the reference configuration, but that isn't necessary.

For the Green-Lagrange strain we have:
\begin{equation}
    \varepsilon = \frac{1}{2}(F^TF - (F^{*})^TF^{*})
\end{equation}
where the $*$'s indicate the value in the reference configuration.

The Green-Lagrange strain will be used in the definition of the strain energy.

# Physics

For the physical behavior of the Cosserat body we can use Hamilton's principle to model the dynamics. One thing to note is that technically Hamilton's principal only applies to conservative systems and to include non-conservative loads we use the Lagrange-D'Alembert principle which are essentially the same in practice. Essentially Lagrange-D'Alembert adds virtual work to Hamilton's principle to incorporate non-conservative loads. 

For Hamilton's principle we have that the variation of the action integral is stationary which I will write as:
\begin{equation}
    \delta(\int_0^\tau L dt) = \int_0^\tau \inner{\delta h}{\bar{W}} dt
\end{equation}  
where $\delta()$ stands for the variation, $L$ is the Lagrangian, $\tau$ is the time horizon, $\delta h$ is the variation in the displacement, and $\bar{W}$ is the distributed wrench on the body. To keep Hamilton's principle valid (technically Lagrange-D'Alembert) we have to guarantee that the boundary conditions are not changed with the variation. 

The distributed wrench, $\bar{W}$, depends on the type of loads considered in the system; however, for the Lagrangian, $L$, I always use kinetic energy, $T$, and the strain energy, $U$. There are other conservative potentials that can be included, but the strain energy has the special property of being left-invariant (orientation does not matter) which is convenient in the derivation. To define the Lagrangian we have:

\begin{equation}
    L = T-U
\end{equation}

Then we define the kinetic energy, $T$, as:
\begin{equation}
    T = \frac{1}{2} \iiint \rho\inner{\dot{p}}{\dot{p}} ds_1\,ds_2\,ds_3
\end{equation}
where $\rho$ is the density of the material and $\dot{p}$ is the velocity of an arbitrary point in the body.

The strain energy, $U$, is:
\begin{equation}
    U = \frac{1}{2} \iiint \sigma \varepsilon ds_1\,ds_2\,ds_3
\end{equation}
where $\sigma$ is the 2nd Piola-Kirchoff stress. Usually this will be determined by the constitutive law chosen to represent the material properties.

This means Hamilton's principle can be restated as:
\begin{equation}
    \frac{1}{2}\delta(\int_0^\tau (\iiint \rho\inner{\dot{p}}{\dot{p}} - \sigma \varepsilon ds_1\,ds_2\,ds_3) dt) = \int_0^\tau \inner{\delta h}{\bar{W}} dt
\end{equation}

However, when specializing to rods or surfaces the energy relationships can be simplified. 

Evaluating the variations will define the dynamics relationship and removing any time dependent terms gives the statics. 