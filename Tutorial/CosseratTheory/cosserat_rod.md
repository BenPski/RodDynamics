---
title: Cosserat Rod
---

$$
\newcommand{\real}{\mathbb{R}}
\newcommand{\der}[2]{\frac{d#1}{d#2}}
\newcommand{\pder}[2]{\frac{\partial#1}{\partial#2}}
\newcommand{\D}[1]{\Delta#1}
\newcommand{\inner}[2]{\langle#1, #2\rangle}
$$

In the simulations we focus on specializing Cosserat theory to rods or slender bodies. For a rod we simplify to one spatial parameter, $s$, and the cross sections are assumed to be rigid planes. This leads to some simplifications that I'll go through.

# Kinematics

For the kinematics we still have a homogeneous transformation matrix describing the configuration of every point in the body. However, now that we have a single spatial parameter, $s\in[0,L]$ where $L$ is the length of the rod in the reference configuration, we can alter the definition slightly. First, we can determine the centerline which is the curve passing through all the centroids of the cross sections parameterized by $s$. It is not necessary to make the centerline pass through the centroid of the cross sections (just each value of $s$ should be associated uniquely with a cross section), but doing so leads to some simplifications in the derivations of the physics. Note that $s$ can be defined over any bounded interval it is just most common to define it over $[0,L]$.

For the centerline we have the kinematics and configuration specified as:

\begin{equation}
    g(s) = \begin{bmatrix} R(s) & p(s) \\ 0 & 1 \end{bmatrix}
\end{equation} 
where $R\in SO(3)$ and $p\in\real^3$ are the rotation and position repectively. 

Then, since we assume that the cross sections are rigid the configuration for any point is just a translation away from $g$:
\begin{equation}
    G(x,y,s) = g(s)\begin{bmatrix} I_{3\times3} & r(x,y) \\ 0 & 1 \end{bmatrix}
\end{equation}
where $G$ is the configuration of any point in the body, $x$ and $y$ are for locating the position over a cross section, and $r$ is the displacement from the centerline. It is possible to include a rotation rather than an identity matrix to account for things such as fibers in the body, but assuming that the body is heterogeneous this isn't necessary. Then, do to the fact that $g(s)$ was defined to be at the centroid of the cross sections we have:
\begin{equation}
    \iint r dA = 0
\end{equation}
due to symmetry. This will be useful in determining the inertia and stiffness later.

With only one spatial parameter we only have 2 twists: the spatial twist, $\xi$, and the temporal twist, $\eta$.
\begin{gather}
    \pder{}{s}g = g' = g\hat{\xi} \\
    \pder{}{t}g = \dot{g} = g\hat{\eta}
\end{gather}

The relationship between twists is then:
\begin{equation}
    \dot{\xi} = \eta' + ad_\xi\eta
\end{equation}

So, the kinematics simplifies a bit from the most general case.

# Deformations

The deformation gradient and strain definitions simplify a bit as well primarily due to the cross sections being assumed rigid. 

The deformation gradient becomes:
\begin{equation}
    F = R \begin{bmatrix} \vec{x} & \vec{y} & \nu -\hat{r}\omega \end{bmatrix} 
\end{equation} 
where $\vec{x} = [1, 0, 0]^T$, $\vec{y} = [1, 0, 0]^T$, and $\xi = [\nu, \omega]^T$. For the sake of simplifying the notation I introduce:
\begin{gather}
    \gamma + \gamma^{*} = (\D{\nu} - \hat{r}\D{\omega}) + (\nu^{*} - \hat{r}\omega^{*})
\end{gather}
where $\D{}$ stands for change in and $*$ denotes the value in the reference configuration. This allows $F$ to be written:
\begin{equation}
    F = R \begin{bmatrix} \vec{x} & \vec{y} & \gamma + \gamma^{*} \end{bmatrix}
\end{equation}

The Green-Lagrange strain, $\varepsilon$, follows as:

\begin{equation}
    \varepsilon = \frac{1}{2} (F^TF - (F^{*})^TF^{*})
\end{equation}

Breaking the strain into components we get:
\begin{gather}
    \varepsilon_{xx} = \varepsilon_{yy} = \varepsilon_{xy} = 0 \\
    \varepsilon_{xs} = \inner{\gamma+\gamma^{*}}{\vec{x}} \\
    \varepsilon_{ys} = \inner{\gamma+\gamma^{*}}{\vec{y}} \\
    \varepsilon_{ss} = \frac{1}{2} \inner{\gamma}{\gamma} + \inner{\gamma}{\gamma^{*}}\\
\end{gather}
typically $\gamma^{*} = [0, 0, 1]^T$ as that corresponds to a straight reference configuration.

# Physics

For the physics we still use Hamilton's principle, but we can determine simpler forms of the kinetic energy, $T$, and the strain energy, $U$.

## Kinetic Energy

For the kinetic energy we have:
\begin{equation}
    T = \frac{1}{2}\iiint \rho \inner{\dot{p}_G}{\dot{p}_G}dx\,dy\,ds
\end{equation}
where $p_G$ is the position component of $G$. Defining the linear and angular components of $\eta=[w, v]^T$ we can expand the kinetic energy as:
\begin{equation}
    T = \frac{1}{2}\iiint \rho \inner{v - \hat{r}w}{v-\hat{r}w} dx\,dy\,ds = \frac{1}{2}\iiint \rho \inner{v}{v} - 2\inner{v}{\hat{r}w} - \inner{w}{\hat{r}^2w} dx\,dy\,ds 
\end{equation}

Separating out terms that are independent of integration variables we have:
\begin{equation}
    T = \frac{1}{2} \left( \int \rho \inner{v}{v} A ds + 2 \int \inner{v}{\hat{w}\iint r dx\,dy} ds - \int \inner{w}{(\iint \hat{r}^2 dx\,dy) w} ds \right)
\end{equation}

Using the centroid definition this simplifies and rearranges to:
\begin{equation}
    T = \frac{1}{2} \int \inner{\eta}{M\eta} ds
\end{equation}
where $M=rho*diag([I_x,I_y,J,A,A,A])$ with $I_x$ and $I_y$ are the second moments of area, $J$ is the polar second moment of area, and $A$ is the cross sectional area. $M$ is interpreted as the inertia matrix for the cross section.

## Strain Energy

For the strain energy we have a bit more involved derivation. To determine the stress, $\sigma$, we need a constitutive material law and with the Green-Lagrange strain we have quadratic terms to consider. Taking advantage of the 0's in the strain and using a similar process to the kinetic energy we can simplify the strain energy as:
\begin{equation}
    U = \frac{1}{2} \int \inner{\D{\xi}}{\Psi(\xi)} ds
\end{equation}
where $\Psi$ is the wrench on the cross section due to the deformation. $\Psi$ can be rather complex depending on the chosen material properties; however, if we use a linear constitutive law and assume the quadratic terms in the strain are negligible then the strain energy simplifies to:
\begin{equation}
    U = \frac{1}{2} \int \inner{\D{\xi}}{K\D{\xi}} ds
\end{equation}
where $K=diag([EI_x,EI_y,GJ,GA,GA,EA])$ is the stiffness matrix for the cross section with $E$ as the Young's modulus and $G$ as the shear modulus. This is a very convenient and simple approximation to the strain energy that we will use throughout.

## Dynamics Equations

When evaluating Hamilton's principle given the definitions for the kinetic and strain energy we get the dynamics in the weak form. Converting to the strong form we have:
\begin{equation}
(M\dot{\eta} - ad_\eta^TM\eta) - ((K\D{\xi})' - ad_\xi^TK\D{\xi}) + \bar{W} = 0 
\end{equation}

Then, including the kinematics the PDEs describing the dynamcis are:
\begin{gather}
    (M\dot{\eta} - ad_\eta^TM\eta) - ((K\D{\xi})' - ad_\xi^TK\D{\xi}) + \bar{W} = 0 \\
    g' = g\hat{\xi} \\
    \dot{g} = g\hat{\eta} \\
    \dot{\xi} = \eta' + ad_\xi\eta
\end{gather}

For the statics we simply remove the time dependent terms.

## External Loads

The external loads are included in $\bar{W}$ the components of which are actuation loads, gravity, viscosity, etc. I won't be going through their derivation here, but will mention them when relevant.

