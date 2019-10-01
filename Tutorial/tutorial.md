---
title: Tutorial
---

This tutorial goes through the developed simulations for Cosserat rods. There are 3 main parts for learning about the simulation: the background theory, the description of the simulations and their development, and an example use case.

The background theory is found in the `CosseratTheory` directory. Cosserat theory is described and the specific case of Cosserat rods are described as well which are what are implemented. An example of the currently implemented simulation is documented in the `example`.  

The actual implementation of the simulations and their development is detailed in the `Simulations` directory which is split into several sections. Each section contains its own relevant code and a description both in `html` and `pdf` form. The most developed code is found in the `python` directory at the root of the repository and should correspond to the latest section in the tutorial. 

# Cosserat Theory

## Cosserat Theory

Describes the idea behind a Cosserat theory

## Cosserat Rod

The model for a Cosserat material restricted to the rod case.

# Tutorial

## Initial

Goes through the implementation of a cantilever rod subject to now external forces and the code is written in a not very general way.

## First Abstraction

Runs the same simulation, but begins to generalize the code framework for easier use later.

## Loads

Starts to look at external loads. In this case viscosity and gravity.

## Cables

Implements cable loads to demonstrate controlling actuator inputs to the simulations.

## Load Generalizations

Implements a general interface for implementing different kinds of loads for the simulations.

## Segments

Develops separating the rod into multiple serial segments so that point loads and systems with multiple segments can be modelled.

## Other Boundary Conditions

Looks at implementing simulations for boundary conditions other than cantilevered. Note that these do not work well and would need a different simulation approach.

# Example

Goes through an example for the simulation.