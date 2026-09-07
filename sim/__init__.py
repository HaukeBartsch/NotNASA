"""CO2 weather simulation package.

Modules:
    config   -- tunable constants (resolution, physics, time, vertical grid)
    mesh     -- conforming unstructured triangular mesh of the sphere
    wind     -- idealized global wind field (u, v horizontal + w vertical)
    land     -- rasterized land mask from Natural Earth GeoJSON
    sources  -- diurnal/seasonal fire and population CO2 sources
    engine   -- explicit FEM/FVM time-stepping core
    runner   -- CLI: run the yearly simulation and write frame data
"""
