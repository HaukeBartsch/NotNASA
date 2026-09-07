# This is not NASA - Weather simulation

Instructions to Qwen3.8 27B: Generate a weather simulation for planet earth. General goal is to simulate CO2 distribution over time from fire and high density populated areas.

1) Create a discretized spherical model of the planet (scaled by a variable factor). Simplify the finite element model between ground-level (higher resolution at ground, lower resolution at the oceans) and at athmospheric height of 100km (lowest uniform resolution). Implement a convection algorithms that can transfers CO2 in this finite element grid. Add sources that simulate forest fires and CO2 sources from highly populated areas.

2) Show the result of the simulation as an animated rendered movie. Use ffmpeg or three.js, geojson or alternatives. Render the athosphaeric concentraction of CO2 as smoke or clouds up to a height of 100km. The clouds should animate infront of a surface representation of planet earth. Visually, the CO2 plumes emmited from the fires should be moving based on general wind pattern in the northers and southers hemisphere.

The simulation can cover a 1 year period and should be detailed enough to display the daily cycle of CO2 generated mostly by dense population areas and forest fires - with fires flaring up during daytime and dying down during the night.

**Folder guide:** [sim/](sim/README.md) (physics engine) · [server/](server/README.md) (API bridge) · [viewer/](viewer/README.md) (renderer) · [capture/](capture/README.md) (movie capture) · [tests/](tests/README.md) (invariant checks) · [data/](data/README.md) (input data)

# CO₂ Weather Simulation + Movie — Implementation Plan

**Goal:** Simulate one year of atmospheric CO₂ transport on Earth from two source classes (diurnal forest fires, diurnal dense-population emissions), on a true finite-element model over an unstructured spherical mesh (100 km–1000 km adjustable, refined over land, vertical to 100 km), then render it as an animated three.js movie of smoke/clouds over a globe and export MP4.

**Stack (user-chosen):** Python 3.14 + numpy/scipy simulation engine → FastAPI bridge → three.js (npm, v0.185.1) browser viewer → Playwright headless capture + ffmpeg (9.0.1, libx264 verified) for the MP4. All local, all verified present on this Mac (arm64; 704 GiB free).

**Verified environment facts:** Python 3.14.7 (Homebrew, externally managed → venv required; `uv` 0.12.7 available), Node v26.7.0 / npm 11.19.0, ffmpeg 9.0.1 with libx264, cp314/arm64 wheels exist for numpy 2.5.2, scipy 1.18.1, numba 0.67.0 (optional speedup), pillow 12.3.0, fastapi 0.141.1, playwright 1.62.0. Natural Earth `ne_110m_land.geojson` (public domain, 138 KB, 127 polygons) fetch-verified.

The simulation of a full year uses less than 15min and about 100mb of main memory.


# Daily CO2 cycle in cities

Urban CO2 emissions typically peak and trough at specific times of the day:

The Morning Peak (7:00 AM – 9:00 AM): This is usually the sharpest spike of the day. It is heavily driven by the morning commute (ground transportation) and households waking up, cooking, and turning on utilities.

The Midday Dip (11:00 AM – 3:00 PM): Emissions often level off or dip slightly in the middle of the day. While commercial building use is high, traffic congestion drops. Concurrently, urban parks and greenery actively absorb CO2 via photosynthesis, mitigating some local emissions.

The Evening Peak (5:00 PM – 8:00 PM): A second, broader peak occurs as the evening rush hour brings commuters home, and residential energy use for heating, cooling, lighting, and cooking surges.

The Nighttime Trough (11:00 PM – 5:00 AM): Emissions plummet to their lowest levels as traffic disappears, businesses close, and the city sleeps.

<img width="1280" height="720" alt="night-time image" src="https://github.com/user-attachments/assets/08f58b7e-d7c0-4754-b8aa-374e74293017" />
