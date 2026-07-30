# Data access and attribution

Thermodense is code-first: external inputs are downloaded or supplied locally
under ignored `data/` paths. Do not commit or redistribute raw provider
datasets, credentials, cookies, or publisher PDFs from this repository.

| Source | Access and attribution summary |
| --- | --- |
| [TU Delft thermosphere density](https://thermosphere.tudelft.nl/data/data/version_02/) | Attribute TU Delft and review its stated [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) terms. |
| [NOAA GML CO₂](https://gml.noaa.gov/ccgg/trends/data.html) | NOAA describes the data as CC0 and requests citation of the source. |
| [CelesTrak space weather](https://celestrak.org/SpaceData/SW-All.txt) | Download from CelesTrak and attribute it. Its [format](https://celestrak.org/SpaceData/SpaceWx-format.php) is prepared locally as `data/original/space_weather/SW-All.csv`. |
| [HASDM](https://spacewx.com/hasdm/) | Research access is required; redistribution is not assumed. |
| [TIMED/SABER](https://saber.gats-inc.com/) | Follow publication and acknowledgement guidance; this project makes no redistribution claim. |
| [Space-Track](https://www.space-track.org/documentation) | Requires provider credentials and compliance with its citation and use terms; data are not redistributed here. |
| [pymsis](https://github.com/space-physics/pymsis) / NRLMSIS | pymsis is MIT licensed; NRLMSIS model code and data have separate terms and citation requirements. |
| [WACCM-X](https://www.ukssdc.ac.uk/waccm-x/) | Future-only. Licensing and redistribution status must be verified before use or distribution. |
| [Orekit](https://www.orekit.org/) | Apache-2.0; selected as the future JB implementation, subject to its license and source-data terms. |

Before publishing results, verify the current provider terms and cite the
underlying dataset and relevant model publications. This document is a working
summary, not legal advice or a substitute for provider documentation.
