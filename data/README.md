# Data

Trajectory data of 10 camera views at intersections in Chattanooga, from about nine minutes of video per view at 10 frames per second. A trajectory is the sequence of box centres of one tracked object, in pixels of the camera image.

| Folder | Files | Content |
|---|---|---|
| `raw_trajectories/` | 20 | Raw trajectories: one training and one test file per view |
| `processed_06/` | 12 | Processed files of the six views in `configs/clients_06.txt`, with the constants of `configs/standardization_06.json` |
| `processed_23/` | 20 | Processed files of all 10 views, with the constants of `configs/standardization_23.json` |

The files of a view in `processed_06/` and `processed_23/` hold the same trajectories and differ only in the standardization constants. `CHECKSUMS.txt` holds the SHA-256 checksum of every file (`sha256sum -c CHECKSUMS.txt` from this folder). The files are Python pickles: load pickles only from sources you trust.

## Views

| Camera view | In `clients_06` | Training tracks | Training windows | Test tracks | Test windows |
|---|---|---|---|---|---|
| Broad | yes | 601 | 57,523 | 151 | 16,727 |
| Georgia_1 | no | 490 | 58,256 | 123 | 14,663 |
| Georgia_2 | yes | 384 | 41,525 | 96 | 9,364 |
| Hwy27_1 | yes | 356 | 34,971 | 90 | 11,034 |
| Hwy27_2 | no | 245 | 34,678 | 62 | 12,217 |
| Lindsay_1 | no | 309 | 32,272 | 78 | 6,748 |
| Lindsay_2 | yes | 336 | 25,062 | 84 | 5,483 |
| Market | yes | 778 | 62,076 | 195 | 18,876 |
| Pine_1 | no | 326 | 40,317 | 82 | 8,559 |
| Pine_2 | yes | 366 | 35,236 | 92 | 7,186 |

A window is one training or test sample: 8 observed positions (0.8 s) followed by 12 future positions (1.2 s).

## File formats

- `raw_trajectories/<view>_{train,test}.pkl`: a pickled pandas table with the columns `frame`, `trackId`, `x`, `y` (pixels), `sceneId`, `metaId`. The training and the test file of a view share no track.
- `processed_*/<view>_{train,test}.pkl`: a pickled environment object as used by FlowChain and Trajectron++ (written with `dill`), with the position, velocity and acceleration of every track and the standardization constants. Training files also hold rotated copies of the scene for augmentation, so they are much larger. To load a file, `src/` must be on the Python path.

## The other 13 views of `clients_23.txt`

`configs/clients_23.txt` also lists 11 views of CityFlow (C002, C003, C004, C005, C028, C041, C042, C043, C044, C045, C046) and 2 views of the TUMTraf Intersection dataset, sequence R02_S03 (s110_camera_basler_south1_8mm, s110_camera_basler_south2_8mm). Their files are not distributed here because of the usage restrictions of these datasets. To build them, request the datasets from their providers, run a detector and tracker on the videos (YOLO26 with ByteTrack in the paper), and follow "Prepare the data" and "Reproducing the paper" in the main README. Use `configs/standardization_23.json` in step 3 and write the files to `data/processed_23/` under the names in `configs/clients_23.txt`.
