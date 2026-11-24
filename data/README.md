# Data

Trajectory data of the 10 Chattanooga camera views used in the paper, from about nine minutes of video per view at 10 frames per second. A trajectory is the sequence of box centres of one tracked object, in pixels of the camera image.

| Folder | Files | Content |
|---|---|---|
| `raw_trajectories/` | 20 | Output of step 1 (track preprocessing): one training and one test file per view |
| `processed_06/` | 12 | Processed files of the six intersections, with the standardization constants of the six-intersection setting |
| `processed_23/` | 20 | Processed files of the 10 Chattanooga views, with the standardization constants of the 23-view setting |

`CHECKSUMS.txt` holds the SHA-256 checksum of every file (`sha256sum -c CHECKSUMS.txt` from this folder).

## Views

| Camera view | In the six-intersection setting | Training tracks | Training windows | Test tracks | Test windows |
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

- `raw_trajectories/<view>_{train,test}.pkl`: a pickled pandas table with the columns `frame`, `trackId`, `x`, `y` (pixels), `sceneId`, `metaId`. Training and test files of a view share no track.
- `processed_*/<view>_{train,test}.pkl`: a pickled environment object as used by FlowChain and Trajectron++ (written with `dill`). It holds, for every track, the position divided by 50, the velocity and the acceleration, and the standardization constants. Training files also hold 24 rotated copies of the scene for augmentation, which is why they are much larger than the test files. To load a file, `src/` must be on the Python path, as the launchers set it.

The files of the same view in `processed_06/` and `processed_23/` hold the same trajectories and differ only in the standardization constants.

## The other 13 views of the 23-view setting

The 23-view setting also uses 11 views of CityFlow (C002, C003, C004, C005, C028, C041, C042, C043, C044, C045, C046) and 2 views of the TUMTraf Intersection dataset, sequence R02_S03 (s110_camera_basler_south1_8mm, s110_camera_basler_south2_8mm). Because of the usage restrictions of these datasets, their files are not distributed here. To build them, request access to the datasets from their providers, run YOLO26 with ByteTrack on the videos of these views, and follow the data processing described in the main README.

The data files are Python pickles. Load pickles only from sources you trust.
