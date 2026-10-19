| run | backbone | params | train (500 steps) | NFE 5 | NFE 10 | NFE 20 | NFE 50 | NFE 100 | 2nd sampler NFE 10 / 50 | s / 100 img (NFE 20) |
|---|---|---|---|---|---|---|---|---|---|---|
| `ddpm-eps` (`ddim`) | unet | 0.21 M | 4.0 min | 516.53 | 420.69 | 369.87 | 347.39 | 343.34 | `ddpm` 278.24 / 107.59 | 4.3 s |
| `ddpm-x0` (`ddim`) | unet | 0.21 M | 3.6 min | 2.09 | 1.41 | 1.19 | 1.06 | 1.06 | `ddpm` 1.42 / 1.04 | 5.9 s |
| `ddpm-v` (`ddim`) | unet | 0.21 M | 3.3 min | 5.91 | 3.12 | 2.43 | 2.20 | 2.17 | `ddpm` 2.51 / 1.71 | 6.5 s |
| `score-sde-vp` (`ode`) | unet | 0.21 M | 4.0 min | 372.30 | 477.73 | 539.30 | 561.07 | 574.30 | `em` 1400.73 / 75.44 | 3.9 s |
| `score-sde-ve` (`ode`) | unet | 0.21 M | 3.3 min | 1761.96 | 1553.18 | 489.96 | 436.06 | 492.07 | `em` 1913.14 / 266.46 | 3.8 s |
| `flow-matching` (`euler`) | unet | 0.21 M | 8.8 min | 4.24 | 2.43 | 1.92 | 1.68 | 1.62 | `heun` 1.67 / 1.56 | 10.7 s |
| `edm` (`heun`) | unet | 0.21 M | 4.0 min | 492.34 | 30.80 | 2.60 | 2.43 | 2.40 | `euler` 2.86 / 2.44 | 4.3 s |
| `flow-matching-cfg` (`euler`) | unet | 0.21 M | 3.4 min | 3.33 | 2.31 | 2.07 | 1.81 | 1.82 | – | 4.6 s |
| `flow-matching-dit` (`euler`) | dit | 1.25 M | 3.9 min | 13.75 | 15.35 | 16.54 | 17.26 | 17.62 | `heun` 16.68 / 17.50 | 4.9 s |

real vs real floor (200 training images vs the test set): 0.13

| guidance w | NFE | FID | label accuracy |
|---|---|---|---|
| 0 | 20 | 3.20 | 11.0% |
| 1 | 20 | 2.07 | 58.5% |
| 3 | 20 | 1.10 | 96.0% |
