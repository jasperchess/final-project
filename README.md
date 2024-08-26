## Looking at the results

All of the grid-search, and evaluations are fed into Tensorboard.

If you install the dependencies in the notebook and your terminal is in the same python venv you should be able to run Tensorboard by running:

`tensorboard --logdir tensorboard`

I have tried to keep the directory structure fairly straight forward. All of the segmentation models are top level, while the classification models are under the `/classification` namespace.

## Scratch Pad

There is another directory - [scratch_pads](./scratch_pads/README.md) which contains quick experiments to gather a better understanding of the domain