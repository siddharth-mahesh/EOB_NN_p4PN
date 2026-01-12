# Todo list for first Neural EOB paper

[x] Build the NN architecture.
[] Trajectory-based training - Train on SEOBNRv5 uncalibrated trajectories to learn PN dynamics.
    [] Build a higher PN dataloader. Either use SEOBNRv5 or NRPy+ to generate trajectories.
    [] Train and compare effective potentials.
[] Waveform-based training - Train on NR waveforms to infer dynamics and waveform modes
    [x] Build dataloaders for NRSur and SXS.
    [] Train and visualize effective potentials.
[] Code organization.
    [] Add documentation.
[] Literature.
    [] Introduction
    [] Setup
    [] Results
    [] Discussion
    [] Conclusion