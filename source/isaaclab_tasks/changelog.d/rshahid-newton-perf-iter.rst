Changed
^^^^^^^

* Changed direct in-hand manipulation tasks to use fused Newton reset, done, and reward
  kernels when available.
* Changed goal marker updates in direct in-hand manipulation tasks to skip USD marker
  synchronization when no GUI, RTX sensor, or active visualizer can observe the markers.
