"""Compatibility shim for code written against the pre-1.0 Thalamus.

``RecordingDevice`` now lives in :mod:`thalamus.devices` and does considerably
more: it implements the socket loop, reconnection, timestamping, and drift-free
pacing, none of which your subclass has to write any more. The abstract method is
still ``start()``, so an existing device that implements ``start()`` itself keeps
working exactly as before — but a device that implements :meth:`samples` instead
gets all of the above for free.

Old:

    class SimulatedDevice(RecordingDevice):
        def __init__(self, device_id, interval=1.0, **kwargs):
            super().__init__(device_id, **kwargs)
            self.df = pd.read_csv('eeg.csv')
            self.entry_index = 0
        def start(self):
            with socket.socket(...) as sock:      # 15 more lines of socket handling
                ...

New:

    from thalamus import ReplayDevice
    ReplayDevice('eeg', 'eeg.csv', rate=250).run()

This module will be removed in a future release. Import from ``thalamus`` instead.
"""

import warnings

from thalamus.devices.base import RecordingDevice  # noqa: F401

warnings.warn(
    "importing from 'device_interface' is deprecated; use "
    "'from thalamus import RecordingDevice' instead. See examples/ for the new, "
    "much shorter way to write a device.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["RecordingDevice"]
