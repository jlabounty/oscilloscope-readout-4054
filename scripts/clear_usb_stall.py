#!/usr/bin/env python3
"""Clear a stalled USBTMC endpoint on the Tektronix DPO4054.

Run this when pyvisa reports '[Errno 32] Pipe error' on connect.
Equivalent to unplugging and replugging the USB-B cable.
"""
import usb.core
import usb.util

dev = usb.core.find(idVendor=0x0699, idProduct=0x0401)
if dev is None:
    raise SystemExit("DPO4054 not found on USB.")

usb.util.claim_interface(dev, 0)
dev.clear_halt(0x01)   # bulk OUT
dev.clear_halt(0x82)   # bulk IN
usb.util.release_interface(dev, 0)
print("Endpoint halts cleared — scope ready.")
