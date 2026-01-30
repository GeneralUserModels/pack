#!/usr/bin/env python3
"""Quick script to check display DPI and scale factor using various methods."""

import sys


def check_screeninfo():
    """Check DPI using screeninfo library."""
    print("=" * 50)
    print("Method 1: screeninfo")
    print("=" * 50)
    try:
        from screeninfo import get_monitors
        for i, m in enumerate(get_monitors()):
            print(f"\nMonitor {i}: {m.name}")
            print(f"  Resolution: {m.width} x {m.height} pixels")
            print(f"  Position: ({m.x}, {m.y})")
            if m.width_mm and m.height_mm:
                dpi_x = m.width / (m.width_mm / 25.4)
                dpi_y = m.height / (m.height_mm / 25.4)
                print(f"  Physical size: {m.width_mm} x {m.height_mm} mm")
                print(f"  Calculated DPI: {dpi_x:.1f} x {dpi_y:.1f}")
            else:
                print("  Physical size: Not available")
    except Exception as e:
        print(f"  Error: {e}")


def check_macos_quartz():
    """Check scale factor using macOS Quartz."""
    print("\n" + "=" * 50)
    print("Method 2: macOS Quartz (CGDisplay)")
    print("=" * 50)
    if sys.platform != "darwin":
        print("  Skipped: Not on macOS")
        return
    
    try:
        import Quartz
        main_display = Quartz.CGMainDisplayID()
        
        # Get pixel dimensions
        pixel_width = Quartz.CGDisplayPixelsWide(main_display)
        pixel_height = Quartz.CGDisplayPixelsHigh(main_display)
        
        # Get display bounds (in points, which may differ from pixels on Retina)
        bounds = Quartz.CGDisplayBounds(main_display)
        point_width = bounds.size.width
        point_height = bounds.size.height
        
        scale_factor = pixel_width / point_width
        
        print(f"  Pixel dimensions: {pixel_width} x {pixel_height}")
        print(f"  Point dimensions: {point_width:.0f} x {point_height:.0f}")
        print(f"  Scale factor: {scale_factor:.1f}x")
        print(f"  Is Retina: {'Yes' if scale_factor > 1 else 'No'}")
    except Exception as e:
        print(f"  Error: {e}")


def check_macos_appkit():
    """Check scale factor using macOS AppKit/NSScreen."""
    print("\n" + "=" * 50)
    print("Method 3: macOS AppKit (NSScreen)")
    print("=" * 50)
    if sys.platform != "darwin":
        print("  Skipped: Not on macOS")
        return
    
    try:
        from AppKit import NSScreen
        for i, screen in enumerate(NSScreen.screens()):
            backing_scale = screen.backingScaleFactor()
            frame = screen.frame()
            print(f"\nScreen {i}:")
            print(f"  Frame: {frame.size.width:.0f} x {frame.size.height:.0f} points")
            print(f"  Backing scale factor: {backing_scale}x")
            print(f"  Actual pixels: {frame.size.width * backing_scale:.0f} x {frame.size.height * backing_scale:.0f}")
    except Exception as e:
        print(f"  Error: {e}")


def check_mss():
    """Check what mss captures (actual screenshot size)."""
    print("\n" + "=" * 50)
    print("Method 4: mss (actual screenshot capture)")
    print("=" * 50)
    try:
        import mss
        with mss.mss() as sct:
            for i, monitor in enumerate(sct.monitors):
                print(f"\nMonitor {i}: {monitor}")
                if i > 0:  # Skip the "all monitors" entry
                    # Capture a tiny portion to check actual pixel size
                    region = {
                        "left": monitor["left"],
                        "top": monitor["top"],
                        "width": 100,
                        "height": 100
                    }
                    img = sct.grab(region)
                    print(f"  Requested 100x100, got: {img.width} x {img.height}")
                    if img.width != 100:
                        print(f"  → Detected scale factor: {img.width / 100}x")
    except Exception as e:
        print(f"  Error: {e}")


def recommend_scale():
    """Provide a recommended scale factor."""
    print("\n" + "=" * 50)
    print("RECOMMENDATION")
    print("=" * 50)
    
    scale = 1.0
    
    # Try mss method first (most reliable for actual capture size)
    try:
        import mss
        with mss.mss() as sct:
            if len(sct.monitors) > 1:
                monitor = sct.monitors[1]
                region = {
                    "left": monitor["left"],
                    "top": monitor["top"],
                    "width": 100,
                    "height": 100
                }
                img = sct.grab(region)
                if img.width != 100:
                    scale = img.width / 100
    except:
        pass
    
    # Fallback to AppKit on macOS
    if scale == 1.0 and sys.platform == "darwin":
        try:
            from AppKit import NSScreen
            scale = NSScreen.mainScreen().backingScaleFactor()
        except:
            pass
    
    print(f"\n  Detected display scale: {scale}x")
    print(f"\n  To save at logical resolution (1:1 with display points):")
    print(f"    --scale {1/scale:.2f}")
    print(f"\n  To save at native resolution (full pixels):")
    print(f"    --scale 1.0")


if __name__ == "__main__":
    print("\nDPI / Scale Factor Detection\n")
    
    check_screeninfo()
    check_macos_quartz()
    check_macos_appkit()
    check_mss()
    recommend_scale()
    
    print("\n")
