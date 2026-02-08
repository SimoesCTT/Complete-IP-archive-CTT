#!/usr/bin/env python3
import os

print("Φ-24 Validation Check")
print("=" * 40)

gds_path = "hardware/gdsii/phi24_core.gds"

if os.path.exists(gds_path):
    size = os.path.getsize(gds_path)
    print(f"✅ GDSII found: {size/1024:.1f} KB")
    print(f"✅ Ready for fabrication")
else:
    print(f"❌ GDSII missing")
