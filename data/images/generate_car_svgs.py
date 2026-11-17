"""Generate high-resolution modern automotive vector graphics for Indian pre-owned cars.
Provides 100% offline-resilient, fast-rendering car visuals for the valuation dashboard.
"""

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# 1. Maruti Swift (Sporty Crimson Red Hatchback)
SWIFT_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 240" width="100%" height="100%">
  <defs>
    <linearGradient id="bg_swift" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1a0b10"/>
      <stop offset="100%" stop-color="#2a0f16"/>
    </linearGradient>
    <linearGradient id="car_red" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#dc2626"/>
      <stop offset="50%" stop-color="#ef4444"/>
      <stop offset="100%" stop-color="#b91c1c"/>
    </linearGradient>
    <linearGradient id="glass_swift" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.8"/>
      <stop offset="100%" stop-color="#0284c7" stop-opacity="0.3"/>
    </linearGradient>
  </defs>
  <rect width="600" height="240" rx="16" fill="url(#bg_swift)"/>
  <!-- Ground Shadow -->
  <ellipse cx="300" cy="205" rx="240" ry="14" fill="#000" opacity="0.5"/>
  <!-- Body Outline -->
  <path d="M 90,170 L 105,145 Q 120,135 150,130 L 220,105 Q 260,75 350,75 L 430,85 Q 470,110 500,140 L 520,165 Q 525,175 510,180 L 440,180 A 35,35 0 0,0 370,180 L 210,180 A 35,35 0 0,0 140,180 L 95,180 Z" fill="url(#car_red)"/>
  <!-- Roof / Glass Area -->
  <path d="M 230,105 L 345,82 L 420,90 L 465,130 L 230,130 Z" fill="url(#glass_swift)"/>
  <!-- Window Pillars -->
  <line x1="335" y1="83" x2="335" y2="130" stroke="#111" stroke-width="4"/>
  <!-- Headlight & Tail Light -->
  <polygon points="505,145 520,158 510,165 495,155" fill="#fef08a" opacity="0.9"/>
  <polygon points="98,145 92,158 100,165 106,155" fill="#f87171" opacity="0.9"/>
  <!-- Wheels -->
  <circle cx="175" cy="180" r="32" fill="#1e293b" stroke="#64748b" stroke-width="6"/>
  <circle cx="175" cy="180" r="16" fill="#cbd5e1"/>
  <circle cx="405" cy="180" r="32" fill="#1e293b" stroke="#64748b" stroke-width="6"/>
  <circle cx="405" cy="180" r="16" fill="#cbd5e1"/>
  <!-- Badging -->
  <text x="30" y="45" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="#f87171">MARUTI SUZUKI SWIFT</text>
  <text x="30" y="68" font-family="Arial, sans-serif" font-size="12" fill="#94a3b8">Sport Hatchback · High Market Liquidity</text>
  <rect x="470" y="25" width="100" height="26" rx="13" fill="rgba(239, 68, 68, 0.2)" stroke="#ef4444" stroke-width="1"/>
  <text x="520" y="42" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#fca5a5" text-anchor="middle">PETROL / CNG</text>
</svg>"""

# 2. Tata Nexon EV (Electric Teal & Lightning Glow)
NEXON_EV_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 240" width="100%" height="100%">
  <defs>
    <linearGradient id="bg_nexon" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#041a1a"/>
      <stop offset="100%" stop-color="#082f30"/>
    </linearGradient>
    <linearGradient id="car_teal" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0d9488"/>
      <stop offset="50%" stop-color="#14b8a6"/>
      <stop offset="100%" stop-color="#0f766e"/>
    </linearGradient>
    <linearGradient id="ev_glass" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#00f0ff" stop-opacity="0.8"/>
      <stop offset="100%" stop-color="#0891b2" stop-opacity="0.3"/>
    </linearGradient>
  </defs>
  <rect width="600" height="240" rx="16" fill="url(#bg_nexon)"/>
  <!-- Ground Shadow with EV Glow -->
  <ellipse cx="300" cy="205" rx="245" ry="14" fill="#00f0ff" opacity="0.15"/>
  <ellipse cx="300" cy="205" rx="240" ry="12" fill="#000" opacity="0.5"/>
  <!-- High Stance SUV Body -->
  <path d="M 85,160 L 98,135 Q 110,120 140,115 L 210,95 Q 260,65 370,65 L 440,75 Q 485,100 515,130 L 530,155 Q 535,168 515,172 L 450,172 A 38,38 0 0,0 374,172 L 210,172 A 38,38 0 0,0 134,172 L 90,172 Z" fill="url(#car_teal)"/>
  <!-- Dual-tone White Roof -->
  <path d="M 220,95 Q 260,65 370,65 L 435,75 L 435,85 L 220,102 Z" fill="#ffffff"/>
  <!-- Glasshouse -->
  <path d="M 225,100 L 365,72 L 430,80 L 475,120 L 225,120 Z" fill="url(#ev_glass)"/>
  <line x1="345" y1="73" x2="345" y2="120" stroke="#042f2e" stroke-width="4"/>
  <!-- EV Aero Blade Wheels -->
  <circle cx="172" cy="172" r="34" fill="#0f172a" stroke="#00f0ff" stroke-width="4"/>
  <circle cx="172" cy="172" r="16" fill="#14b8a6"/>
  <circle cx="412" cy="172" r="34" fill="#0f172a" stroke="#00f0ff" stroke-width="4"/>
  <circle cx="412" cy="172" r="16" fill="#14b8a6"/>
  <!-- LED DRL Signature -->
  <polygon points="515,135 530,148 518,155 505,145" fill="#00f0ff" opacity="0.95"/>
  <polygon points="90,135 84,148 92,155 98,145" fill="#f43f5e" opacity="0.9"/>
  <!-- Badging -->
  <text x="30" y="45" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="#2dd4bf">TATA NEXON EV</text>
  <text x="30" y="68" font-family="Arial, sans-serif" font-size="12" fill="#99f6e4">Clean Electric Powertrain · Smart Battery Fleet</text>
  <rect x="460" y="25" width="110" height="26" rx="13" fill="rgba(0, 240, 255, 0.15)" stroke="#00f0ff" stroke-width="1"/>
  <text x="515" y="42" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#67e8f9" text-anchor="middle">⚡ 100% ELECTRIC</text>
</svg>"""

# 3. Hyundai Creta (Premium Silver/Graphite SUV)
CRETA_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 240" width="100%" height="100%">
  <defs>
    <linearGradient id="bg_creta" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="100%" stop-color="#1e293b"/>
    </linearGradient>
    <linearGradient id="car_silver" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#94a3b8"/>
      <stop offset="50%" stop-color="#cbd5e1"/>
      <stop offset="100%" stop-color="#64748b"/>
    </linearGradient>
  </defs>
  <rect width="600" height="240" rx="16" fill="url(#bg_creta)"/>
  <ellipse cx="300" cy="205" rx="240" ry="14" fill="#000" opacity="0.5"/>
  <!-- Muscular SUV Profile -->
  <path d="M 85,160 L 95,130 Q 110,115 145,110 L 220,90 Q 270,62 385,62 L 450,72 Q 495,95 520,128 L 532,155 Q 536,168 518,172 L 452,172 A 38,38 0 0,0 376,172 L 210,172 A 38,38 0 0,0 134,172 L 88,172 Z" fill="url(#car_silver)"/>
  <!-- Roof Rails -->
  <line x1="260" y1="56" x2="420" y2="65" stroke="#e2e8f0" stroke-width="4" stroke-linecap="round"/>
  <!-- Glass -->
  <path d="M 230,95 L 375,68 L 440,78 L 480,118 L 230,118 Z" fill="#38bdf8" fill-opacity="0.6"/>
  <line x1="360" y1="70" x2="360" y2="118" stroke="#334155" stroke-width="4"/>
  <!-- Wheels -->
  <circle cx="172" cy="172" r="34" fill="#0f172a" stroke="#cbd5e1" stroke-width="5"/>
  <circle cx="172" cy="172" r="16" fill="#94a3b8"/>
  <circle cx="414" cy="172" r="34" fill="#0f172a" stroke="#cbd5e1" stroke-width="5"/>
  <circle cx="414" cy="172" r="16" fill="#94a3b8"/>
  <!-- Headlights -->
  <polygon points="515,130 530,145 520,152 505,142" fill="#fef08a" opacity="0.9"/>
  <!-- Badging -->
  <text x="30" y="45" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="#f1f5f9">HYUNDAI CRETA</text>
  <text x="30" y="68" font-family="Arial, sans-serif" font-size="12" fill="#94a3b8">Executive Compact SUV · India's Volume Leader</text>
  <rect x="470" y="25" width="100" height="26" rx="13" fill="rgba(148, 163, 184, 0.2)" stroke="#94a3b8" stroke-width="1"/>
  <text x="520" y="42" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#e2e8f0" text-anchor="middle">DIESEL / PETROL</text>
</svg>"""

# 4. Mahindra Thar (Rugged Amber 4x4 Offroader)
THAR_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 240" width="100%" height="100%">
  <defs>
    <linearGradient id="bg_thar" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1c1005"/>
      <stop offset="100%" stop-color="#2d1a08"/>
    </linearGradient>
    <linearGradient id="car_amber" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#d97706"/>
      <stop offset="50%" stop-color="#f59e0b"/>
      <stop offset="100%" stop-color="#b45309"/>
    </linearGradient>
  </defs>
  <rect width="600" height="240" rx="16" fill="url(#bg_thar)"/>
  <ellipse cx="300" cy="208" rx="240" ry="14" fill="#000" opacity="0.6"/>
  <!-- Boxy Rugged 4x4 Silhouette -->
  <path d="M 90,150 L 105,95 L 210,95 L 225,58 L 415,58 L 430,95 L 500,105 L 525,140 L 525,165 A 42,42 0 0,0 441,165 L 230,165 A 42,42 0 0,0 146,165 L 90,165 Z" fill="url(#car_amber)"/>
  <!-- Hardtop Black Roof -->
  <path d="M 225,58 L 415,58 L 415,95 L 210,95 Z" fill="#18181b"/>
  <!-- Upright Windows -->
  <rect x="235" y="65" width="80" height="30" fill="#38bdf8" fill-opacity="0.6" rx="4"/>
  <rect x="325" y="65" width="75" height="30" fill="#38bdf8" fill-opacity="0.6" rx="4"/>
  <!-- Chunky Offroad Wheels -->
  <circle cx="188" cy="165" r="38" fill="#18181b" stroke="#78350f" stroke-width="6"/>
  <circle cx="188" cy="165" r="18" fill="#d97706"/>
  <circle cx="483" cy="165" r="38" fill="#18181b" stroke="#78350f" stroke-width="6"/>
  <circle cx="483" cy="165" r="18" fill="#d97706"/>
  <!-- Badging -->
  <text x="30" y="45" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="#f59e0b">MAHINDRA THAR</text>
  <text x="30" y="68" font-family="Arial, sans-serif" font-size="12" fill="#fcd34d">Legendary 4x4 Offroader · High Resale Value</text>
  <rect x="470" y="25" width="100" height="26" rx="13" fill="rgba(245, 158, 11, 0.2)" stroke="#f59e0b" stroke-width="1"/>
  <text x="520" y="42" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#fde68a" text-anchor="middle">4x4 ALL-TERRAIN</text>
</svg>"""

# 5. Toyota Hyryder (Eco Emerald Hybrid)
HYRYDER_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 240" width="100%" height="100%">
  <defs>
    <linearGradient id="bg_hy" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#041a0e"/>
      <stop offset="100%" stop-color="#0a2e19"/>
    </linearGradient>
    <linearGradient id="car_green" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#059669"/>
      <stop offset="50%" stop-color="#10b981"/>
      <stop offset="100%" stop-color="#047857"/>
    </linearGradient>
  </defs>
  <rect width="600" height="240" rx="16" fill="url(#bg_hy)"/>
  <ellipse cx="300" cy="205" rx="240" ry="14" fill="#10b981" opacity="0.12"/>
  <ellipse cx="300" cy="205" rx="240" ry="12" fill="#000" opacity="0.5"/>
  <path d="M 85,160 L 98,135 Q 110,120 140,115 L 210,95 Q 260,65 370,65 L 440,75 Q 485,100 515,130 L 530,155 Q 535,168 515,172 L 450,172 A 38,38 0 0,0 374,172 L 210,172 A 38,38 0 0,0 134,172 L 90,172 Z" fill="url(#car_green)"/>
  <path d="M 225,100 L 365,72 L 430,80 L 475,120 L 225,120 Z" fill="#6ee7b7" fill-opacity="0.5"/>
  <circle cx="172" cy="172" r="34" fill="#0f172a" stroke="#10b981" stroke-width="4"/>
  <circle cx="172" cy="172" r="16" fill="#34d399"/>
  <circle cx="412" cy="172" r="34" fill="#0f172a" stroke="#10b981" stroke-width="4"/>
  <circle cx="412" cy="172" r="16" fill="#34d399"/>
  <text x="30" y="45" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="#34d399">TOYOTA URBAN HYRYDER</text>
  <text x="30" y="68" font-family="Arial, sans-serif" font-size="12" fill="#a7f3d0">Self-Charging Hybrid Electric Crossover</text>
  <rect x="460" y="25" width="110" height="26" rx="13" fill="rgba(16, 185, 129, 0.2)" stroke="#10b981" stroke-width="1"/>
  <text x="515" y="42" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#a7f3d0" text-anchor="middle">🌿 STRONG HYBRID</text>
</svg>"""

# 6. BMW 3-Series (Luxury Midnight Blue Sedan)
BMW_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 240" width="100%" height="100%">
  <defs>
    <linearGradient id="bg_bmw" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0b1329"/>
      <stop offset="100%" stop-color="#141f3d"/>
    </linearGradient>
    <linearGradient id="car_blue" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#1d4ed8"/>
      <stop offset="50%" stop-color="#2563eb"/>
      <stop offset="100%" stop-color="#1e40af"/>
    </linearGradient>
  </defs>
  <rect width="600" height="240" rx="16" fill="url(#bg_bmw)"/>
  <ellipse cx="300" cy="205" rx="245" ry="14" fill="#000" opacity="0.6"/>
  <!-- Aerodynamic Low Stance Luxury Sedan -->
  <path d="M 80,165 L 95,140 Q 110,130 145,125 L 225,100 Q 275,70 380,70 L 440,78 Q 485,105 525,135 L 545,160 Q 550,172 530,176 L 458,176 A 35,35 0 0,0 388,176 L 210,176 A 35,35 0 0,0 140,176 L 85,176 Z" fill="url(#car_blue)"/>
  <!-- Glasshouse -->
  <path d="M 235,100 L 375,76 L 435,84 L 485,125 L 235,125 Z" fill="#93c5fd" fill-opacity="0.6"/>
  <line x1="365" y1="78" x2="365" y2="125" stroke="#1e293b" stroke-width="4"/>
  <!-- M-Sport Rims -->
  <circle cx="175" cy="176" r="33" fill="#0f172a" stroke="#93c5fd" stroke-width="5"/>
  <circle cx="175" cy="176" r="15" fill="#60a5fa"/>
  <circle cx="423" cy="176" r="33" fill="#0f172a" stroke="#93c5fd" stroke-width="5"/>
  <circle cx="423" cy="176" r="15" fill="#60a5fa"/>
  <!-- Angel Eye DRL -->
  <polygon points="530,140 545,152 535,160 520,150" fill="#93c5fd" opacity="0.95"/>
  <!-- Badging -->
  <text x="30" y="45" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="#60a5fa">BMW 3-SERIES</text>
  <text x="30" y="68" font-family="Arial, sans-serif" font-size="12" fill="#93c5fd">Executive Sports Sedan · Luxury Tier</text>
  <rect x="470" y="25" width="100" height="26" rx="13" fill="rgba(37, 99, 235, 0.2)" stroke="#3b82f6" stroke-width="1"/>
  <text x="520" y="42" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#bfdbfe" text-anchor="middle">👑 LUXURY TIER</text>
</svg>"""

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "swift.svg").write_text(SWIFT_SVG, encoding="utf-8")
    (OUT_DIR / "nexon_ev.svg").write_text(NEXON_EV_SVG, encoding="utf-8")
    (OUT_DIR / "creta.svg").write_text(CRETA_SVG, encoding="utf-8")
    (OUT_DIR / "thar.svg").write_text(THAR_SVG, encoding="utf-8")
    (OUT_DIR / "hyryder.svg").write_text(HYRYDER_SVG, encoding="utf-8")
    (OUT_DIR / "bmw.svg").write_text(BMW_SVG, encoding="utf-8")
    print("Created 6 modern car graphics in data/images/")

if __name__ == "__main__":
    main()
