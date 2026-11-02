// Generates header banner for MoistMachine / MoistViT (1200x400)
import { writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

const OUT = resolve("assets");
mkdirSync(OUT, { recursive: true });

const MONO = `'Consolas','JetBrains Mono',monospace`;
const DISP = `'Arial Black',Impact,sans-serif`;
const SANS = `'Segoe UI',system-ui,sans-serif`;

function moistBanner() {
  const W = 1200, H = 400;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#06121e"/>
    <stop offset="0.6" stop-color="#081e2e"/>
    <stop offset="1" stop-color="#042838"/>
  </linearGradient>
  <radialGradient id="glow" cx="0.8" cy="0.5" r="0.6">
    <stop offset="0" stop-color="#00d0ff" stop-opacity="0.30"/>
    <stop offset="0.6" stop-color="#38bdf8" stop-opacity="0.10"/>
    <stop offset="1" stop-color="#000000" stop-opacity="0"/>
  </radialGradient>
  <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#ffffff" stroke-opacity="0.04" stroke-width="1"/>
  </pattern>
  <style>
    .mono { font-family: ${MONO}; font-weight: 700; }
    .disp { font-family: ${DISP}; font-weight: 900; }
    .sans { font-family: ${SANS}; }
    .blink { animation: blink 1.2s steps(1,end) infinite; }
    @keyframes blink { 50% { opacity: 0; } }
  </style>
</defs>

<!-- BG -->
<rect width="${W}" height="${H}" fill="url(#bg)"/>
<rect width="${W}" height="${H}" fill="url(#grid)"/>
<ellipse cx="960" cy="200" rx="360" ry="240" fill="url(#glow)"/>

<!-- Top Status Line -->
<g transform="translate(60, 48)">
  <rect x="0" y="0" width="135" height="26" rx="6" fill="#00d0ff" fill-opacity="0.18" stroke="#00d0ff" stroke-width="1.5"/>
  <circle cx="14" cy="13" r="4" fill="#00d0ff" class="blink"/>
  <text x="26" y="17" class="mono" font-size="11" letter-spacing="1.5" fill="#38bdf8">FLOOD AI / EDGE</text>

  <text x="150" y="17" class="mono" font-size="11" letter-spacing="1.5" fill="#8a9ba8">// MULTI-HEAD SEMANTIC SEGMENTATION &#183; MOISTVIT</text>
</g>

<!-- Big Title -->
<g transform="translate(56, 146)">
  <text class="disp" font-size="86" letter-spacing="-1" fill="#00d0ff" opacity="0.3" x="4" y="4">MOISTMACHINE</text>
  <text class="disp" font-size="86" letter-spacing="-1" fill="#ffffff">MOISTMACHINE</text>
  <text class="mono" font-size="16" letter-spacing="3" fill="#38bdf8" x="650" y="-30">MoistViT</text>
</g>

<!-- Subtitle & Specs -->
<g transform="translate(60, 202)">
  <text class="sans" font-size="20" font-weight="600" fill="#e2e8f0">Multi-Head Vision Transformer &amp; Edge Flood Segmentation</text>
  <text class="sans" font-size="15" fill="#94a3b8" y="28">Low-Latency Aerial Flood Disaster Risk Assessment &amp; Semi-Supervised Edge Inference</text>

  <!-- Spec Pills -->
  <g transform="translate(0, 75)">
    <!-- Pill 1 -->
    <rect x="0" y="0" width="130" height="32" rx="7" fill="#ffffff" fill-opacity="0.06" stroke="#ffffff" stroke-opacity="0.15" stroke-width="1.2"/>
    <text x="65" y="20" text-anchor="middle" class="mono" font-size="11" fill="#f8fafc">MoistViT</text>

    <!-- Pill 2 -->
    <rect x="142" y="0" width="165" height="32" rx="7" fill="#ffffff" fill-opacity="0.06" stroke="#ffffff" stroke-opacity="0.15" stroke-width="1.2"/>
    <text x="224" y="20" text-anchor="middle" class="mono" font-size="11" fill="#f8fafc">MobileNetV3 U-Net</text>

    <!-- Pill 3 -->
    <rect x="319" y="0" width="175" height="32" rx="7" fill="#ffffff" fill-opacity="0.06" stroke="#ffffff" stroke-opacity="0.15" stroke-width="1.2"/>
    <text x="406" y="20" text-anchor="middle" class="mono" font-size="11" fill="#f8fafc">Teacher-Student</text>

    <!-- Pill 4 -->
    <rect x="506" y="0" width="170" height="32" rx="7" fill="#00d0ff" fill-opacity="0.15" stroke="#00d0ff" stroke-width="1.2"/>
    <text x="591" y="20" text-anchor="middle" class="mono" font-size="11" fill="#38bdf8">3-Head Output</text>
  </g>

  <!-- Metric highlight -->
  <text x="0" y="145" class="mono" font-size="12" letter-spacing="1.5" fill="#64748b">ADAPTED ON PAKISTAN FLOOD AERIAL DATA &#183; 512/768 ONNX &amp; TORCHSCRIPT LITE</text>
</g>

<!-- Right Side: Graphic Visualizer -->
<g transform="translate(860, 75)">
  <!-- Outer Box -->
  <rect x="0" y="0" width="280" height="260" rx="14" fill="#061826" stroke="#ffffff" stroke-opacity="0.12" stroke-width="2"/>
  
  <!-- Wave Simulation -->
  <path d="M 20 80 Q 50 65 80 80 T 140 80 T 200 80 T 260 80" fill="none" stroke="#00d0ff" stroke-width="2.5"/>
  <path d="M 20 100 Q 50 85 80 100 T 140 100 T 200 100 T 260 100" fill="none" stroke="#38bdf8" stroke-width="1.8" opacity="0.6"/>

  <!-- Water mask simulation -->
  <path d="M 30 190 Q 70 140 120 155 Q 180 170 210 135 Q 240 150 250 190 Z" fill="#00d0ff" fill-opacity="0.25" stroke="#00d0ff" stroke-width="2"/>
  
  <!-- Floating tag -->
  <rect x="40" y="130" width="95" height="22" rx="4" fill="#00d0ff"/>
  <text x="48" y="145" class="mono" font-size="10" font-weight="700" fill="#061826">FLOOD: +0.42m</text>

  <!-- Crosshairs -->
  <path d="M 140 20 V 40 M 140 220 V 240 M 20 130 H 40 M 240 130 H 260" stroke="#38bdf8" stroke-width="1.5" opacity="0.7"/>

  <!-- Footer Tag -->
  <rect x="15" y="215" width="250" height="30" rx="6" fill="#000000" fill-opacity="0.5"/>
  <text x="25" y="234" class="mono" font-size="10" fill="#94a3b8">LATENCY: <tspan fill="#34d399">12ms</tspan> &#183; RESOLUTION: <tspan fill="#00d0ff">768x768</tspan></text>
</g>

<!-- Accent Lines -->
<rect x="0" y="396" width="${W}" height="4" fill="#00d0ff"/>
</svg>`;
}

writeFileSync(resolve(OUT, "moistmachine-banner.svg"), moistBanner(), "utf8");
console.log("MoistMachine banner created!");
