# ECG Arrhythmia Monitor — UI Design Specification

## 1. Design Goal

Create a professional, clinical, easy-to-scan dashboard for an ECG arrhythmia monitoring system. Preserve the existing information architecture and all displayed data, but improve hierarchy, legibility, spacing, and visual balance.

The interface should feel like a modern medical monitoring product—not a generic AI landing page. Use a **light-first palette** with cool whites, pale blue-gray surfaces, dark navy text, and restrained teal/green accents. Keep the ECG waveform prominent without making the whole interface neon or overly dark.

> **Important:** This UI is a software monitoring concept, not a certified medical device. Predictions and confidence values must not be presented as a definitive diagnosis.

## 2. Visual Direction

- **Theme:** Light clinical dashboard
- **Style:** Minimal, precise, data-dense, card-based
- **Mood:** Calm, trustworthy, technical, readable
- **Primary canvas:** Soft blue-gray
- **Cards:** White with thin cool-gray borders
- **Accent:** Teal/emerald for live signal and positive status
- **Warning:** Amber for attention states
- **Typography:** Inter, with system sans-serif fallbacks
- **Icons:** Lucide-style outline icons
- **Charts:** Fine gridlines, clear axes, high-contrast waveform
- **Avoid:** Full-page gradients, excessive glow, large decorative illustrations, too many saturated colors, and relying on color alone to convey status.

## 3. Color Tokens

| Token | Hex | Intended use |
|---|---|---|
| `page-bg` | `#F3F6FA` | Overall page background |
| `surface` | `#FFFFFF` | Main panels and cards |
| `surface-subtle` | `#F8FAFC` | Nested cards and chart surface |
| `surface-selected` | `#EAF5F4` | Selected/active subtle background |
| `text-primary` | `#172B4D` | Main headings and important values |
| `text-secondary` | `#52647A` | Labels and supporting information |
| `text-muted` | `#718096` | Timestamps, helper text, axis labels |
| `border` | `#DCE4ED` | Card outlines and dividers |
| `gridline` | `#E5ECF3` | ECG chart grid |
| `primary` | `#0F9F8F` | Main teal accent and waveform |
| `success` | `#16865C` | Connected/ready/success state |
| `success-bg` | `#E7F7EF` | Soft background for positive badges |
| `info` | `#3478D4` | Secondary information and links |
| `info-bg` | `#EAF2FF` | Soft blue status badges |
| `warning` | `#B77909` | Waiting, caution, or attention state |
| `warning-bg` | `#FFF4D6` | Soft amber status background |
| `danger` | `#C2414B` | Errors or critical system faults |
| `danger-bg` | `#FDEBEC` | Soft red status background |
| `waveform` | `#079B88` | ECG trace on a light chart |
| `focus-ring` | `#6AA7F8` | Keyboard focus outline |

### Color usage rules

1. Use teal for the ECG trace and active data, not as a background for large areas.
2. Use green for positive states such as **Leads Connected** and **CNN-BiLSTM Ready**.
3. Use amber for **Waiting for stream** or other non-critical attention states.
4. Use red only for genuine errors or critical warnings.
5. Ensure text and controls remain legible against their backgrounds. Never communicate status through color alone—include text and/or an icon.

## 4. Typography

**Font family:** `Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif`

| Element | Size | Weight | Notes |
|---|---:|---:|---|
| Product name | 18–20 px | 650–700 | Header title |
| Page/section heading | 15–17 px | 600–650 | Clear hierarchy |
| Card heading | 13–15 px | 600 | Keep concise |
| Main prediction | 20–24 px | 700 | Most prominent result |
| Metric value | 16–19 px | 650–700 | Use tabular numerals |
| Body text | 13–14 px | 400 | Comfortable readability |
| Field label | 11–12 px | 500 | Uppercase optional; avoid excessive tracking |
| Chart axis | 11–12 px | 400–500 | Muted but readable |
| Small metadata | 11–12 px | 400 | Do not make essential status too small |

Use a line height of **1.45–1.6** for normal text. Use `font-variant-numeric: tabular-nums` for values that update frequently, such as ADC readings, sample counts, confidence, BPM, and uptime.

## 5. Layout and Page Structure

### Desktop layout

Use a responsive dashboard grid:

- **Main workspace:** roughly 70–75% of the available width
- **Diagnostics sidebar:** roughly 25–30%
- **Outer padding:** 20–24 px
- **Card gap:** 14–18 px
- **Card padding:** 16–20 px
- **Card radius:** 12–16 px
- **Control radius:** 8–10 px
- **Border:** 1 px solid `#DCE4ED`
- **Shadow:** subtle only, such as `0 2px 8px rgba(23, 43, 77, 0.04)`

### Recommended structure

1. **Top header**
   - ECG icon and “ECG Arrhythmia Monitor”
   - Hardware mode: “Single-Lead II”
   - Sampling rate: “250 Hz”
   - Status badges: “Mock Mode (No HW)”, “Leads Connected”, “CNN-BiLSTM Ready”
   - Keep status badges compact and consistent.

2. **Capture / inference status banner**
   - “10s Window Captured — Continuous Live Inference”
   - “2500 Samples Ready · Updating Every 2s”
   - Slim progress bar with teal fill.
   - In mock mode, clearly indicate that the signal is simulated.

3. **ECG waveform panel**
   - Title: “ECG Signal”
   - Large plot area with visible axes, time labels, and light gridlines.
   - Preserve the waveform’s relative shape and existing axis values.
   - Use a white or very pale blue plot background with a teal waveform.
   - Keep chart labels and gridlines subtle, not invisible.

4. **Diagnostic inference panel**
   - Section label: “Diagnostic Inference”
   - Nested result card:
     - “Current Rhythm”
     - “Normal Sinus Rhythm”
     - “Model Confidence: 97.0%”
     - “Window: 10s · Stride: 2s”
   - Use a soft teal background tint for the result, not a bright solid fill.

5. **Class distribution**
   - Preserve all classes and percentages:
     - Normal Sinus Rhythm — 97.0%
     - Atrial Fibrillation — 0.0%
     - Supraventricular Arrhythmia — 2.6%
     - Ventricular Arrhythmia — 0.0%
     - Noise / Artifact — 0.4%
   - Use horizontal progress bars with a light track and clearly visible values.
   - Ensure the small 0.4% and 2.6% values remain readable even when the bars are tiny.

6. **System health diagnostics**
   - Serial Source: Mock Generator
   - Data Transmission: Waiting for stream
   - Total Samples: 3,499
   - Inference Model: `best_ecg_cnn_bilstm.pt`
   - Server Uptime: 13s
   - Align labels and values in two columns.
   - Make “Waiting for stream” amber; distinguish the mock source from a real hardware stream.

7. **Bottom metric strip**
   - Current ADC: 2082
   - Peak-to-Peak (VPP): 690 ADC
   - Signal Baseline: 2076 ADC
   - Estimated HR: 67 BPM
   - Window Buffer: 2500 / 2500
   - Use compact white metric cards or one divided strip. Use icons sparingly.

8. **Footer**
   - “AI Arrhythmia Detection System · Lead II · AD8232 Hardware Pipeline”
   - Timestamp: “9:01:24 PM”
   - Keep the footer muted and visually quiet.

## 6. Component Details

### Status badges
- Use pill-shaped badges with a pale tinted background and border.
- Include a small dot or simple icon and explicit text.
- Example: pale amber for “Mock Mode (No HW)”; pale green for “Leads Connected” and “CNN-BiLSTM Ready”.
- The mock-mode badge must not imply physical hardware is connected.

### ECG chart
- Chart surface: `#FBFDFF`
- Grid: `#E5ECF3`, with slightly stronger major lines if needed.
- Waveform: `#079B88`, approximately 1.5–2 px stroke.
- Axis text: `#718096`
- Use a restrained glow only if it improves clarity; default to no glow.
- Do not smooth or fabricate physiological data in a way that could be mistaken for a real reading.

### Confidence bar
- Track: `#E8EEF4`
- Fill: `#16865C`
- Display the percentage as text, independently of the bar.

### Class distribution bars
- Track: `#E8EEF4`
- Normal: teal/green
- Atrial fibrillation: blue or neutral when zero
- Supraventricular arrhythmia: amber
- Ventricular arrhythmia: neutral when zero
- Noise/artifact: muted amber or slate
- Always show the exact percentage beside each label.

### Metric cards
- Labels are secondary text; values are primary text.
- Use tabular numerals so rapidly changing values do not jump horizontally.
- Keep the cards equal in height and align values consistently.

## 7. Interaction and Responsive Behavior

- **Live updates:** update values without shifting the page layout.
- **Loading state:** show an explicit loading label and progress indicator.
- **No hardware / mock mode:** show a persistent, clear “Mock Mode (No HW)” state.
- **Disconnected leads:** use a warning badge and descriptive text.
- **Serial stream unavailable:** show “Waiting for stream” rather than implying live acquisition.
- **Inference failure:** show a clear error state and retain the last successful result only if it is marked as stale.
- **Keyboard access:** all interactive controls should be focusable with a visible focus ring.
- **Responsive layout:**
  - Wide desktop: waveform and diagnostics side by side.
  - Tablet: reduce gaps and stack metric cards into two or three columns.
  - Mobile: stack waveform, inference, distribution, diagnostics, and metrics vertically.
- Avoid horizontal scrolling on common laptop and mobile widths.

## 8. Accessibility and Medical-UI Safety

- Maintain readable contrast for body text and status labels.
- Do not use color as the only signal for success, warning, or failure.
- Keep chart axes, units, and time window visible.
- Distinguish mock data from measured hardware data in the UI.
- Display a disclaimer near the prediction if this is used outside a prototype:
  “For research and demonstration only. This output is not a medical diagnosis.”
- Do not imply the model is clinically validated unless that has been independently established.
- Treat “Model Confidence” as the model’s reported confidence, not as a guarantee of diagnostic correctness.

## 9. General Design Principles

- **Hierarchy:** make the waveform and current rhythm the two most prominent areas.
- **Consistency:** use the same spacing scale, typography, border, and badge styles throughout.
- **Whitespace:** separate groups enough to reduce cognitive load while preserving useful data density.
- **Restraint:** reserve saturated colors for meaningful states and the waveform.
- **Precision:** align labels, values, and chart axes; keep units visible.
- **Trust:** show the source of data and distinguish simulated data from hardware input.
- **Scalability:** define colors, spacing, typography, and radii as reusable design tokens.
- **Maintainability:** implement repeated visual patterns as reusable components rather than styling each instance independently.

## 10. Suggested CSS Variables

```css
:root {
  color-scheme: light;
  --page-bg: #F3F6FA;
  --surface: #FFFFFF;
  --surface-subtle: #F8FAFC;
  --surface-selected: #EAF5F4;
  --text-primary: #172B4D;
  --text-secondary: #52647A;
  --text-muted: #718096;
  --border: #DCE4ED;
  --gridline: #E5ECF3;
  --primary: #0F9F8F;
  --success: #16865C;
  --success-bg: #E7F7EF;
  --info: #3478D4;
  --info-bg: #EAF2FF;
  --warning: #B77909;
  --warning-bg: #FFF4D6;
  --danger: #C2414B;
  --danger-bg: #FDEBEC;
  --waveform: #079B88;
  --focus-ring: #6AA7F8;
  --radius-card: 14px;
  --radius-control: 9px;
  --shadow-card: 0 2px 8px rgba(23, 43, 77, 0.04);
}
```

## 11. Content Preservation Checklist

- [ ] ECG Arrhythmia Monitor
- [ ] Single-Lead II · 250 Hz
- [ ] Mock Mode (No HW)
- [ ] Leads Connected
- [ ] CNN-BiLSTM Ready
- [ ] 10s Window Captured — Continuous Live Inference
- [ ] 2500 Samples Ready · Updating Every 2s
- [ ] Current Rhythm: Normal Sinus Rhythm
- [ ] Model Confidence: 97.0%
- [ ] Class distribution and percentages preserved
- [ ] System health diagnostics preserved
- [ ] Current ADC: 2082
- [ ] Peak-to-Peak (VPP): 690 ADC
- [ ] Signal Baseline: 2076 ADC
- [ ] Estimated HR: 67 BPM
- [ ] Window Buffer: 2500 / 2500
- [ ] Hardware pipeline footer and timestamp preserved
