"""Quick check of SPRSound dataset structure and CAS distribution."""
import glob, json, os

root = "data/SPRSound/SPRSound-main/SPRSound-main"

# Check Detection
det_jsons = glob.glob(os.path.join(root, "Detection", "**", "*.json"), recursive=True)
det_wavs = glob.glob(os.path.join(root, "Detection", "**", "*.wav"), recursive=True)

print(f"Detection JSONs: {len(det_jsons)}")
print(f"Detection WAVs: {len(det_wavs)}")

total_events = 0
cas_events = 0
no_cas = 0
files_with_cas = 0
type_counts = {}

for jf in det_jsons:
    with open(jf, "r", encoding="utf-8") as f:
        d = json.load(f)
    evts = d.get("event_annotation", [])
    has_cas = False
    for e in evts:
        t = e.get("type", "")
        type_counts[t] = type_counts.get(t, 0) + 1
        tl = t.lower()
        if "wheeze" in tl or "rhonchi" in tl:
            cas_events += 1
            has_cas = True
        else:
            no_cas += 1
        total_events += 1
    if has_cas:
        files_with_cas += 1

print(f"\nFiles with CAS events: {files_with_cas}/{len(det_jsons)}")
print(f"Total events: {total_events}")
print(f"CAS events: {cas_events}")
print(f"non-CAS events: {no_cas}")
print(f"\nEvent types:")
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t}: {c}")

# Check if WAV paths are aligned with JSON
sample = det_jsons[0]
wav_candidate = sample.replace(".json", ".wav").replace("_json", "_wav")
print(f"\nSample JSON: {sample}")
print(f"Expected WAV: {wav_candidate}")
print(f"WAV exists: {os.path.exists(wav_candidate)}")

# Check subject ID extraction from filename
basename = os.path.basename(sample)
parts = basename.replace(".json", "").split("_")
print(f"\nFilename parts: {parts}")
print(f"Possible patient ID: {parts[0]}")
