import os

def fix_file(path):
    if not os.path.exists(path): return
    with open(path, 'r') as f:
        content = f.read()
    
    # Convert single newlines to double newlines if not already double
    # This is a brute force way to ensure everything has blank lines between them
    lines = content.split('\n')
    fixed_lines = []
    for line in lines:
        fixed_lines.append(line)
        if line.strip() != "":
            fixed_lines.append("") # Adds a blank line
            
    # Write back
    with open(path, 'w') as f:
        f.write('\n'.join(fixed_lines))

fix_file("01_JOURNEY_LOG.md")
fix_file("02_EXPERIMENT_TRACKER.md")
print("Markdown formatting fixed.")

# Also append the scientific lock
with open("02_EXPERIMENT_TRACKER.md", "a") as f:
    f.write("\n| `006` | 2026-07-28 12:28 | Phase 6: Evaluation & Kaggle Packaging | N/A | Win Rate & TarGz | [ACTIVE LOCK] |\n")
