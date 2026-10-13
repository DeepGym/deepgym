set -euo pipefail
docs="$HOME/Documents"
rm -rf "$docs"
mkdir -p "$docs"
printf 'Q3 revenue summary\n' > "$docs/report.pdf"
printf 'milk, eggs, bread\n' > "$docs/notes.txt"
printf 'not really a photo\n' > "$docs/photo.jpg"
(setsid nohup thunar "$docs" >/tmp/deepgym_thunar.log 2>&1 &)
sleep 4
