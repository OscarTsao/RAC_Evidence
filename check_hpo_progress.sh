#!/bin/bash
# Quick HPO progress checker

python - <<'PY'
import sqlite3
from datetime import datetime, timedelta

conn = sqlite3.connect('optuna.db')
cursor = conn.cursor()

cursor.execute("""
    SELECT trial_id, state, datetime_start, datetime_complete
    FROM trials 
    WHERE study_id = 3
    ORDER BY trial_id
""")

all_trials = cursor.fetchall()
completed = [t for t in all_trials if str(t[1]) == "COMPLETE" or str(t[1]) == "1"]
running = [t for t in all_trials if str(t[1]) == "RUNNING" or str(t[1]) == "0"]

durations = []
if completed:
    for trial_id, state, dt_start, dt_complete in completed:
        if dt_start and dt_complete:
            try:
                if isinstance(dt_start, str):
                    start = datetime.strptime(dt_start, "%Y-%m-%d %H:%M:%S.%f")
                else:
                    start = datetime.fromtimestamp(dt_start / 1000.0)
                if isinstance(dt_complete, str):
                    complete = datetime.strptime(dt_complete, "%Y-%m-%d %H:%M:%S.%f")
                else:
                    complete = datetime.fromtimestamp(dt_complete / 1000.0)
                duration = (complete - start).total_seconds()
                durations.append(duration)
            except:
                pass

print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Progress: {len(completed)}/15 ({len(completed)/15*100:.1f}%)", end="", flush=True)

if durations and running:
    avg_duration = sum(durations) / len(durations)
    remaining_trials = 15 - len(completed) - 1
    if running[0][2]:
        try:
            if isinstance(running[0][2], str):
                start = datetime.strptime(running[0][2], "%Y-%m-%d %H:%M:%S.%f")
            else:
                start = datetime.fromtimestamp(running[0][2] / 1000.0)
            elapsed = (datetime.now() - start).total_seconds()
            current_remaining = max(0, avg_duration - elapsed)
        except:
            current_remaining = avg_duration
    else:
        current_remaining = avg_duration
    
    total_remaining_sec = current_remaining + (avg_duration * remaining_trials)
    estimated_finish = datetime.now() + timedelta(seconds=total_remaining_sec)
    print(f" | ETA: {estimated_finish.strftime('%H:%M')}", end="", flush=True)

conn.close()
PY
