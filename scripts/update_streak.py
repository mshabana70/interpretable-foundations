import subprocess
import json
from datetime import datetime, timedelta

def get_commit_dates():
    result = subprocess.run(['git', 'log', '--format=%Y-%m-%d'], capture_output=True, text=True)
    dates = result.stdout.strip().split('\n')
    return sorted(list(set(dates)), reverse=True)

def calculate_streak(dates):
    if not dates:
         return 0
    
    today = datetime.utcnow().date()
    current_streak = 0
    check_date = today

    if dates[0] != str(today):
        if dates[0] == str(today - timedelta(days=1)):
            check_date = today - timedelta(days=1)
        else:
            return 0

    for date_str in dates:
        if date_str == str(check_date):
            current_streak += 1
            check_date -= timedelta(days=1)
        elif date_str > str(check_date):
            continue 
        else:
            break
            
    return current_streak

dates = get_commit_dates()
streak = calculate_streak(dates)

with open('streak.json', 'w') as f:
    json.dump({
        "schemaVersion": 1,
        "label": "Study Streak",
        "message": f"{streak} Days",
        "color": "orange"
    }, f)