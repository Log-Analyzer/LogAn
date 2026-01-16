import datetime
import random
from dateutil import parser as god_parse
import json

logstr = 'My name is Pranjal'

def generate_timestamps(delta, num):
    end_ts = datetime.datetime.now()
    start_ts = end_ts - delta
    
    start_ts = int(start_ts.timestamp())
    end_ts = int(end_ts.timestamp())
    
    return random.sample(range(start_ts, end_ts), num)


def generate_data(num, delta):
    ts_list = sorted(generate_timestamps(delta, num))
    
    return [
        f'{datetime.datetime.fromtimestamp(ts)} {logstr}'
        for ts in ts_list
    ]

logs = []

delta = datetime.timedelta(days=1)
logs.extend(generate_data(100, delta))

delta = datetime.timedelta(days=2)
logs.extend(generate_data(100, delta))

delta = datetime.timedelta(days=7)
logs.extend(generate_data(100, delta))

delta = datetime.timedelta(days=14)
logs.extend(generate_data(100, delta))

delta = datetime.timedelta(days=30)
logs.extend(generate_data(100, delta))

all_ts = [god_parse.parse(log.strip(logstr)) for log in logs]
count_dict = {}

for ts in all_ts:
    date = str(ts.date())
    
    if date not in count_dict:
        count_dict[date] = 0
    
    count_dict[date] += 1

with open('stats.json', 'w') as writer:
    json.dump(count_dict, writer, indent=2, sort_keys=True)

with open('test.log', 'w') as writer:
    writer.write('\n'.join(logs))

    
    
