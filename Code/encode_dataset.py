import pandas as pd
import csv 
from datetime import datetime
from time import time
data = pd.read_csv("data/BPI20.csv")

filename = "data/BPI20_tanay.csv"
csvfile = open(filename, 'w', newline='')
csvwriter = csv.writer(csvfile)
csvwriter.writerow(["CaseID","ActivityID","CompleteTimestamp"])

caseid = 0; last_case = 0
activityid = {}; activity_max = 1
start = time()
for i,row in data.iterrows():
        
    case = row["case:Project"]
    activity = row["concept:name"]
    timestamp = row["time:timestamp"]
    
    if last_case!=case:
        last_case = case
        caseid+=1
        
    if activityid.get(activity,0):
        activity = activityid[activity]
    else:
        activityid[activity] = activity_max
        activity = activity_max
        activity_max += 1
    
    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S%z")
    timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
    
    line = [caseid,activity,timestamp]
    csvwriter.writerow(line)
    
csvfile.close()
print(f"CSV file '{filename}' has been created successfully.")

print(activityid)