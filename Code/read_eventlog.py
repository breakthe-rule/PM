import csv
import time
from datetime import datetime

# This function reads the eventlog of the process.
def read_eventlog(eventlog):
    
  # Read eventlog of form CaseID, ActivityID, Timestamp
  csvfile = open(eventlog, 'r')
  spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
  next(spamreader, None)  # skip the headers

  #used to track the current case ID
  lastcase = ''
  #accumulate characters representing event types for the current case.
  line = ''
  
  firstLine = True
  
  #store the sequences of event types (as strings) for each case.
  lines = []
  # store lists of time differences between consecutive events for each case.
  timeseqs = []
  #store lists of time differences from the case start time to each event.
  timeseqs2 = []
  #store lists of seconds past midnight for each event.
  timeseqs3 = []
  #store lists of the day of the week for each event (0 for Monday, 6 for Sunday).
  timeseqs4 = []
  
  #accumulate the time differences between consecutive events for the current case.
  times = []
  #accumulate the time differences from the case start time for the current case.
  times2 = []
  #accumulate the seconds past midnight for each event in the current case.
  times3 = []
  #accumulate the day of the week for each event in the current case. (0 for Monday, 6 for Sunday)
  times4 = []
  
  #number of unique cases processed.
  numlines = 0
  
  #store the timestamp of the first event of the current case.
  casestarttime = None
  #store the timestamp of the most recently processed event within the current case.
  lasteventtime = None
  
  # convert numerical event types to unique characters
  ascii_offset = 161 
  
  # iterate over each row in eventlog
  for row in spamreader:
      # timestamp of current activity
      t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
      
      # When new case begins
      if row[0]!=lastcase:
          
          # Set Case start time
          casestarttime = t
          # Since new event is started: last event time is equal to time of first event
          lasteventtime = t
          # Update last case
          lastcase = row[0]
          
          if not firstLine:
              
              # Store all the lists of previous case.
              lines.append(line)
              timeseqs.append(times)
              timeseqs2.append(times2)
              timeseqs3.append(times3)
              timeseqs4.append(times4)
              
          line = ''
          times = []
          times2 = []
          times3 = []
          times4 = []
          numlines+=1
          
      # sequence of activity in each case
      line+=chr(int(row[1])+ascii_offset)
      
      timesincelastevent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lasteventtime))
      timesincecasestart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(casestarttime))
      
      midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
      timesincemidnight = datetime.fromtimestamp(time.mktime(t))-midnight
      
      # Converting above time variables to seconds
      timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
      timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
      timediff3 = timesincemidnight.seconds #this leaves only time even occured after midnight
      timediff4 = datetime.fromtimestamp(time.mktime(t)).weekday() #day of the week
      
      # Appending above time variables in respective list
      times.append(timediff)
      times2.append(timediff2)
      times3.append(timediff3)
      times4.append(timediff4)
      
      lasteventtime = t
      firstLine = False

  # add last case
  lines.append(line)
  timeseqs.append(times)
  timeseqs2.append(times2)
  timeseqs3.append(times3)
  timeseqs4.append(times4)
  numlines+=1

  '''
  lines: list of activities in each case
  timeseqs: list of list of time difference between consecutive activities in each case
  timeseqs2: list of list of time difference between current activity and start of case
  timeseqs3: list of list of time difference between current activity and midnight
  timeseqs4: list of list of week day (0 monday - 6 sunday) when the activity is performed
  numlines: number of unique cases.
  '''
  return lines,timeseqs,timeseqs2,timeseqs3,timeseqs4,numlines