import csv
import time
from datetime import datetime

def read_eventlog(eventlog):
  csvfile = open(eventlog, 'r')
  spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
  next(spamreader, None)  # skip the headers

  lastcase = ''
  line = ''
  firstLine = True
  lines = []
  timeseqs = []
  timeseqs2 = []
  timeseqs3 = []
  timeseqs4 = []
  times = []
  times2 = []
  times3 = []
  times4 = []
  numlines = 0
  casestarttime = None
  lasteventtime = None
  ascii_offset = 161
  for row in spamreader:
      t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
      if row[0]!=lastcase:
          casestarttime = t
          lasteventtime = t
          lastcase = row[0]
          if not firstLine:
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
      line+=chr(int(row[1])+ascii_offset)
      timesincelastevent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lasteventtime))
      timesincecasestart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(casestarttime))
      midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
      timesincemidnight = datetime.fromtimestamp(time.mktime(t))-midnight
      timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
      timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
      timediff3 = timesincemidnight.seconds #this leaves only time even occured after midnight
      timediff4 = datetime.fromtimestamp(time.mktime(t)).weekday() #day of the week
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

  return lines,timeseqs,timeseqs2,timeseqs3,timeseqs4,numlines