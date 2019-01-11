#!/home/twixtrom/miniconda3/envs/research/bin/python

# Python script for advancing time by an arbitrary number of days

from datetime import datetime, timedelta
import sys

indate = sys.argv[1]
ndays = sys.argv[2]
nhours = sys.argv[3]

date1 = datetime.strptime(indate, '%Y%m%d%H')
date2 = date1 + timedelta(days=int(ndays), hours=int(nhours))
print(date2.strftime('%Y%m%d%H'))
