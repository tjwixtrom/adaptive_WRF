#!/home/twixtrom/miniconda3/envs/analogue/bin/python
from datetime import datetime, timedelta


def increment_time(date1, days=0, hours=0):
    """
    Increment time from start by a specified number of days or hours

    Parameters:
        date1: datetime.datetime
        days: int, number of days to advance
        hours: int, number of hours to advance
    Returns: datetime.datetime, incremented time and date
    """
    return date1 + timedelta(days=days, hours=hours)


dir_store = '/lustre/scratch/twixtrom/adaptive_wrf_save/control_WRF2M/'
date = datetime(2016, 5, 1, 12)
end_date = datetime(2016, 5, 31, 12)

while date <= end_date:
    log = dir_store+date.strftime('%Y%m%d%H')+'/rslout_wrf_'+date.strftime('%Y%m%d%H')+'.log'
    logfile = open(log)
    last = logfile.readlines()[-1]
    find = last.find('SUCCESS COMPLETE WRF')
    if find == -1:
        print('WRF not complete '+str(date))
    date += timedelta(hours=24)
