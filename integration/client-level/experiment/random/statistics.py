#!/usr/bin/env python
import sys
import os
from operator import itemgetter

if __name__ == '__main__':
    # print(len(sys.argv))
    arguments = len(sys.argv) - 1
    if (arguments != 4):
        print ("We need 4 args \n   Usage: python statistics.py <path to replayed trace> <runtime> <late_rate> <slack_rate>")
        exit()
    runtime, late_rate, slack_rate = sys.argv[2], sys.argv[3], sys.argv[4]
    sorted_io = []
    readbandwidth = 0
    readlatency = 0
    totalread = 0
    writebandwidth = 0
    writelatency = 0
    totalwrite = 0
    last_io_time = -1

    with open(sys.argv[1]) as f:
        for line in f:
            tok = list(map(str.strip, line.split(",")))
            sorted_io.append([float(tok[0]),int(tok[1]),float(tok[2]),int(tok[3]),float(tok[4])])

    for io in sorted(sorted_io, key=itemgetter(0)):
        if (io[2] == 1): #read
            readbandwidth += (io[3]/1024) / (io[1]/1000000.0)
            readlatency += io[1]
            totalread += 1
        else: #write
            writebandwidth += (io[3]/1024) / (io[1]/1000000.0)
            writelatency += io[1]
            totalwrite += 1
        last_io_time = io[0]

    read_ratio = int((float(totalread)/(totalwrite + totalread)) * 100)

    print ("==========   Statistics   ==========")
    print ("Trace name   = " + os.path.basename(sys.argv[1]))
    print ("R:W ratio    = " + str(read_ratio) + ":" + str(100 - read_ratio))
    print ("Duration     = " + str(round(last_io_time / 1000, 2)) + " s")
    print ("#IO          = " + str(totalwrite + totalread))
    print ("#writes      = " + str(totalwrite))
    print ("#reads       = " + str(totalread))
    print ("IOPS         = " + "%.2f" % (float(totalwrite + totalread) / (last_io_time / 1000)))
    print ("Write IOPS   = " + "%.2f" % (float(totalwrite) / (last_io_time / 1000)))
    print ("Read IOPS    = " + "%.2f" % (float(totalread) / (last_io_time / 1000)))
    print ("Last time    = " + str(last_io_time))
    if totalwrite != 0:
        print ("Avg write throughput = " + "%.2f" % (writebandwidth / totalwrite / 1000) + " MB/s")
        print ("Avg write latency    = " + "%.2f" % (writelatency / totalwrite) + " us")
    if totalread != 0:
        print ("Avg read throughput  = " + "%.2f" % (readbandwidth / totalread / 1000) + " MB/s")
        print ("Avg read latency     = " + "%.2f" % (readlatency / totalread) + " us")
    print ("Interarrival avg     = " + "%.2f" % (last_io_time / (totalread + totalwrite)) + " ms")
    print ("Total run time       = " + str(round(float(runtime)/60,2)) + " mins")
    print ("Late rate            = " + str(late_rate)+ " %")
    print ("Slack rate           = " + str(slack_rate)+ " %")
    print ("====================================")
