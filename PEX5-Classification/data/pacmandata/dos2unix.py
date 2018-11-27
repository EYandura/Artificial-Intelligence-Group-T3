#!/usr/bin/env python
"""\
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py <input> <output>
"""
import sys
import os

#if len(sys.argv[1:]) != 1:
#  sys.exit(__doc__)

files = [f for f in os.listdir('.\\old\\')]
for f in files:
   print(f)

   content = ''
   outsize = 0
   with open('.\\old\\' + f, 'rb') as infile:
     content = infile.read()
   with open(f, 'wb') as output:
     for line in content.splitlines():
       outsize += len(line) + 1
       output.write(line + str.encode('\n'))

   print("Done. Saved %s bytes." % (len(content)-outsize))
