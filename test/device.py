import re
import subprocess
import sounddevice as sd
print(sd.query_devices()) 
print("\n##################################################################\n")
device_re = re.compile("Bus\s+(?P<bus>\d+)\s+Device\s+(?P<device>\d+).+ID\s(?P<id>\w+:\w+)\s(?P<tag>.+)$", re.I)
df = str(subprocess.check_output("lsusb"))
#print("df: ", df)
devices = []
for i in df.split('Bus'):
    if i:
        print("i: ", i)
        info = device_re.match(i)
        if info:
            dinfo = info.groupdict()
            dinfo['device'] = '/dev/bus/usb/%s/%s' % (dinfo.pop('bus'), dinfo.pop('device'))
            devices.append(dinfo)
print(devices)
